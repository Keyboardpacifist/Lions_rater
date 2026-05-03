"""Per-team Volume Amplification Alpha — projected 2026 pass attempts.

Output: data/scheme/volume_alpha.parquet

The third of four fantasy alpha factors. The Usage Autopsy is
vacancy-driven; QB Trajectory is QB-driven; this module is **team-
volume driven**: receivers gain when their team simply throws the
ball more total.

Three signals drive 2026 volume:

  1. Defensive regression (the main signal)
     Teams whose 2025 defense was elite (low PPG allowed) project
     to regress toward league mean → opponents score more → team
     trails more → team passes more. Conversely, bad defenses
     regress UP, lead more, pass less.

  2. Pace (plays per game — sticky YoY, ~0.6 correlation)
     Up-tempo teams stay up-tempo. Sets the volume baseline.

  3. Pass-rate continuity (pass attempts / total plays — sticky)
     Pass-happy schemes stay pass-happy. Multiplies the play volume
     into pass volume.

Schema
------
    team, prior_season,
    pass_attempts_2025, plays_2025, pass_rate_2025,
    points_allowed_2025, league_avg_points_allowed,
    def_regression_pts, def_volume_bump,
    proj_pass_attempts_2026, attempts_delta,
    volume_label, rationale

volume_label:
    🚀 RISING     — projected attempts delta >= +20
    ➡️ STABLE     — within +/- 20
    ⬇️ DECLINING  — delta <= -20

Methodology notes
-----------------
- We use TOTAL pass attempts (regular season only, 17 games), not
  per-game, so the delta is in absolute attempts that translate
  directly into target-share math downstream.
- def_regression_pts = (league_avg_PPG_allowed_2025 -
                          team_PPG_allowed_2025), so positive means
  elite defense expected to regress.
- def_volume_bump = def_regression_pts × 0.6 attempts/game × 17 games.
  The 0.6-attempts-per-1-PPG-shift coefficient is conservative;
  empirically the relationship is closer to 1 attempt/PPG, but we
  keep it tight for v1 to avoid over-projection.
"""
from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "scheme"
OUT = OUT_DIR / "volume_alpha.parquet"

PRIOR_SEASON = 2025
ATTEMPTS_PER_PPG = 0.6      # attempts/game shift per +1 PPG
                              # mean-reversion in defensive scoring
GAMES_PER_SEASON = 17
DEF_REVERSION_SHARE = 0.6   # empirically defensive PPG allowed has
                              # YoY correlation ≈ 0.4 → ~60% of the
                              # gap to league mean reverts each year.
                              # Without this we'd over-project elite
                              # defenses (e.g. SEA at +60 attempts).


def main() -> None:
    print(f"→ loading PBP {PRIOR_SEASON}...")
    pbp = nfl.load_pbp([PRIOR_SEASON]).to_pandas()
    pbp = pbp[pbp["season_type"] == "REG"]

    # ── Per-team offensive volume ─────────────────────────────────
    # Pass attempts: any play tagged as a pass (excluding 2pt etc)
    pass_plays = pbp[
        (pbp["play_type"] == "pass") & pbp["posteam"].notna()
    ]
    rush_plays = pbp[
        (pbp["play_type"] == "run") & pbp["posteam"].notna()
    ]

    pass_by_team = (
        pass_plays.groupby("posteam").size()
                    .rename("pass_attempts").reset_index()
    )
    rush_by_team = (
        rush_plays.groupby("posteam").size()
                    .rename("rush_attempts").reset_index()
    )
    vol = pass_by_team.merge(rush_by_team, on="posteam", how="outer")
    vol = vol.rename(columns={"posteam": "team"})
    vol["plays"] = vol["pass_attempts"] + vol["rush_attempts"]
    vol["pass_rate"] = (vol["pass_attempts"] / vol["plays"]).round(3)
    print(f"  team-volume rows: {len(vol)}")

    # ── Per-team defensive points allowed ─────────────────────────
    # Points allowed = points scored against this team's defense.
    # Cleanest source: each game has total_home_score and
    # total_away_score at end-of-game. Build per-game team rows.
    eog = (
        pbp.sort_values("play_id")
            .drop_duplicates(subset="game_id", keep="last")
            [["game_id", "home_team", "away_team",
              "total_home_score", "total_away_score"]]
    )
    eog["home_pa"] = eog["total_away_score"]   # pts allowed by home D
    eog["away_pa"] = eog["total_home_score"]   # pts allowed by away D

    home_rows = eog.rename(columns={"home_team": "team",
                                          "home_pa": "points_allowed"})[
        ["team", "points_allowed"]]
    away_rows = eog.rename(columns={"away_team": "team",
                                          "away_pa": "points_allowed"})[
        ["team", "points_allowed"]]
    pa = pd.concat([home_rows, away_rows], ignore_index=True)
    pa = (pa.groupby("team", as_index=False)
              .agg(games=("points_allowed", "count"),
                   total_points_allowed=("points_allowed", "sum")))
    pa["points_allowed_per_game"] = (
        pa["total_points_allowed"] / pa["games"].clip(lower=1)
    ).round(2)
    league_avg_ppg = float(
        pa["points_allowed_per_game"].mean().round(2))
    print(f"  league avg PPG allowed (2025): {league_avg_ppg:.2f}")

    # ── Merge volume + defense ────────────────────────────────────
    df = vol.merge(pa, on="team", how="left")
    df["def_regression_pts"] = (
        league_avg_ppg - df["points_allowed_per_game"]
    ).round(2)

    # Defensive volume bump (in total pass attempts over 17 games).
    # Shrinks the raw mean-distance by DEF_REVERSION_SHARE so we
    # don't assume 100% reversion — some teams stay elite/bad.
    df["def_volume_bump"] = (
        df["def_regression_pts"]
        * DEF_REVERSION_SHARE
        * ATTEMPTS_PER_PPG
        * GAMES_PER_SEASON
    ).round(1)

    df["proj_pass_attempts_2026"] = (
        df["pass_attempts"] + df["def_volume_bump"]
    ).round(0).astype(int)
    df["attempts_delta"] = (
        df["proj_pass_attempts_2026"] - df["pass_attempts"]
    ).astype(int)

    def _label(d: int) -> str:
        if d >= 20:
            return "🚀 RISING"
        if d <= -20:
            return "⬇️ DECLINING"
        return "➡️ STABLE"

    df["volume_label"] = df["attempts_delta"].apply(_label)

    def _rationale(row: pd.Series) -> str:
        bits = []
        d = row["def_regression_pts"]
        if d >= 2:
            bits.append(
                f"elite D ({row['points_allowed_per_game']:.1f} PPG "
                f"vs {league_avg_ppg:.1f} avg) regresses → "
                f"more trailing → +{row['def_volume_bump']:.0f} att"
            )
        elif d <= -2:
            bits.append(
                f"struggling D ({row['points_allowed_per_game']:.1f} "
                f"PPG vs {league_avg_ppg:.1f} avg) regresses up → "
                f"lead more → {row['def_volume_bump']:.0f} att"
            )
        else:
            bits.append("defense near league average — minimal "
                          "regression-driven volume shift")
        return "; ".join(bits)

    df["rationale"] = df.apply(_rationale, axis=1)
    df = df.rename(columns={
        "pass_attempts": "pass_attempts_2025",
        "plays": "plays_2025",
        "pass_rate": "pass_rate_2025",
        "points_allowed_per_game": "points_allowed_2025",
    })
    df["league_avg_points_allowed"] = league_avg_ppg
    df["prior_season"] = PRIOR_SEASON

    out = df[[
        "team", "prior_season",
        "pass_attempts_2025", "plays_2025", "pass_rate_2025",
        "points_allowed_2025", "league_avg_points_allowed",
        "def_regression_pts", "def_volume_bump",
        "proj_pass_attempts_2026", "attempts_delta",
        "volume_label", "rationale",
    ]].sort_values("attempts_delta", ascending=False).reset_index(
        drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()

    print("=== VOLUME ALPHA — projected 2026 pass-attempt deltas ===")
    show = out[[
        "team", "pass_attempts_2025", "points_allowed_2025",
        "def_regression_pts", "attempts_delta",
        "proj_pass_attempts_2026", "volume_label", "rationale",
    ]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()

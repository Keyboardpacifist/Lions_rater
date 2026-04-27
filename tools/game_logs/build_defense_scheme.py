#!/usr/bin/env python3
"""
Per-game and per-season defensive scheme summaries — NFL.

Joins play-by-play with participation (personnel + coverage labels),
filters to scrimmage plays, and aggregates by (defense_team, season,
week) into a clean scheme profile. Lets us answer questions like:

  • How did MIN play Gibbs in week 7 vs their season norm?
    → Avg box, blitz rate, man/zone split, coverage shell mix, etc.
  • What's Aaron Glenn's defensive identity in DET 2024?
    → Aggregate season profile + percentile vs league.

Inputs (must already exist):
  data/games/nfl_pbp.parquet
  data/games/nfl_participation.parquet

Outputs:
  data/games/nfl_defense_game_scheme.parquet     — per (def, season, week)
  data/games/nfl_defense_season_scheme.parquet   — per (def, season)

Notes on data fidelity:
  • Box / blitz / pressure / personnel: 2016+ at 100% per-play coverage
  • Man-vs-zone + coverage shell (Cover-1, Cover-2, etc.): 2018+ only,
    and ~40–50% of plays per game are labeled (every pass dropback is
    typically tagged; some get null/blown — those are excluded from
    rate calcs)

Usage:
    python tools/game_logs/build_defense_scheme.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PBP_PATH = REPO_ROOT / "data" / "games" / "nfl_pbp.parquet"
PART_PATH = REPO_ROOT / "data" / "games" / "nfl_participation.parquet"
GAME_OUT = REPO_ROOT / "data" / "games" / "nfl_defense_game_scheme.parquet"
SEASON_OUT = REPO_ROOT / "data" / "games" / "nfl_defense_season_scheme.parquet"


_SECONDARY_POSITIONS = {"CB", "DB", "FS", "SS", "S", "NB"}


def _classify_personnel(d: str | None) -> str | None:
    """Bucket defense_personnel into base/nickel/dime/etc by counting
    secondary players. Two formats appear in the source feed:
      • Abstract: '4 DL, 2 LB, 5 DB'
      • Granular: '2 CB, 2 DE, 1 DT, 1 FS, 3 ILB, 1 NT, 1 SS'
    Both forms decompose the same way — sum every position whose
    abbreviation belongs to the secondary."""
    if d is None or not d:
        return None
    db = 0
    for chunk in str(d).split(","):
        chunk = chunk.strip()
        parts = chunk.split(" ", 1)
        if len(parts) != 2:
            continue
        try:
            n = int(parts[0])
        except ValueError:
            continue
        if parts[1].strip() in _SECONDARY_POSITIONS:
            db += n
    if db == 4:
        return "Base"
    if db == 5:
        return "Nickel"
    if db == 6:
        return "Dime"
    if db >= 7:
        return "Quarter"
    return "Other"


def main():
    if not PBP_PATH.exists():
        raise SystemExit(f"Missing {PBP_PATH}. Run `make game-logs-pbp`.")
    if not PART_PATH.exists():
        raise SystemExit(
            f"Missing {PART_PATH}. Run `make game-logs-participation`.")

    print("Loading inputs…")
    pbp = pl.read_parquet(PBP_PATH)
    part = pl.read_parquet(PART_PATH)
    print(f"  pbp:  {pbp.shape[0]:,} plays × {pbp.shape[1]} cols")
    print(f"  part: {part.shape[0]:,} plays × {part.shape[1]} cols")

    # PBP has the cleanest play_type / defteam / season-week labels;
    # participation has the personnel + coverage labels. Join on the
    # nflverse_game_id + play_id pair (PBP uses `game_id`).
    pbp_slim = pbp.select([
        "game_id", "play_id", "season", "week", "season_type",
        "defteam", "posteam", "play_type", "qb_dropback",
        "sack", "qb_hit", "epa", "score_differential",
    ])
    part_slim = part.select([
        pl.col("nflverse_game_id").alias("game_id"),
        "play_id",
        "defenders_in_box", "number_of_pass_rushers",
        "was_pressure", "defense_personnel",
        "defense_man_zone_type", "defense_coverage_type",
    ])

    plays = pbp_slim.join(part_slim, on=["game_id", "play_id"], how="inner")
    plays = plays.filter(pl.col("play_type").is_in(["run", "pass"])
                          & pl.col("defteam").is_not_null())
    plays = plays.with_columns(
        pl.col("defense_personnel").map_elements(
            _classify_personnel, return_dtype=pl.String
        ).alias("def_pers_bucket")
    )
    print(f"  joined scrimmage plays: {plays.shape[0]:,}\n")

    # ── Per-game (defense_team, season, week) aggregates ──
    print("Building per-game scheme…")
    t0 = time.time()
    run_plays = plays.filter(pl.col("play_type") == "run")
    pass_plays = plays.filter(pl.col("play_type") == "pass")

    def _agg(df, scope: str, group_keys):
        """Compute aggregate columns for a given grouping. `scope`
        marks if it's per-game or per-season for column naming."""
        return df.group_by(group_keys).agg([
            pl.len().alias("n_plays"),
            (pl.col("play_type") == "run").sum().alias("n_run"),
            (pl.col("play_type") == "pass").sum().alias("n_pass"),
        ])

    base_keys_game = ["defteam", "season", "week"]
    base_keys_season = ["defteam", "season"]

    def _build(group_keys, scope_label):
        # Run-side stats
        run_agg = run_plays.group_by(group_keys).agg([
            pl.col("defenders_in_box").mean().alias("avg_box_run"),
            (pl.col("defenders_in_box") >= 8).cast(pl.Float64).mean().alias("pct_stacked_box"),
            (pl.col("defenders_in_box") <= 6).cast(pl.Float64).mean().alias("pct_light_box"),
            pl.len().alias("n_run_plays"),
        ])
        # Pass-side stats
        pass_agg = pass_plays.group_by(group_keys).agg([
            pl.col("number_of_pass_rushers").mean().alias("avg_pass_rushers"),
            (pl.col("number_of_pass_rushers") >= 5).cast(pl.Float64).mean().alias("pct_blitz"),
            pl.col("was_pressure").cast(pl.Float64).mean().alias("pressure_rate"),
            pl.len().alias("n_pass_plays"),
        ])
        # Coverage rates — only count plays where the label is present
        cov_pool = pass_plays.filter(
            pl.col("defense_man_zone_type").is_in(["MAN_COVERAGE", "ZONE_COVERAGE"])
        )
        cov_agg = cov_pool.group_by(group_keys).agg([
            pl.len().alias("n_cov_labeled"),
            (pl.col("defense_man_zone_type") == "MAN_COVERAGE").cast(pl.Float64).mean().alias("pct_man"),
            (pl.col("defense_man_zone_type") == "ZONE_COVERAGE").cast(pl.Float64).mean().alias("pct_zone"),
        ])
        shell_pool = pass_plays.filter(
            pl.col("defense_coverage_type").is_not_null()
            & (pl.col("defense_coverage_type") != "")
            & (pl.col("defense_coverage_type") != "BLOWN")
        )
        # Pivot a few of the most-common shells.
        shell_agg = (
            shell_pool.group_by(group_keys + ["defense_coverage_type"])
                      .agg(pl.len().alias("n"))
        )
        # Total labeled-shell plays per group, for rate calcs
        shell_tot = shell_agg.group_by(group_keys).agg(pl.col("n").sum().alias("n_shell_total"))
        shell_wide = (
            shell_agg.join(shell_tot, on=group_keys)
                     .with_columns((pl.col("n") / pl.col("n_shell_total")).alias("pct"))
                     .pivot(values="pct", index=group_keys, on="defense_coverage_type",
                             aggregate_function="first")
        )
        # Personnel split
        pers_pool = plays.filter(pl.col("def_pers_bucket").is_not_null())
        pers_agg = pers_pool.group_by(group_keys + ["def_pers_bucket"]).agg(pl.len().alias("n"))
        pers_tot = pers_agg.group_by(group_keys).agg(pl.col("n").sum().alias("n_pers_total"))
        pers_wide = (
            pers_agg.join(pers_tot, on=group_keys)
                    .with_columns((pl.col("n") / pl.col("n_pers_total")).alias("pct"))
                    .pivot(values="pct", index=group_keys, on="def_pers_bucket",
                            aggregate_function="first")
        )

        # Total play count
        total_agg = plays.group_by(group_keys).agg(pl.len().alias("n_plays"))

        # Stitch them together via outer-join
        out = total_agg
        for piece in (run_agg, pass_agg, cov_agg, shell_wide, pers_wide):
            out = out.join(piece, on=group_keys, how="left")

        # Run-rate the offense managed against this defense
        out = out.with_columns(
            (pl.col("n_run_plays") / pl.col("n_plays")).alias("run_rate_allowed")
        )

        # Standard column ordering for readability
        rename_shells = {
            "COVER_0": "cover_0", "COVER_1": "cover_1",
            "COVER_2": "cover_2", "COVER_3": "cover_3",
            "COVER_4": "cover_4", "COVER_6": "cover_6",
            "COVER_9": "cover_9", "2_MAN": "two_man",
            "COMBO": "combo",
        }
        out = out.rename({k: v for k, v in rename_shells.items() if k in out.columns})
        rename_pers = {
            "Base": "pct_base", "Nickel": "pct_nickel",
            "Dime": "pct_dime", "Quarter": "pct_quarter",
            "Other": "pct_other_pers",
        }
        out = out.rename({k: v for k, v in rename_pers.items() if k in out.columns})
        # Rename defteam → defense_team for cross-table consistency
        out = out.rename({"defteam": "defense_team"})
        return out

    game_df = _build(base_keys_game, "game")
    season_df = _build(base_keys_season, "season")
    print(f"  per-game rows:   {game_df.shape[0]:,}")
    print(f"  per-season rows: {season_df.shape[0]:,}")
    print(f"  ({(time.time()-t0):.2f}s)\n")

    GAME_OUT.parent.mkdir(parents=True, exist_ok=True)
    game_df.write_parquet(GAME_OUT)
    season_df.write_parquet(SEASON_OUT)
    print(f"✅ wrote:")
    print(f"   {GAME_OUT.relative_to(REPO_ROOT)}")
    print(f"   {SEASON_OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

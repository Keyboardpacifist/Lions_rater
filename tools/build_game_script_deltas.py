"""Game-Script Simulator deltas — Feature 4.2.

Output: data/game_script_deltas.parquet

For every team-game from 2013+ (snap-count era), determines whether
each key starter (QB1, RB1, WR1, TE1) was ACTIVE or MISSED. Then for
each "starter loss scenario," computes the league-wide delta in
scheme metrics relative to the team's baseline-with-starter games.

Schema
------
role_lost                  — QB1, RB1, WR1, TE1 (or NONE = baseline)
n_games                    — sample size
pass_rate                  — overall pass rate in this regime
pass_rate_delta            — vs the baseline (no starters out)
early_down_pass_rate       — early-down pass rate
early_down_pass_rate_delta
shotgun_rate               — shotgun rate in this regime
shotgun_rate_delta
no_huddle_rate             — no-huddle rate in this regime
no_huddle_rate_delta
plays_per_game             — pace in this regime
plays_per_game_delta
points_per_game            — team scoring in this regime
points_per_game_delta

The simulator UI takes a current matchup, asks the user to toggle
which starter (if any) is out, and shows: "When teams play without
their {role}, league-wide they shift to +X pass rate / -Y plays per
game / -Z points per game."
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
SNAPS = REPO / "data" / "nfl_snap_counts.parquet"
SCHEDULES = REPO / "data" / "nfl_schedules.parquet"
OUT = REPO / "data" / "game_script_deltas.parquet"


def _identify_team_starters(ps: pd.DataFrame) -> pd.DataFrame:
    """Return long-format (team, season, role, player_id) for starters."""
    qb = (ps[ps["position"] == "QB"]
          .groupby(["team", "season", "player_id"], as_index=False)
          ["attempts"].sum())
    qb1 = (qb.sort_values("attempts", ascending=False)
              .drop_duplicates(["team", "season"]))
    qb1["role"] = "QB1"

    rb = (ps[ps["position"] == "RB"]
          .groupby(["team", "season", "player_id"], as_index=False)
          ["carries"].sum())
    rb1 = (rb.sort_values("carries", ascending=False)
              .drop_duplicates(["team", "season"]))
    rb1["role"] = "RB1"

    wr = (ps[ps["position"] == "WR"]
          .groupby(["team", "season", "player_id"], as_index=False)
          ["targets"].sum())
    wr1 = (wr.sort_values("targets", ascending=False)
              .drop_duplicates(["team", "season"]))
    wr1["role"] = "WR1"

    te = (ps[ps["position"] == "TE"]
          .groupby(["team", "season", "player_id"], as_index=False)
          ["targets"].sum())
    te1 = (te.sort_values("targets", ascending=False)
              .drop_duplicates(["team", "season"]))
    te1["role"] = "TE1"

    return pd.concat([
        qb1[["team", "season", "player_id", "role"]],
        rb1[["team", "season", "player_id", "role"]],
        wr1[["team", "season", "player_id", "role"]],
        te1[["team", "season", "player_id", "role"]],
    ], ignore_index=True)


def main() -> None:
    print("→ loading pbp + player stats + snap counts + schedules...")
    pbp = pd.read_parquet(PBP)
    ps = pd.read_parquet(PLAYER_STATS)
    snaps = pd.read_parquet(SNAPS)
    sch = pd.read_parquet(SCHEDULES)

    starters = _identify_team_starters(ps)
    print(f"  starters identified: {len(starters):,}")

    # Determine which starters were ACTIVE in each (game, team) cell.
    # "Active" = took ≥1 offensive snap in the snap_counts table.
    snaps = snaps[snaps["offense_snaps"].fillna(0) > 0]
    snaps_keys = snaps[["pfr_player_id", "season", "week", "team",
                         "game_id", "offense_snaps"]].copy()

    # Need to map snap rows to player_id (gsis). Unfortunately
    # snap_counts uses pfr_player_id while everything else uses gsis_id
    # (player_id in player_stats). Use rosters to bridge.
    rosters = pd.read_parquet(REPO / "data" / "nfl_rosters.parquet")
    bridge = rosters[["gsis_id", "season"]].drop_duplicates()
    # nflreadpy rosters has both gsis_id and pfr_id in the full pull,
    # but we trimmed pfr_id out. Re-pull from raw if needed.
    if "pfr_id" not in rosters.columns:
        # Bridge via name+team+season as a fallback
        # Use full_name from player_stats
        bridge_alt = ps[["player_id", "player_display_name",
                          "team", "season"]].drop_duplicates()
        bridge_alt["name_norm"] = (bridge_alt["player_display_name"]
                                    .astype(str).str.lower().str.strip())
        snaps_keys["name_norm"] = (snaps_keys["pfr_player_id"]
                                    .astype(str))  # fallback
        # Actually the cleanest path is to use snap_counts.player which
        # has the full name. Let me re-load snaps with that column:
        snaps_full = pd.read_parquet(SNAPS)
        snaps_full = snaps_full[snaps_full["offense_snaps"].fillna(0) > 0]
        snaps_full["name_norm"] = (snaps_full["player"].astype(str)
                                    .str.lower().str.strip())
        merged = snaps_full.merge(
            bridge_alt[["player_id", "name_norm", "team", "season"]],
            on=["name_norm", "team", "season"], how="left",
        )
        active = merged.dropna(subset=["player_id"])
        active = active.rename(columns={"player_id": "gsis_id"})
        active_keys = active[["gsis_id", "season", "week", "team",
                               "game_id"]].drop_duplicates()
    else:
        active_keys = snaps_keys
    print(f"  active player-game rows: {len(active_keys):,}")

    # Identify which starters were ACTIVE in each (season, week, team).
    # Long-format: starter present rows.
    sa = starters.merge(
        active_keys.rename(columns={"gsis_id": "player_id"}),
        on=["player_id", "season", "team"], how="inner")
    # Now (season, week, team, role) means that role's starter was active
    # in that game. Pivot to wide.
    sa["was_active"] = True
    pivot = (sa.pivot_table(
        index=["season", "week", "team", "game_id"],
        columns="role",
        values="was_active",
        aggfunc="first",
        fill_value=False,
    ).reset_index())
    # Ensure all four role columns exist
    for r in ("QB1", "RB1", "WR1", "TE1"):
        if r not in pivot.columns:
            pivot[r] = False

    # Classify each team-game by which (single) role was missed. To
    # keep it interpretable for v1, label each game by the FIRST role
    # missing in our priority order; multi-out games go to a separate
    # "MULTI" bucket.
    def _scenario(row) -> str:
        out = [r for r in ("QB1", "RB1", "WR1", "TE1") if not row[r]]
        if not out:
            return "NONE"
        if len(out) == 1:
            return out[0]
        return "MULTI"
    pivot["scenario"] = pivot.apply(_scenario, axis=1)
    print(f"  team-games classified: {len(pivot):,}")
    print(pivot["scenario"].value_counts().to_string())

    # Compute scheme metrics per (game, team) using pbp
    pbp = pbp[pbp["season"] >= 2013].copy()
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    plays["is_pass"] = (plays["play_type"] == "pass").astype(int)
    plays["is_early_down"] = plays["down"].isin([1, 2])
    plays["is_shotgun"] = plays["shotgun"].fillna(0).astype(int)
    plays["is_nohuddle"] = plays["no_huddle"].fillna(0).astype(int)

    by_game = (plays.groupby(["season", "week", "posteam", "game_id"])
               .agg(plays=("is_pass", "size"),
                    pass_rate=("is_pass", "mean"),
                    early_down_plays=("is_early_down", "sum"),
                    early_down_pass_rate=("is_pass",
                        lambda x: x[plays.loc[x.index, "is_early_down"]].mean()
                        if plays.loc[x.index, "is_early_down"].any() else float("nan")
                    ),
                    shotgun_rate=("is_shotgun", "mean"),
                    no_huddle_rate=("is_nohuddle", "mean"))
               .reset_index()
               .rename(columns={"posteam": "team"}))

    # Add points_for from schedules
    home_pts = sch[["season", "week", "home_team", "home_score", "game_id"]].rename(
        columns={"home_team": "team", "home_score": "points_for"})
    away_pts = sch[["season", "week", "away_team", "away_score", "game_id"]].rename(
        columns={"away_team": "team", "away_score": "points_for"})
    pts = pd.concat([home_pts, away_pts], ignore_index=True)

    by_game = by_game.merge(pts, on=["season", "week", "team", "game_id"],
                            how="left")

    # Attach scenario
    enriched = by_game.merge(
        pivot[["season", "week", "team", "scenario"]],
        on=["season", "week", "team"], how="left",
    )
    enriched["scenario"] = enriched["scenario"].fillna("UNKNOWN")
    print(f"  enriched team-games: {len(enriched):,}")

    # Aggregate per scenario
    grouped = (enriched.groupby("scenario")
               .agg(n_games=("plays", "size"),
                    plays_per_game=("plays", "mean"),
                    pass_rate=("pass_rate", "mean"),
                    early_down_pass_rate=("early_down_pass_rate", "mean"),
                    shotgun_rate=("shotgun_rate", "mean"),
                    no_huddle_rate=("no_huddle_rate", "mean"),
                    points_per_game=("points_for", "mean"))
               .reset_index())

    # Compute deltas vs the NONE (no starters out) baseline
    base = grouped[grouped["scenario"] == "NONE"].iloc[0]
    for col in ("plays_per_game", "pass_rate", "early_down_pass_rate",
                "shotgun_rate", "no_huddle_rate", "points_per_game"):
        grouped[f"{col}_delta"] = grouped[col] - base[col]

    grouped = grouped.sort_values("n_games", ascending=False).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== League-wide game-script shifts when a starter is out ===")
    print(grouped.to_string())


if __name__ == "__main__":
    main()

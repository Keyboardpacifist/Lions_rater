#!/usr/bin/env python3
"""
Per-(player, game) explosive-play counts + defense baselines.

Industry-standard cuts:
  • Explosive run     = rush attempt of ≥ 10 yards
  • Explosive reception = completed catch of ≥ 20 yards

Output two parquets — kept separate from the main adjusted parquet so
this layer can be regenerated independently and lib_splits can join it
in lazily.

Inputs:
  data/games/nfl_pbp.parquet
  data/games/nfl_weekly_stats.parquet  (for player position lookup)

Outputs:
  data/games/nfl_explosive_player_games.parquet
    one row per (player_id, season, week, team) with:
      explosive_runs, explosive_receptions
  data/games/nfl_explosive_def_baselines.parquet
    one row per (defense_team, season, position) with:
      avg_explosive_runs, avg_explosive_receptions
    using the same qualifying-game thresholds as nfl_defense_baselines.

Usage:
    python tools/game_logs/build_explosive_plays.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PBP_PATH = REPO_ROOT / "data" / "games" / "nfl_pbp.parquet"
WSTATS_PATH = REPO_ROOT / "data" / "games" / "nfl_weekly_stats.parquet"
PLAYER_OUT = REPO_ROOT / "data" / "games" / "nfl_explosive_player_games.parquet"
DEF_OUT = REPO_ROOT / "data" / "games" / "nfl_explosive_def_baselines.parquet"

EXPLOSIVE_RUN_THRESHOLD = 10  # yards
EXPLOSIVE_REC_THRESHOLD = 20  # yards

# Same qualifying-game thresholds the defense_baselines build uses,
# so the explosive averages are apples-to-apples.
POSITION_GROUPS = {
    "QB": {"positions": ["QB"], "gate": "attempts", "min": 10},
    "RB": {"positions": ["RB", "FB"], "gate": "carries", "min": 5},
    "WR": {"positions": ["WR"], "gate": "targets", "min": 2},
    "TE": {"positions": ["TE"], "gate": "targets", "min": 2},
}


def main():
    if not PBP_PATH.exists():
        raise SystemExit(f"Missing {PBP_PATH}. Run `make game-logs-pbp`.")
    if not WSTATS_PATH.exists():
        raise SystemExit(f"Missing {WSTATS_PATH}. Run `make game-logs-nfl`.")

    print("Loading PBP…")
    pbp = pl.read_parquet(PBP_PATH)
    print(f"  {pbp.shape[0]:,} plays")

    # ── 1. Explosive runs per (rusher_player_id, season, week, team) ──
    print("\nComputing explosive runs (≥10 yds)…")
    runs = (
        pbp.filter(pl.col("play_type") == "run")
           .filter(pl.col("rusher_player_id").is_not_null())
           .with_columns([
               (pl.col("yards_gained") >= EXPLOSIVE_RUN_THRESHOLD)
                   .cast(pl.Int64).alias("is_explosive"),
           ])
           .group_by(["season", "week", "posteam", "rusher_player_id"])
           .agg(pl.col("is_explosive").sum().alias("explosive_runs"))
           .rename({"posteam": "team", "rusher_player_id": "player_id"})
    )
    print(f"  {runs.shape[0]:,} (player, game) rusher rows")

    # ── 2. Explosive receptions per (receiver_player_id, season, week, team) ──
    print("\nComputing explosive receptions (≥20 yds, completions only)…")
    recs = (
        pbp.filter(pl.col("play_type") == "pass")
           .filter(pl.col("receiver_player_id").is_not_null())
           .filter(pl.col("complete_pass") == 1)
           .with_columns([
               (pl.col("yards_gained") >= EXPLOSIVE_REC_THRESHOLD)
                   .cast(pl.Int64).alias("is_explosive"),
           ])
           .group_by(["season", "week", "posteam", "receiver_player_id"])
           .agg(pl.col("is_explosive").sum().alias("explosive_receptions"))
           .rename({"posteam": "team", "receiver_player_id": "player_id"})
    )
    print(f"  {recs.shape[0]:,} (player, game) receiver rows")

    # ── 3. Outer join into one player-game table (each player can be
    #       both rusher AND receiver in a given game). ──
    player_games = runs.join(
        recs, on=["season", "week", "team", "player_id"], how="full",
        coalesce=True,
    )
    # Fill nulls with 0 (a player who didn't run or catch any passes
    # simply has 0 explosive plays of that type).
    player_games = player_games.with_columns([
        pl.col("explosive_runs").fill_null(0),
        pl.col("explosive_receptions").fill_null(0),
    ])
    print(f"\nMerged: {player_games.shape[0]:,} (player, game) rows")

    PLAYER_OUT.parent.mkdir(parents=True, exist_ok=True)
    player_games.write_parquet(PLAYER_OUT)
    print(f"  ✓ {PLAYER_OUT.relative_to(REPO_ROOT)}")

    # ── 4. Defense baselines: avg explosive plays allowed per
    #       qualifying player-game, broken by position group. ──
    print("\nBuilding defense baselines…")
    t0 = time.time()
    wstats = pl.read_parquet(WSTATS_PATH).select([
        "player_id", "season", "week", "team", "opponent_team",
        "position", "attempts", "carries", "targets",
    ])

    # Join the explosive layer onto weekly stats so each player-game
    # has both stat-line context AND explosive counts available.
    enriched = wstats.join(
        player_games, on=["season", "week", "team", "player_id"],
        how="left",
    )
    enriched = enriched.with_columns([
        pl.col("explosive_runs").fill_null(0),
        pl.col("explosive_receptions").fill_null(0),
    ])

    parts = []
    for group_name, cfg in POSITION_GROUPS.items():
        # Same qualifying mask as the existing defense baselines build
        qual = enriched.filter(pl.col("position").is_in(cfg["positions"]))
        qual = qual.filter(pl.col(cfg["gate"]).fill_null(0) >= cfg["min"])
        qual = qual.filter(pl.col("opponent_team").is_not_null())

        agg = (
            qual.group_by(["opponent_team", "season"])
                .agg([
                    pl.col("explosive_runs").mean().alias("avg_explosive_runs"),
                    pl.col("explosive_receptions").mean().alias("avg_explosive_receptions"),
                    pl.len().alias("n_player_games"),
                ])
                .with_columns(pl.lit(group_name).alias("position"))
                .rename({"opponent_team": "defense_team"})
        )
        parts.append(agg)

    defense_baselines = pl.concat(parts, how="vertical_relaxed")
    defense_baselines = defense_baselines.sort(["defense_team", "season", "position"])

    DEF_OUT.parent.mkdir(parents=True, exist_ok=True)
    defense_baselines.write_parquet(DEF_OUT)
    print(f"  ✓ {DEF_OUT.relative_to(REPO_ROOT)} "
          f"({defense_baselines.shape[0]:,} rows · {(time.time()-t0):.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

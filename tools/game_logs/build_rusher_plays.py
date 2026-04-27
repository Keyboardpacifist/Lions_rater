#!/usr/bin/env python3
"""
Per-run-play feed for rushers — joins PBP + participation into a
per-run row with everything the run-scheme panel needs:

  • run_location  (left / middle / right)
  • run_gap       (guard / tackle / end)
  • defenders_in_box, offense_formation, offense_personnel
  • result: yards_gained, success, first_down, touchdown, epa
  • situation: down, ydstogo, score_differential

Output:
  data/games/nfl_rusher_plays.parquet
    one row per run play (rushes only — kneel-downs / spikes excluded
    by play_type filter)
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PBP_PATH = REPO_ROOT / "data" / "games" / "nfl_pbp.parquet"
PART_PATH = REPO_ROOT / "data" / "games" / "nfl_participation.parquet"
OUT = REPO_ROOT / "data" / "games" / "nfl_rusher_plays.parquet"


def main():
    if not PBP_PATH.exists():
        raise SystemExit(f"Missing {PBP_PATH}.")
    if not PART_PATH.exists():
        raise SystemExit(f"Missing {PART_PATH}.")

    print("Loading inputs…")
    pbp = pl.read_parquet(PBP_PATH)
    part = pl.read_parquet(PART_PATH)

    pbp_slim = (
        pbp.filter((pl.col("play_type") == "run")
                    & pl.col("rusher_player_id").is_not_null())
           .select([
               "game_id", "play_id", "season", "week", "season_type",
               "posteam", "defteam",
               "rusher_player_id", "rusher_player_name",
               "run_location", "run_gap",
               "yards_gained", "success", "first_down", "touchdown",
               "epa", "down", "ydstogo", "score_differential",
               "shotgun",
           ])
    )

    part_slim = part.select([
        pl.col("nflverse_game_id").alias("game_id"),
        "play_id",
        "defenders_in_box",
        "offense_formation", "offense_personnel",
    ])
    out = pbp_slim.join(part_slim, on=["game_id", "play_id"], how="left")
    out = out.rename({"rusher_player_id": "player_id",
                       "posteam": "team",
                       "defteam": "opponent_team"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)
    print(f"✅ wrote {out.shape[0]:,} rows × {out.shape[1]} cols")
    print(f"   → {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

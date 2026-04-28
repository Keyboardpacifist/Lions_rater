#!/usr/bin/env python3
"""
Per-dropback feed for the QB panel — one row per pass attempt, sack,
or scramble. Foundation for every QB split: pressure, situational,
elite-vs-weak competition, throw map.

Output: data/qb_dropbacks.parquet

Run:
    python tools/build_qb_dropbacks.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "data" / "qb_dropbacks.parquet"

# Match the QB rater's season range
SEASONS = list(range(2016, 2026))

KEEP_COLS = [
    # Identification
    "game_id", "play_id", "season", "week", "season_type",
    "passer_player_id", "passer_player_name",
    "posteam", "defteam",
    # Pre-snap context
    "qtr", "quarter_seconds_remaining", "game_seconds_remaining",
    "down", "ydstogo", "yardline_100", "score_differential",
    "shotgun", "no_huddle", "play_clock", "goal_to_go",
    # Outcome
    "play_type", "pass_attempt", "sack", "complete_pass",
    "interception", "fumble_lost", "qb_hit", "qb_scramble",
    "pass_location", "pass_length", "air_yards",
    "yards_after_catch", "passing_yards",
    "epa", "wpa", "success",
    "receiver_player_id",
]


def main() -> None:
    import nflreadpy as nfl

    print(f"Pulling pbp for {SEASONS[0]}-{SEASONS[-1]}…")
    pbp = nfl.load_pbp(SEASONS).to_pandas()
    print(f"  {len(pbp):,} total plays loaded")

    # Dropback = pass attempt OR sack OR scramble (with a real passer)
    is_dropback = (
        (pbp["pass_attempt"] == 1)
        | (pbp["sack"] == 1)
        | (pbp.get("qb_scramble", 0) == 1)
    )
    has_passer = pbp["passer_player_id"].notna()
    db = pbp[is_dropback & has_passer].copy()
    print(f"  {len(db):,} dropbacks")

    keep = [c for c in KEEP_COLS if c in db.columns]
    db = db[keep].reset_index(drop=True)

    # Pre-computed split flags (faster than recomputing in the UI on every render)
    db["is_third_down"] = (db["down"] == 3).fillna(False)
    db["is_red_zone"] = (db["yardline_100"] <= 20).fillna(False)
    db["is_fourth_quarter"] = (db["qtr"] == 4).fillna(False)
    db["is_two_minute"] = (
        ((db["qtr"] == 2) | (db["qtr"] == 4))
        & (db["quarter_seconds_remaining"] <= 120)
    ).fillna(False)
    db["is_pressured"] = (db.get("qb_hit", 0) == 1).fillna(False)
    db["is_trailing"] = (db["score_differential"] < 0).fillna(False)
    db["is_one_score"] = (db["score_differential"].abs() <= 8).fillna(False)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    db.to_parquet(OUTPUT, index=False)
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\n✓ wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(db):,} rows × {db.shape[1]} cols ({size_mb:.1f} MB)")
    print(f"  Seasons: {sorted(int(s) for s in db['season'].dropna().unique())}")
    print(f"  Unique passers: {db['passer_player_id'].nunique():,}")


if __name__ == "__main__":
    main()

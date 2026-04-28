#!/usr/bin/env python3
"""
Per (QB, season) Next Gen Stats — time-to-throw, aggressiveness,
intended air yards, CPOE, etc. Used by the QB panel's Processing
bucket.

Output: data/qb_ngs_seasons.parquet

Run:
    python tools/build_qb_ngs.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "data" / "qb_ngs_seasons.parquet"

SEASONS = list(range(2016, 2026))

KEEP_COLS = [
    "player_gsis_id", "player_display_name", "team_abbr",
    "season", "season_type", "week",
    "avg_time_to_throw", "avg_completed_air_yards",
    "avg_intended_air_yards", "avg_air_yards_differential",
    "aggressiveness", "max_completed_air_distance",
    "avg_air_yards_to_sticks", "avg_air_distance", "max_air_distance",
    "attempts", "pass_yards", "pass_touchdowns", "interceptions",
    "completions", "completion_percentage",
    "expected_completion_percentage",
    "completion_percentage_above_expectation",
    "passer_rating",
]


def main() -> None:
    import nflreadpy as nfl

    print(f"Pulling NGS passing for {SEASONS[0]}-{SEASONS[-1]}…")
    frames = []
    for s in SEASONS:
        try:
            df = nfl.load_nextgen_stats(seasons=[s], stat_type="passing").to_pandas()
            frames.append(df)
            print(f"  {s}: {len(df):,} rows")
        except Exception as e:
            print(f"  {s}: FAILED ({e})")

    if not frames:
        raise SystemExit("No NGS data loaded.")

    raw = pd.concat(frames, ignore_index=True)

    # NGS gives weekly + season-summary rows (week=0 = season summary).
    # Keep season summaries for season-level joins; weekly stays
    # available if we need it later but we don't right now.
    season_rows = raw[raw["week"] == 0].copy()
    # Regular season only for the season summary (NGS publishes
    # season summaries by season_type)
    season_rows = season_rows[season_rows["season_type"] == "REG"].reset_index(drop=True)

    keep = [c for c in KEEP_COLS if c in season_rows.columns]
    season_rows = season_rows[keep]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    season_rows.to_parquet(OUTPUT, index=False)
    print(f"\n✓ wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(season_rows):,} (player, season) rows × "
          f"{season_rows.shape[1]} cols")


if __name__ == "__main__":
    main()

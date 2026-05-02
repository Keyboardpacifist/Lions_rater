"""Pull historical NFL weekly player stats.

Output: data/nfl_player_stats_weekly.parquet

Per-player-per-game stat lines used for SGP correlation matrices.
"""
from __future__ import annotations

from pathlib import Path

import nflreadpy

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "nfl_player_stats_weekly.parquet"


def main() -> None:
    print("→ pulling weekly player stats...")
    df = nflreadpy.load_player_stats(seasons=True)
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    print(f"  rows: {len(df):,}")
    keep = ["season", "week", "team", "opponent_team", "player_id",
            "player_name", "player_display_name", "position",
            "position_group",
            "completions", "attempts", "passing_yards", "passing_tds",
            "passing_interceptions", "passing_first_downs",
            "carries", "rushing_yards", "rushing_tds",
            "targets", "receptions", "receiving_yards", "receiving_tds",
            "receiving_air_yards", "receiving_yards_after_catch",
            "target_share", "air_yards_share", "wopr",
            "fantasy_points", "fantasy_points_ppr"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    print(f"  seasons: {int(df['season'].min())}–{int(df['season'].max())}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

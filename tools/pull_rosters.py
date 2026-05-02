"""Pull historical NFL weekly rosters → player position lookup.

Output: data/nfl_rosters.parquet

Used to attach a position to every play in the pbp data so DvP / cohort
engines can group by position group.
"""
from __future__ import annotations

from pathlib import Path

import nflreadpy

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "nfl_rosters.parquet"


def main() -> None:
    print("→ pulling NFL rosters...")
    df = nflreadpy.load_rosters(seasons=True)
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    print(f"  rows: {len(df):,}")
    keep = ["season", "team", "position", "depth_chart_position",
            "full_name", "gsis_id", "ngs_position"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    print(f"  seasons: {int(df['season'].min())}–{int(df['season'].max())}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

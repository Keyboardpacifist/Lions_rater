"""Pull historical NFL snap counts from nflreadpy.

Output: data/nfl_snap_counts.parquet

Each row is one player-game with offense_snaps, defense_snaps, st_snaps
(plus pct of team total). This is the GOLD-STANDARD outcome variable for
the injury cohort engine — joined back to the Friday injury report it
tells us "given this Friday status, did the player actually play?"

Snap counts only go back to 2012.
"""
from __future__ import annotations

from pathlib import Path

import nflreadpy

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "nfl_snap_counts.parquet"


def main() -> None:
    print("→ pulling NFL snap counts 2012-present...")
    df = nflreadpy.load_snap_counts(seasons=True)
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    print(f"  rows: {len(df):,}")
    print(f"  cols: {df.columns.tolist()}")
    if "season" in df.columns:
        seasons = sorted(df["season"].dropna().unique().tolist())
        print(f"  seasons: {int(seasons[0])}–{int(seasons[-1])}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

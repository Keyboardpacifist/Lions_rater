"""Pull historical NFL injury reports from nflreadpy.

Output: data/nfl_injuries_historical.parquet

Covers 2009-present. Each row is one player-week-status entry from
official NFL practice reports — includes practice status (DNP /
Limited / Full), game-day designation (Active / Inactive / etc.),
and the body-part / report description.

This is the foundation for the cohort-matching injury model.
"""
from __future__ import annotations

from pathlib import Path

import nflreadpy

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "nfl_injuries_historical.parquet"


def main() -> None:
    print("→ pulling NFL injuries 2009-present (this can take 30-60s)...")
    df = nflreadpy.load_injuries(seasons=True)
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

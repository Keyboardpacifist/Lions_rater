"""Pull NextGenStats player tracking data for passing / rushing /
receiving and save to parquets. One-time + can be re-run to refresh.

Outputs:
  data/ngs_passing.parquet
  data/ngs_rushing.parquet
  data/ngs_receiving.parquet
"""
from __future__ import annotations

from pathlib import Path

import nflreadpy

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stat_type in ("passing", "rushing", "receiving"):
        print(f"→ pulling NGS {stat_type}…")
        # `seasons=True` loads every year back to 2016
        df = nflreadpy.load_nextgen_stats(stat_type=stat_type,
                                              seasons=True)
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()
        # Year range
        season_col = "season" if "season" in df.columns else None
        seasons = (sorted(df[season_col].unique())
                   if season_col else [])
        path = OUT / f"ngs_{stat_type}.parquet"
        df.to_parquet(path, index=False)
        print(f"  ✓ wrote {path.relative_to(REPO)} "
              f"({len(df):,} rows · seasons "
              f"{seasons[0]}–{seasons[-1] if seasons else '?'})")
        print(f"  cols: {df.columns.tolist()[:10]}…")


if __name__ == "__main__":
    main()

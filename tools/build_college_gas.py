"""Build College GAS Scores for QB / RB / WR / TE.

Outputs:
  data/college/college_qb_gas_seasons.parquet
  data/college/college_rb_gas_seasons.parquet
  data/college/college_wr_gas_seasons.parquet
  data/college/college_te_gas_seasons.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib_college_gas import (  # noqa: E402
    COLLEGE_QB_SPEC, COLLEGE_RB_SPEC, COLLEGE_WR_SPEC, COLLEGE_TE_SPEC,
    NEGATIVE_STATS_QB, NEGATIVE_STATS_RB,
    NEGATIVE_STATS_WR, NEGATIVE_STATS_TE,
    compute_college_gas,
)

CONFIGS = [
    ("QB", "college_qb_all_seasons.parquet",
     COLLEGE_QB_SPEC, NEGATIVE_STATS_QB),
    ("RB", "college_rb_all_seasons.parquet",
     COLLEGE_RB_SPEC, NEGATIVE_STATS_RB),
    ("WR", "college_wr_all_seasons.parquet",
     COLLEGE_WR_SPEC, NEGATIVE_STATS_WR),
    ("TE", "college_te_all_seasons.parquet",
     COLLEGE_TE_SPEC, NEGATIVE_STATS_TE),
]


def _normalize(master: pd.DataFrame) -> pd.DataFrame:
    if "season" in master.columns and "season_year" not in master.columns:
        master = master.rename(columns={"season": "season_year"})
    elif "season" in master.columns and "season_year" in master.columns:
        master = master.drop(columns=["season"])
    return master


def main() -> None:
    college_dir = REPO / "data" / "college"
    for label, fname, spec, neg in CONFIGS:
        in_path = college_dir / fname
        out_path = college_dir / fname.replace(
            "_all_seasons.parquet", "_gas_seasons.parquet")

        print(f"\n→ College {label} GAS — loading {fname}")
        master = pd.read_parquet(in_path)
        print(f"  rows: {len(master):,}")
        master = _normalize(master)
        graded = compute_college_gas(master, spec=spec,
                                          negative_stats=neg)
        graded.to_parquet(out_path, index=False)
        print(f"  ✓ wrote {out_path.relative_to(REPO)}")

        # Spot check 2024 top 10
        if "games" in graded.columns:
            min_g = 8
            recent = graded[(graded["season_year"] == 2024)
                              & (graded["games"] >= min_g)]
        else:
            recent = graded[graded["season_year"] == 2024]
        if len(recent):
            top = recent.nlargest(10, "gas_score")
            cols = ["player_display_name", "team", "games",
                     "gas_score", "gas_label"]
            cols = [c for c in cols if c in top.columns]
            print(f"  2024 top 10:")
            print(top[cols].to_string(index=False))

    print("\n✓ College GAS build complete.")


if __name__ == "__main__":
    main()

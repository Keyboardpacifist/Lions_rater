"""Build RB GAS Score table.

Output: data/rb_gas_seasons.parquet

Joins the existing master league_rb_all_seasons z-cols with the
SOS-adjusted z-cols built by build_rb_sos_adjusted.py, then runs
lib_rb_gas.compute_rb_gas to produce composite GAS + sub-grades.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MASTER = REPO / "data" / "league_rb_all_seasons.parquet"
SOS = REPO / "data" / "rb_sos_adjusted_z.parquet"
OUT = REPO / "data" / "rb_gas_seasons.parquet"

from lib_rb_gas import compute_rb_gas  # noqa: E402


def main() -> None:
    print("→ loading master + SOS z-cols...")
    master = pd.read_parquet(MASTER)
    sos = pd.read_parquet(SOS)
    print(f"  master rows: {len(master):,}")
    print(f"  SOS rows: {len(sos):,}")

    merged = master.merge(sos, on=["player_id", "season_year"],
                            how="left")
    print(f"  merged rows: {len(merged):,}")
    matched = merged["adj_epa_per_rush_z"].notna().sum()
    print(f"  with SOS data: {matched:,} "
          f"({matched / max(len(merged), 1):.0%})")

    graded = compute_rb_gas(merged)
    print(f"  graded rows: {len(graded):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    graded.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    cols = ["player_display_name", "recent_team", "games",
            "gas_score", "gas_label",
            "gas_rushing_efficiency_grade", "gas_receiving_grade",
            "gas_volume_durability_grade", "gas_explosiveness_grade",
            "gas_red_zone_grade", "gas_short_yardage_grade"]
    cols = [c for c in cols if c in graded.columns]
    print("=== 2024 Top 12 RBs by GAS ===")
    s24 = graded[(graded["season_year"] == 2024)
                  & (graded["games"] >= 10)].nlargest(12, "gas_score")
    print(s24[cols].to_string(index=False))


if __name__ == "__main__":
    main()

"""Build WR GAS Score table.

Output: data/wr_gas_seasons.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MASTER = REPO / "data" / "league_wr_all_seasons.parquet"
SOS = REPO / "data" / "wr_sos_adjusted_z.parquet"
OUT = REPO / "data" / "wr_gas_seasons.parquet"

from lib_wr_gas import compute_wr_gas  # noqa: E402


def main() -> None:
    print("→ loading master + SOS z-cols...")
    master = pd.read_parquet(MASTER)
    sos = pd.read_parquet(SOS)
    print(f"  master rows (all positions): {len(master):,}")

    # Filter to WR-only — the master file may contain TE rows too
    if "position" in master.columns:
        before = len(master)
        master = master[master["position"] == "WR"].copy()
        print(f"  filtered to position=WR: {len(master):,}  "
              f"(dropped {before - len(master)})")
    print(f"  SOS rows: {len(sos):,}")

    merged = master.merge(sos, on=["player_id", "season_year"],
                            how="left")
    print(f"  merged rows: {len(merged):,}")
    matched = merged["adj_epa_per_target_z"].notna().sum()
    print(f"  with SOS data: {matched:,} "
          f"({matched / max(len(merged), 1):.0%})")

    graded = compute_wr_gas(merged)
    print(f"  graded rows: {len(graded):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    graded.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== 2024 Top 15 WRs ===")
    cols = ["player_display_name", "recent_team", "games",
            "gas_score", "gas_label",
            "gas_per_target_efficiency_grade",
            "gas_volume_role_grade",
            "gas_coverage_beating_grade",
            "gas_yac_grade",
            "gas_scoring_chains_grade"]
    cols = [c for c in cols if c in graded.columns]
    s24 = graded[(graded["season_year"] == 2024)
                  & (graded["games"] >= 12)].nlargest(15, "gas_score")
    print(s24[cols].to_string(index=False))


if __name__ == "__main__":
    main()

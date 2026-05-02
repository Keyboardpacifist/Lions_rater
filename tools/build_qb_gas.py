"""Build QB GAS Score table.

Output: data/qb_gas_seasons.parquet

Joins league_qb_all_seasons.parquet (pre-computed z-cols for
efficiency / volume / ball-security / mobility bundles) with
qb_pressure_clutch_z.parquet (newly-computed z-cols for the
pressure + clutch bundles), then runs lib_qb_gas.compute_qb_gas
to produce the composite GAS Score and per-bundle sub-grades.

Output schema (one row per (player_id, season_year)):
    player_id, player_display_name, recent_team, season_year, games,
    gas_efficiency_grade, gas_volume_grade, gas_ball_security_grade,
    gas_pressure_grade, gas_mobility_grade, gas_clutch_grade,
    gas_score, gas_label, gas_confidence
    + every input z-col (kept for transparency / debugging)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MASTER = REPO / "data" / "league_qb_all_seasons.parquet"
PRESSURE = REPO / "data" / "qb_pressure_clutch_z.parquet"
OUT = REPO / "data" / "qb_gas_seasons.parquet"

from lib_qb_gas import compute_qb_gas  # noqa: E402


def main() -> None:
    print("→ loading master + pressure z-cols...")
    master = pd.read_parquet(MASTER)
    pressure = pd.read_parquet(PRESSURE)
    print(f"  master rows: {len(master):,}")
    print(f"  pressure rows: {len(pressure):,}")

    # Merge on (player_id, season_year)
    merged = master.merge(
        pressure,
        on=["player_id", "season_year"],
        how="left",
    )
    print(f"  merged rows: {len(merged):,}")
    matched = merged["epa_under_pressure_z"].notna().sum()
    print(f"  with pressure data: {matched:,} ({matched / max(len(merged), 1):.0%})")

    # Apply GAS scoring
    graded = compute_qb_gas(merged)
    print(f"  graded rows: {len(graded):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    graded.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()

    # ── Spot checks ─────────────────────────────────────────────
    print("=== Top 12 QBs by GAS Score, 2024 ===")
    cols = ["player_display_name", "recent_team", "games",
            "gas_score", "gas_label", "gas_confidence",
            "gas_efficiency_grade", "gas_volume_grade",
            "gas_ball_security_grade", "gas_pressure_grade",
            "gas_mobility_grade", "gas_clutch_grade"]
    cols = [c for c in cols if c in graded.columns]
    s24 = graded[graded["season_year"] == 2024].nlargest(12, "gas_score")
    print(s24[cols].to_string())
    print()
    print("=== Top 12 QBs by GAS Score, 2023 ===")
    s23 = graded[graded["season_year"] == 2023].nlargest(12, "gas_score")
    print(s23[cols].to_string())
    print()
    print("=== Bottom 5 (full-season starters, 2024) ===")
    bot = graded[(graded["season_year"] == 2024)
                  & (graded["games"] >= 14)].nsmallest(5, "gas_score")
    print(bot[cols].to_string())


if __name__ == "__main__":
    main()

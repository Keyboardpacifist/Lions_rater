"""Build CB GAS Score table.

Output: data/cb_gas_seasons.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MASTER = REPO / "data" / "league_cb_all_seasons.parquet"
OUT = REPO / "data" / "cb_gas_seasons.parquet"

from lib_cb_gas import compute_cb_gas  # noqa: E402


def main() -> None:
    print("→ loading CB master...")
    master = pd.read_parquet(MASTER)
    print(f"  rows: {len(master):,}")
    print(f"  positions: {master['position'].value_counts().to_dict()}")

    # Filter to CBs (not DBs/Safeties — those are separate position pages)
    before = len(master)
    master = master[master["position"] == "CB"].copy()
    print(f"  filtered to CB: {len(master):,}  "
          f"(dropped {before - len(master)})")

    # Rename season → season_year for consistency
    if "season_year" not in master.columns and "season" in master.columns:
        master = master.rename(columns={"season": "season_year"})
    elif "season" in master.columns and "season_year" in master.columns:
        master = master.drop(columns=["season"])

    graded = compute_cb_gas(master)
    print(f"  graded rows: {len(graded):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    graded.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== Top 15 CB by GAS, 2024 (≥600 def_snaps) ===")
    cols = ["player_display_name", "team", "def_snaps",
            "gas_score", "gas_label", "gas_confidence",
            "gas_coverage_grade", "gas_ball_production_grade",
            "gas_tackling_grade"]
    cols = [c for c in cols if c in graded.columns]
    s24 = graded[(graded["season_year"] == 2024)
                  & (graded["def_snaps"] >= 600)
                  ].nlargest(15, "gas_score")
    print(s24[cols].to_string(index=False))
    print()
    print("=== Lions CBs 2024 ===")
    det = graded[(graded["season_year"] == 2024)
                  & (graded["team"] == "DET")
                  & (graded["def_snaps"] >= 100)].sort_values(
        "gas_score", ascending=False)
    print(det[cols].to_string(index=False))
    print()
    df_y = graded[graded["def_snaps"] >= 500][
        ["player_id", "season_year", "gas_score"]
    ].dropna(subset=["player_id"]).sort_values(
        ["player_id", "season_year"]).copy()
    df_y["next"] = df_y.groupby("player_id")["gas_score"].shift(-1)
    df_y["next_season"] = df_y.groupby("player_id"
                                          )["season_year"].shift(-1)
    yoy = df_y.dropna(subset=["next"])
    yoy = yoy[yoy["next_season"] - yoy["season_year"] == 1]
    print(f"CB YoY r (≥500 snaps both years): "
          f"{yoy['gas_score'].corr(yoy['next']):.3f}  (n={len(yoy)})")


if __name__ == "__main__":
    main()

"""Build DE + DT GAS Score tables.

Outputs:
  data/de_gas_seasons.parquet
  data/dt_gas_seasons.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DE_MASTER = REPO / "data" / "league_de_all_seasons.parquet"
DT_MASTER = REPO / "data" / "league_dt_all_seasons.parquet"
DE_OUT = REPO / "data" / "de_gas_seasons.parquet"
DT_OUT = REPO / "data" / "dt_gas_seasons.parquet"

from lib_dl_gas import DE_SPEC, DT_SPEC, compute_dl_gas  # noqa: E402


def _normalize(master: pd.DataFrame) -> pd.DataFrame:
    if "season" in master.columns and "season_year" in master.columns:
        master = master.drop(columns=["season"])
    elif "season_year" not in master.columns and "season" in master.columns:
        master = master.rename(columns={"season": "season_year"})
    return master


def _build(master_path: Path, spec, out_path: Path,
             label: str) -> None:
    print(f"→ {label} GAS — loading master...")
    master = pd.read_parquet(master_path)
    print(f"  rows: {len(master):,}")
    master = _normalize(master)
    graded = compute_dl_gas(master, spec=spec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    graded.to_parquet(out_path, index=False)
    print(f"  ✓ wrote {out_path.relative_to(REPO)}")
    print()
    print(f"=== Top 12 {label} by GAS, 2024 (≥500 def_snaps) ===")
    cols = ["player_display_name", "recent_team", "position",
            "def_snaps", "gas_score", "gas_label",
            "gas_pass_rush_grade", "gas_run_defense_grade",
            "gas_disruption_grade"]
    cols = [c for c in cols if c in graded.columns]
    s24 = graded[(graded["season_year"] == 2024)
                  & (graded["def_snaps"] >= 500)
                  ].nlargest(12, "gas_score")
    print(s24[cols].to_string(index=False))
    print()
    print(f"=== Lions {label} 2024 ===")
    det = graded[(graded["season_year"] == 2024)
                  & (graded["recent_team"] == "DET")
                  & (graded["def_snaps"] >= 100)].sort_values(
        "gas_score", ascending=False)
    print(det[cols].to_string(index=False))
    print()
    df_y = graded[graded["def_snaps"] >= 400][
        ["player_id", "season_year", "gas_score"]
    ].dropna(subset=["player_id"]).sort_values(
        ["player_id", "season_year"]).copy()
    df_y["next"] = df_y.groupby("player_id")["gas_score"].shift(-1)
    df_y["next_season"] = df_y.groupby("player_id"
                                          )["season_year"].shift(-1)
    yoy = df_y.dropna(subset=["next"])
    yoy = yoy[yoy["next_season"] - yoy["season_year"] == 1]
    print(f"{label} YoY r: {yoy['gas_score'].corr(yoy['next']):.3f}  "
          f"(n={len(yoy)})")


def main() -> None:
    _build(DE_MASTER, DE_SPEC, DE_OUT, "DE")
    print()
    print("=" * 70)
    print()
    _build(DT_MASTER, DT_SPEC, DT_OUT, "DT")


if __name__ == "__main__":
    main()

"""Build OC GAS Score parquet.

Inputs:
    data/master_ocs_with_z.parquet         (career z-scores + adj_z)
    data/master_ocs_2024_with_z.parquet    (single-season z-scores + adj_z)
    data/scheme/oc_fulcrum_profile.parquet (clutch metrics, long format)

Outputs:
    data/oc_gas_career.parquet  — career composite GAS for all 106 OCs
    data/oc_gas_2024.parquet    — single-season GAS for 2024 OCs
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CAREER = REPO / "data" / "master_ocs_with_z.parquet"
SEASON = REPO / "data" / "master_ocs_2024_with_z.parquet"
SEASON_2025 = REPO / "data" / "master_ocs_2025_with_z.parquet"
FULCRUM = REPO / "data" / "scheme" / "oc_fulcrum_profile.parquet"
OUT_CAREER = REPO / "data" / "oc_gas_career.parquet"
OUT_2024 = REPO / "data" / "oc_gas_2024.parquet"
OUT_2025 = REPO / "data" / "oc_gas_2025.parquet"

from lib_oc_gas import compute_oc_gas  # noqa: E402


def _wide_clutch(fulcrum: pd.DataFrame, leverage_def: str = "wp_volatility"
                 ) -> pd.DataFrame:
    """Pivot the clutch profile to wide: one row per OC, with
    clutch_EPA_z, clutch_EPA_adj_z, clutch_EPA_elev_z, clutch_succ_z,
    clutch_succ_adj_z, clutch_succ_elev_z columns."""
    if fulcrum.empty:
        return pd.DataFrame(columns=["oc_name"])
    sub = fulcrum[fulcrum["leverage_def"] == leverage_def].copy()
    rows = {}
    for _, r in sub.iterrows():
        oc = r["oc_name"]
        short = "EPA" if r["metric"] == "epa_per_play" else "succ"
        rows.setdefault(oc, {"oc_name": oc})
        rows[oc][f"clutch_{short}_z"]     = r.get("fulcrum_z")
        rows[oc][f"clutch_{short}_adj_z"] = r.get("fulcrum_adj_z")
        rows[oc][f"clutch_{short}_elev_z"] = r.get("elevation_z")
    return pd.DataFrame(rows.values())


def _build(df_path: Path, out_path: Path, label: str) -> None:
    print(f"→ loading {df_path.relative_to(REPO)}")
    df = pd.read_parquet(df_path).copy()
    fulcrum = pd.read_parquet(FULCRUM) if FULCRUM.exists() else pd.DataFrame()
    print(f"  rows: {len(df):,}  fulcrum rows: {len(fulcrum):,}")

    name_col = "coordinator" if "coordinator" in df.columns else "oc_name"
    if name_col != "oc_name":
        df = df.rename(columns={name_col: "oc_name"})

    clutch = _wide_clutch(fulcrum)
    print(f"  clutch wide rows: {len(clutch):,}")
    df = df.merge(clutch, on="oc_name", how="left")

    if "seasons" not in df.columns:
        df["seasons"] = 1

    graded = compute_oc_gas(df)
    print(f"  graded: {len(graded):,}")

    # Restore the original name column for downstream consumers
    graded = graded.rename(columns={"oc_name": "coordinator"})

    OUT_DIR = out_path.parent
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    graded.to_parquet(out_path, index=False)
    print(f"  ✓ wrote {out_path.relative_to(REPO)}")
    print()
    print(f"=== {label} — Top 12 by GAS Score ===")
    cols_show = ["coordinator", "teams" if "teams" in graded.columns else "team",
                 "seasons", "gas_score", "gas_label", "gas_confidence",
                 "gas_efficiency_grade", "gas_explosiveness_grade",
                 "gas_situational_grade", "gas_clutch_grade"]
    cols_show = [c for c in cols_show if c in graded.columns]
    top = graded.nlargest(12, "gas_score")[cols_show]
    print(top.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print()


def main() -> None:
    if CAREER.exists():
        _build(CAREER, OUT_CAREER, "Career (2016-2025)")
    if SEASON.exists():
        _build(SEASON, OUT_2024, "2024 season")
    if SEASON_2025.exists():
        _build(SEASON_2025, OUT_2025, "2025 season")


if __name__ == "__main__":
    main()

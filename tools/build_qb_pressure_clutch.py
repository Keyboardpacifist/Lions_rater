"""Compute QB pressure + clutch z-scores from qb_dropbacks.parquet.

Output: data/qb_pressure_clutch_z.parquet

Adds five new per-(player, season) z-cols that the QB GAS Score
references but aren't in the existing master table:

  epa_under_pressure_z   — mean dropback EPA when is_pressured == 1
  sack_avoided_rate_z    — 1 - (sacks / pressured dropbacks).
                            Higher = better at escaping pressure.
  third_down_epa_z       — mean EPA on 3rd downs (any distance)
  red_zone_epa_z         — mean EPA inside opponent 20
  late_close_epa_z       — mean EPA in Q4 within one score
                            (is_fourth_quarter & is_one_score)

These are joined onto the existing league_qb_all_seasons table at
build time. Min 100 dropbacks per season for inclusion (otherwise the
sample is too thin and we'd just be measuring noise).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DROPBACKS = REPO / "data" / "qb_dropbacks.parquet"
OUT = REPO / "data" / "qb_pressure_clutch_z.parquet"

MIN_DROPBACKS = 100


def _z_within_season(df: pd.DataFrame, col: str) -> pd.Series:
    """Standard z-score within each season's QB population."""
    means = df.groupby("season")[col].transform("mean")
    stds = df.groupby("season")[col].transform("std").replace(0, np.nan)
    return ((df[col] - means) / stds).fillna(0)


def main() -> None:
    print("→ loading qb_dropbacks...")
    db = pd.read_parquet(DROPBACKS)
    print(f"  {len(db):,} dropbacks")
    print(f"  cols sample: {db.columns[:30].tolist()[:8]}...")

    # Identify QB by passer_player_id (any pbp dropback row has this)
    if "passer_player_id" not in db.columns:
        print("ERROR: no passer_player_id column.")
        return

    db = db.dropna(subset=["passer_player_id", "season"])
    db["season"] = db["season"].astype(int)

    # Per-(qb, season) aggregations
    grp = db.groupby(["passer_player_id", "season"])

    # 1. Total dropbacks for filtering
    n_dropbacks = grp.size().rename("dropbacks")

    # 2. EPA under pressure
    def _epa_under_pressure(g):
        sub = g[g.get("is_pressured", 0) == 1]
        return sub["epa"].mean() if len(sub) else np.nan
    epa_pressure = grp.apply(_epa_under_pressure,
                              include_groups=False).rename(
                                  "epa_under_pressure")

    # 3. Sack-avoided rate = 1 - (sacks / pressured)
    def _sack_avoided(g):
        pressured = g[g.get("is_pressured", 0) == 1]
        if len(pressured) == 0:
            return np.nan
        sack_col = "sack" if "sack" in pressured.columns else None
        if sack_col is None:
            return np.nan
        sack_rate = pressured[sack_col].fillna(0).mean()
        return 1.0 - sack_rate
    sack_avoided = grp.apply(_sack_avoided,
                              include_groups=False).rename(
                                  "sack_avoided_rate")

    # 4. Third-down EPA
    def _third_down_epa(g):
        sub = g[g.get("is_third_down", 0) == 1]
        return sub["epa"].mean() if len(sub) else np.nan
    third_down_epa = grp.apply(_third_down_epa,
                                include_groups=False).rename(
                                    "third_down_epa")

    # 5. RZ EPA
    def _rz_epa(g):
        sub = g[g.get("is_red_zone", 0) == 1]
        return sub["epa"].mean() if len(sub) else np.nan
    rz_epa = grp.apply(_rz_epa,
                        include_groups=False).rename("red_zone_epa")

    # 6. Late-close EPA (Q4 + one score)
    def _late_close_epa(g):
        sub = g[(g.get("is_fourth_quarter", 0) == 1)
                & (g.get("is_one_score", 0) == 1)]
        return sub["epa"].mean() if len(sub) else np.nan
    late_close_epa = grp.apply(_late_close_epa,
                                include_groups=False).rename(
                                    "late_close_epa")

    out = pd.concat([
        n_dropbacks, epa_pressure, sack_avoided,
        third_down_epa, rz_epa, late_close_epa,
    ], axis=1).reset_index()

    print(f"  per-(qb, season) rows: {len(out):,}")

    # Filter to substantial samples
    out = out[out["dropbacks"] >= MIN_DROPBACKS].copy()
    print(f"  after >={MIN_DROPBACKS} dropbacks: {len(out):,}")

    # Z-score each metric within season
    for col in ["epa_under_pressure", "sack_avoided_rate",
                 "third_down_epa", "red_zone_epa",
                 "late_close_epa"]:
        out[f"{col}_z"] = _z_within_season(out, col)

    # Rename keys to match the existing master file (player_id + season_year)
    out = out.rename(columns={
        "passer_player_id": "player_id",
        "season": "season_year",
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("Sample (top 5 by epa_under_pressure_z, 2024):")
    sample = out[out["season_year"] == 2024].nlargest(5,
                                                        "epa_under_pressure_z")
    print(sample[["player_id", "dropbacks",
                   "epa_under_pressure", "epa_under_pressure_z",
                   "sack_avoided_rate", "third_down_epa"]].to_string())


if __name__ == "__main__":
    main()

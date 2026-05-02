"""One-shot: 2024 QB GAS at mobility=12% (current) vs mobility=18%.

Production weights are kept for column A. For column B, mobility is
lifted to 18% and the 6% comes from volume (17% → 11%) — passing
yards/TDs are partly captured by efficiency stats anyway, so trimming
volume is the cleanest cut.

Both columns use the production SOS data (def-LOO + 75% weapons-adj).
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from lib_qb_gas import QB_SPEC, compute_qb_gas  # noqa: E402

MASTER = REPO / "data" / "league_qb_all_seasons.parquet"
PRESSURE = REPO / "data" / "qb_pressure_clutch_z.parquet"
SOS = REPO / "data" / "qb_sos_adjusted_z.parquet"


def main() -> None:
    master = pd.read_parquet(MASTER)
    pressure = pd.read_parquet(PRESSURE)
    sos = pd.read_parquet(SOS)
    merged = master.merge(pressure, on=["player_id", "season_year"],
                            how="left")
    merged = merged.merge(sos, on=["player_id", "season_year"],
                            how="left")

    # Spec A: current (mobility 12%, volume 17%)
    spec_a = QB_SPEC

    # Spec B: mobility 18%, volume trimmed to 11%
    spec_b = deepcopy(QB_SPEC)
    spec_b.bundle_weights = {
        "efficiency":    0.45,
        "volume":        0.11,
        "ball_security": 0.10,
        "pressure":      0.13,
        "mobility":      0.18,
        "clutch":        0.03,
    }
    assert abs(sum(spec_b.bundle_weights.values()) - 1.0) < 1e-9

    # Spec C: mobility 21%, volume trimmed to 8% (same playbook —
    # all 9% of new mobility weight comes from volume)
    spec_c = deepcopy(QB_SPEC)
    spec_c.bundle_weights = {
        "efficiency":    0.45,
        "volume":        0.08,
        "ball_security": 0.10,
        "pressure":      0.13,
        "mobility":      0.21,
        "clutch":        0.03,
    }
    assert abs(sum(spec_c.bundle_weights.values()) - 1.0) < 1e-9

    g_a = compute_qb_gas(merged, spec=spec_a)
    g_b = compute_qb_gas(merged, spec=spec_b)
    g_c = compute_qb_gas(merged, spec=spec_c)

    s24_a = g_a[(g_a["season_year"] == 2024)
                 & (g_a["games"] >= 12)][
        ["player_id", "player_display_name", "gas_score",
         "gas_mobility_grade"]
    ].rename(columns={"gas_score": "score_mob12",
                          "gas_mobility_grade": "mobility"})
    s24_b = g_b[(g_b["season_year"] == 2024)
                 & (g_b["games"] >= 12)][
        ["player_id", "gas_score"]
    ].rename(columns={"gas_score": "score_mob18"})
    s24_c = g_c[(g_c["season_year"] == 2024)
                 & (g_c["games"] >= 12)][
        ["player_id", "gas_score"]
    ].rename(columns={"gas_score": "score_mob21"})

    merged_out = s24_a.merge(s24_b, on="player_id").merge(
        s24_c, on="player_id")
    merged_out["delta_18"] = (merged_out["score_mob18"]
                                - merged_out["score_mob12"])
    merged_out["delta_21"] = (merged_out["score_mob21"]
                                - merged_out["score_mob12"])

    out = merged_out.sort_values("score_mob12", ascending=False).head(20)
    print("=" * 105)
    print("2024 QB GAS  —  mobility 12% / 18% / 21%  "
          "(75% weapons-LOO, def-LOO)")
    print("=" * 105)
    print(out[["player_display_name", "score_mob12", "score_mob18",
                "score_mob21", "delta_18", "delta_21",
                "mobility"]].to_string(index=False))


if __name__ == "__main__":
    main()

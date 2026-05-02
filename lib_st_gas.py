"""Kicker + Punter GAS Score — shared lib, two specs.

Both rely on z-cols already direction-corrected at master-pull time.

Kicker bundles
  Accuracy (50%) — fg_pct_z
  Value (35%)    — fg_epa_z   (EPA captures distance × situation)
  XP (15%)       — xp_pct_z

Punter bundles
  Distance (30%) — avg_distance_z, avg_net_z
  Pinning (40%)  — inside_20_rate_z, pin_rate_z, touchback_rate_z
  Coverage (15%) — fair_catch_rate_z
  Value (15%)    — punt_epa_z
"""
from __future__ import annotations

import pandas as pd

from lib_grade import (
    BundleSpec,
    PositionGradeSpec,
    bundle_grade,
    composite_grade,
    grade_label,
    shrunk_z,
    z_to_grade,
)


NEGATIVE_STATS: set[str] = set()


# ── KICKER ──────────────────────────────────────────────────────

K_ACCURACY = BundleSpec(
    name="Accuracy",
    stats={"fg_pct_z": 1.00},
)

K_VALUE = BundleSpec(
    name="Value",
    stats={"fg_epa_z": 1.00},
)

K_XP = BundleSpec(
    name="XP",
    stats={"xp_pct_z": 1.00},
)

K_SPEC = PositionGradeSpec(
    position="K",
    name_for_grade="GAS Score",
    bundles={"accuracy": K_ACCURACY, "value": K_VALUE, "xp": K_XP},
    bundle_weights={"accuracy": 0.50, "value": 0.35, "xp": 0.15},
)


# ── PUNTER ──────────────────────────────────────────────────────

P_DISTANCE = BundleSpec(
    name="Distance",
    stats={
        "avg_distance_z": 0.50,
        "avg_net_z":      0.50,
    },
)

P_PINNING = BundleSpec(
    name="Pinning",
    stats={
        "inside_20_rate_z":  0.40,
        "pin_rate_z":        0.30,
        "touchback_rate_z":  0.30,
    },
)

P_COVERAGE = BundleSpec(
    name="Coverage",
    stats={"fair_catch_rate_z": 1.00},
)

P_VALUE = BundleSpec(
    name="Value",
    stats={"punt_epa_z": 1.00},
)

P_SPEC = PositionGradeSpec(
    position="P",
    name_for_grade="GAS Score",
    bundles={
        "distance": P_DISTANCE,
        "pinning":  P_PINNING,
        "coverage": P_COVERAGE,
        "value":    P_VALUE,
    },
    bundle_weights={
        "distance": 0.30,
        "pinning":  0.40,
        "coverage": 0.15,
        "value":    0.15,
    },
)


def _stat_grade(z_value: float, stat_name: str) -> float:
    if z_value is None or (isinstance(z_value, float)
                            and z_value != z_value):
        return 50.0
    z = float(z_value)
    if stat_name in NEGATIVE_STATS:
        z = -z
    return z_to_grade(z)


def compute_st_gas(df: pd.DataFrame,
                     spec: PositionGradeSpec,
                     min_games_full_grade: int = 12,
                     shrinkage_tau: float = 4.0,
                     ) -> pd.DataFrame:
    out = df.copy()
    games_col = "games" if "games" in out.columns else "games_played"

    all_stats = set()
    for bundle in spec.bundles.values():
        all_stats.update(bundle.stats.keys())
    for stat in all_stats:
        col = f"_{stat}_grade"
        if stat not in out.columns:
            out[col] = 50.0
        else:
            def _grade_one(row, _stat=stat):
                z = row.get(_stat)
                games = row.get(games_col, 14)
                if games is None or (isinstance(games, float)
                                       and games != games):
                    games = 14
                z_shrunk = shrunk_z(z, int(games), prior_z=0.0,
                                       tau=shrinkage_tau)
                return _stat_grade(z_shrunk, _stat)
            out[col] = out.apply(_grade_one, axis=1)

    for bundle_key, bundle in spec.bundles.items():
        col = f"gas_{bundle_key}_grade"
        out[col] = out.apply(
            lambda r: bundle_grade(
                stat_grades={s: r.get(f"_{s}_grade", 50.0)
                              for s in bundle.stats.keys()},
                weights=bundle.stats,
            ),
            axis=1,
        )

    out["gas_score"] = out.apply(
        lambda r: composite_grade(
            bundle_grades={k: r.get(f"gas_{k}_grade", 50.0)
                            for k in spec.bundles.keys()},
            bundle_weights=spec.bundle_weights,
        ),
        axis=1,
    )
    out["gas_label"] = out["gas_score"].apply(grade_label)

    if games_col in out.columns:
        def _conf(n):
            n = int(n) if pd.notna(n) else 0
            if n >= 16:
                return "HIGH"
            if n >= min_games_full_grade:
                return "MEDIUM"
            return "LOW"
        out["gas_confidence"] = out[games_col].apply(_conf)
    else:
        out["gas_confidence"] = "MEDIUM"

    out = out.drop(columns=[c for c in out.columns
                             if c.startswith("_") and c.endswith("_grade")])
    return out

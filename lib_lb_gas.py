"""LB GAS Score — bundle-weighted, season-level LB grade.

Covers ILB, MLB, OLB. CAVEATS:
1. OLB / ILB ROLES DIFFER. 3-4 OLB ≈ edge rusher; ILB / MLB are
   traditional run-stoppers. Same spec for all v1 — bundle weights
   are designed so that elite pass-rushing OLBs (Parsons) AND elite
   ILBs (Roquan Smith) can both grade highly through different
   bundles.
2. NO COVERAGE-MATCHUP ADJUSTMENT (same as CB / Safety).

Bundles
-------
  Run defense / tackling (30%)
    tackles_per_game_z   0.30
    tfl_per_game_z       0.25
    solo_tackle_rate_z   0.20
    missed_tackle_pct_z  0.25  (already direction-corrected)

  Pass rush (25%)
    sacks_per_game_z         0.30
    qb_hits_per_game_z       0.25
    pressures_per_game_z     0.25
    hurries_per_game_z       0.20

  Coverage (20%)
    completion_pct_allowed_z   0.40
    yards_per_target_allowed_z 0.35
    passer_rating_allowed_z    0.25

  Ball production (15%)
    passes_defended_per_game_z 0.40
    interceptions_per_game_z   0.30
    forced_fumbles_per_game_z  0.30

  Volume / role (10%)
    tackles_per_snap_z 1.00

Min 250 def_snaps for full grade.
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


RUN_DEFENSE = BundleSpec(
    name="Run defense / tackling",
    stats={
        "tackles_per_game_z":   0.30,
        "tfl_per_game_z":       0.25,
        "solo_tackle_rate_z":   0.20,
        "missed_tackle_pct_z":  0.25,
    },
)

PASS_RUSH = BundleSpec(
    name="Pass rush",
    stats={
        "sacks_per_game_z":     0.30,
        "qb_hits_per_game_z":   0.25,
        "pressures_per_game_z": 0.25,
        "hurries_per_game_z":   0.20,
    },
)

COVERAGE = BundleSpec(
    name="Coverage",
    stats={
        "completion_pct_allowed_z":   0.40,
        "yards_per_target_allowed_z": 0.35,
        "passer_rating_allowed_z":    0.25,
    },
)

BALL_PRODUCTION = BundleSpec(
    name="Ball production",
    stats={
        "passes_defended_per_game_z": 0.40,
        "interceptions_per_game_z":   0.30,
        "forced_fumbles_per_game_z":  0.30,
    },
)

VOLUME_ROLE = BundleSpec(
    name="Volume / role",
    stats={
        "tackles_per_snap_z": 1.00,
    },
)


LB_SPEC = PositionGradeSpec(
    position="LB",
    name_for_grade="GAS Score",
    bundles={
        "run_defense":     RUN_DEFENSE,
        "pass_rush":       PASS_RUSH,
        "coverage":        COVERAGE,
        "ball_production": BALL_PRODUCTION,
        "volume_role":     VOLUME_ROLE,
    },
    bundle_weights={
        "run_defense":     0.30,
        "pass_rush":       0.25,
        "coverage":        0.20,
        "ball_production": 0.15,
        "volume_role":     0.10,
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


def compute_lb_gas(df: pd.DataFrame,
                     spec: PositionGradeSpec = LB_SPEC,
                     min_snaps_full_grade: int = 250,
                     shrinkage_tau: float = 4.0,
                     ) -> pd.DataFrame:
    out = df.copy()
    games_col = "games" if "games" in out.columns else "games_snap"

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

    snaps_col = "def_snaps" if "def_snaps" in out.columns else None
    if snaps_col is not None:
        def _conf(n):
            n = int(n) if pd.notna(n) else 0
            if n >= 700:
                return "HIGH"
            if n >= min_snaps_full_grade:
                return "MEDIUM"
            return "LOW"
        out["gas_confidence"] = out[snaps_col].apply(_conf)
    else:
        out["gas_confidence"] = "MEDIUM"

    out = out.drop(columns=[c for c in out.columns
                             if c.startswith("_") and c.endswith("_grade")])
    return out

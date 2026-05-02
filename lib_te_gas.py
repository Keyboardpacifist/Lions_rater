"""TE GAS Score — bundle-weighted, season-level TE grade.

IMPORTANT CAVEAT: TE has two jobs (receiving + blocking). Free
public data covers receiving thoroughly but blocking thinly. v1
grades RECEIVING ONLY. Blocking-only TEs (Y-TEs, Pharaoh Brown
types) will be undergraded — we'll document this explicitly in
the UI when we render the score.

Spec — five bundles. Same Path A discipline. Volume_role gets
heavy weight (same as WR) so role-defining alpha TEs don't lose
to low-volume efficient ones.

  Per-target efficiency (30%)
    SOS-adjusted via per-player leave-one-out.
      adj_epa_per_target_z       0.30
      adj_yards_per_target_z     0.25
      racr_z                     0.15
      adj_success_rate_z         0.15
      avg_cpoe_z                 0.15

  Volume / role (30%)
      target_share_z      0.40
      air_yards_share_z   0.30
      wopr_z              0.30

  YAC (15%)
    YAC matters more for TEs than WRs (slants, crossers, RAC).
      yac_above_exp_z       0.60
      yac_per_reception_z   0.40

  Coverage-beating (10%)
      avg_separation_z 1.00

  Scoring + chains (15%)
      rec_tds_z         0.55
      first_down_rate_z 0.45

  (No blocking bundle — flagged as a known v1 limitation.)
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


PER_TARGET_EFFICIENCY = BundleSpec(
    name="Per-target efficiency",
    stats={
        "adj_epa_per_target_z":   0.30,
        "adj_yards_per_target_z": 0.25,
        "racr_z":                 0.15,
        "adj_success_rate_z":     0.15,
        "avg_cpoe_z":             0.15,
    },
)

VOLUME_ROLE = BundleSpec(
    name="Volume / role",
    stats={
        "target_share_z":    0.40,
        "air_yards_share_z": 0.30,
        "wopr_z":            0.30,
    },
)

YAC = BundleSpec(
    name="YAC",
    stats={
        "yac_above_exp_z":     0.60,
        "yac_per_reception_z": 0.40,
    },
)

COVERAGE_BEATING = BundleSpec(
    name="Coverage-beating",
    stats={
        "avg_separation_z": 1.00,
    },
)

SCORING_CHAINS = BundleSpec(
    name="Scoring + chains",
    stats={
        "rec_tds_z":         0.55,
        "first_down_rate_z": 0.45,
    },
)


TE_SPEC = PositionGradeSpec(
    position="TE",
    name_for_grade="GAS Score",
    bundles={
        "per_target_efficiency": PER_TARGET_EFFICIENCY,
        "volume_role":           VOLUME_ROLE,
        "yac":                   YAC,
        "coverage_beating":      COVERAGE_BEATING,
        "scoring_chains":        SCORING_CHAINS,
    },
    bundle_weights={
        "per_target_efficiency": 0.30,
        "volume_role":           0.30,
        "yac":                   0.15,
        "coverage_beating":      0.10,
        "scoring_chains":        0.15,
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


def compute_te_gas(df: pd.DataFrame,
                     spec: PositionGradeSpec = TE_SPEC,
                     min_games_full_grade: int = 10,
                     shrinkage_tau: float = 4.0,
                     ) -> pd.DataFrame:
    out = df.copy()

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
                games = row.get("games", 14)
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

    if "games" in out.columns:
        def _conf(n):
            n = int(n) if pd.notna(n) else 0
            if n >= 14:
                return "HIGH"
            if n >= min_games_full_grade:
                return "MEDIUM"
            return "LOW"
        out["gas_confidence"] = out["games"].apply(_conf)
    else:
        out["gas_confidence"] = "MEDIUM"

    out = out.drop(columns=[c for c in out.columns
                             if c.startswith("_") and c.endswith("_grade")])
    return out

"""College GAS Score — simpler specs than NFL counterparts.

College data (CFBD) lacks per-target efficiency, separation, and
SOS-adjusted stats. Specs use what's available: counting stats +
basic rates. Limitations:

1. NO SOS ADJUSTMENT. CFBD splits FBS / FCS / G5 / P5 levels but
   we don't subtract opponent quality at v1. Future v1.1 could
   layer in conference / opponent strength.
2. NO COMBINE/PRO-DAY INTEGRATION HERE — those live separately
   and are used in scouting comps, not the GAS score.
3. THRESHOLDS DIFFER per position (carries / targets / attempts)
   to give meaningful samples without excluding part-time players.

Rule of thumb: college GAS should be read as "production within his
program / conference cohort" rather than "talent translation to NFL"
— the latter belongs to comps + scouting modules.
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


# int_rate_z direction depends on whether master pre-flipped. We
# flip here to be safe — lower INT rate = better. If master is
# already flipped, this would invert; verify first run.
NEGATIVE_STATS_QB: set[str] = {"int_rate_z"}
NEGATIVE_STATS_RB: set[str] = set()
NEGATIVE_STATS_WR: set[str] = set()
NEGATIVE_STATS_TE: set[str] = set()


# ── COLLEGE QB ────────────────────────────────────────────────

CQB_EFFICIENCY = BundleSpec(
    name="Efficiency",
    stats={
        "completion_pct_z":   0.45,
        "yards_per_attempt_z": 0.55,
    },
)

CQB_PRODUCTION = BundleSpec(
    name="Production",
    stats={
        "pass_tds_z": 0.45,
        "td_rate_z":  0.55,
    },
)

CQB_BALL_SECURITY = BundleSpec(
    name="Ball security",
    stats={"int_rate_z": 1.00},
)

CQB_MOBILITY = BundleSpec(
    name="Mobility",
    stats={"rush_yards_total_z": 1.00},
)

COLLEGE_QB_SPEC = PositionGradeSpec(
    position="College QB",
    name_for_grade="GAS Score",
    bundles={
        "efficiency":    CQB_EFFICIENCY,
        "production":    CQB_PRODUCTION,
        "ball_security": CQB_BALL_SECURITY,
        "mobility":      CQB_MOBILITY,
    },
    bundle_weights={
        "efficiency":    0.50,
        "production":    0.30,
        "ball_security": 0.10,
        "mobility":      0.10,
    },
)


# ── COLLEGE RB ────────────────────────────────────────────────

CRB_EFFICIENCY = BundleSpec(
    name="Efficiency",
    stats={"yards_per_carry_z": 1.00},
)

CRB_PRODUCTION = BundleSpec(
    name="Production",
    stats={
        "rush_yards_total_z": 0.50,
        "rush_tds_total_z":   0.30,
        "total_tds_z":        0.20,
    },
)

CRB_RECEIVING = BundleSpec(
    name="Receiving",
    stats={
        "receptions_total_z": 0.50,
        "rec_yards_total_z":  0.50,
    },
)

COLLEGE_RB_SPEC = PositionGradeSpec(
    position="College RB",
    name_for_grade="GAS Score",
    bundles={
        "efficiency": CRB_EFFICIENCY,
        "production": CRB_PRODUCTION,
        "receiving":  CRB_RECEIVING,
    },
    bundle_weights={
        "efficiency": 0.35,
        "production": 0.45,
        "receiving":  0.20,
    },
)


# ── COLLEGE WR ────────────────────────────────────────────────

CWR_PRODUCTION = BundleSpec(
    name="Production",
    stats={
        "rec_yards_total_z":  0.40,
        "rec_tds_total_z":    0.30,
        "receptions_total_z": 0.30,
    },
)

CWR_EFFICIENCY = BundleSpec(
    name="Efficiency",
    stats={
        "yards_per_rec_z": 0.60,
        "rec_long_z":      0.40,
    },
)

COLLEGE_WR_SPEC = PositionGradeSpec(
    position="College WR",
    name_for_grade="GAS Score",
    bundles={
        "production": CWR_PRODUCTION,
        "efficiency": CWR_EFFICIENCY,
    },
    bundle_weights={
        "production": 0.65,
        "efficiency": 0.35,
    },
)


# ── COLLEGE TE ────────────────────────────────────────────────

CTE_PRODUCTION = BundleSpec(
    name="Production",
    stats={
        "rec_yards_total_z":  0.40,
        "rec_tds_total_z":    0.30,
        "receptions_total_z": 0.30,
    },
)

CTE_EFFICIENCY = BundleSpec(
    name="Efficiency",
    stats={"yards_per_rec_z": 1.00},
)

COLLEGE_TE_SPEC = PositionGradeSpec(
    position="College TE",
    name_for_grade="GAS Score",
    bundles={
        "production": CTE_PRODUCTION,
        "efficiency": CTE_EFFICIENCY,
    },
    bundle_weights={
        "production": 0.70,
        "efficiency": 0.30,
    },
)


# ── COMPUTE ────────────────────────────────────────────────────

def _stat_grade(z_value: float, stat_name: str,
                  negative_stats: set[str]) -> float:
    if z_value is None or (isinstance(z_value, float)
                            and z_value != z_value):
        return 50.0
    z = float(z_value)
    if stat_name in negative_stats:
        z = -z
    return z_to_grade(z)


def compute_college_gas(df: pd.DataFrame,
                          spec: PositionGradeSpec,
                          negative_stats: set[str],
                          min_games_full_grade: int = 8,
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
                games = row.get(games_col, 12)
                if games is None or (isinstance(games, float)
                                       and games != games):
                    games = 12
                z_shrunk = shrunk_z(z, int(games), prior_z=0.0,
                                       tau=shrinkage_tau)
                return _stat_grade(z_shrunk, _stat, negative_stats)
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
            if n >= 12:
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

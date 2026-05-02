"""Safety GAS Score — bundle-weighted, season-level safety grade.

CAVEATS — public-data limits.
1. NO SOS/MATCHUP ADJUSTMENT. Same as CB — without route charting we
   can't isolate the WR/TE the safety covered.
2. STRONG vs FREE SAFETY ROLE NOT DIFFERENTIATED. SS plays more
   in-the-box, FS plays deep. Same GAS spec for both. Future v1.1
   could split by `position` ('FS' vs 'SS') with role-weighted bundles.
3. NO DEEP-COVERAGE / SPLIT-SAFETY scheme metric. Two-deep safeties
   look "less involved" by raw stats — they're doing their job
   when nothing happens.

Bundles
-------
  Coverage (40%) — what they allowed when targeted
    completion_pct_allowed_z   0.40
    yards_per_target_allowed_z 0.35
    passer_rating_allowed_z    0.25

  Run support / tackling (25%)
    solo_tackle_rate_z   0.30
    missed_tackle_pct_z  0.30  (already direction-corrected)
    tfl_per_game_z       0.20
    tackles_per_snap_z   0.20

  Ball production (25%)
    passes_defended_per_game_z 0.50
    interceptions_per_game_z   0.50

  Playmaking / blitz (10%)
    sacks_per_game_z          0.55
    forced_fumbles_per_game_z 0.45

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


COVERAGE = BundleSpec(
    name="Coverage",
    stats={
        "completion_pct_allowed_z":   0.40,
        "yards_per_target_allowed_z": 0.35,
        "passer_rating_allowed_z":    0.25,
    },
)

RUN_SUPPORT = BundleSpec(
    name="Run support / tackling",
    stats={
        "solo_tackle_rate_z":  0.30,
        "missed_tackle_pct_z": 0.30,
        "tfl_per_game_z":      0.20,
        "tackles_per_snap_z":  0.20,
    },
)

BALL_PRODUCTION = BundleSpec(
    name="Ball production",
    stats={
        "passes_defended_per_game_z": 0.50,
        "interceptions_per_game_z":   0.50,
    },
)

PLAYMAKING = BundleSpec(
    name="Playmaking / blitz",
    stats={
        "sacks_per_game_z":         0.55,
        "forced_fumbles_per_game_z": 0.45,
    },
)


SAFETY_SPEC = PositionGradeSpec(
    position="Safety",
    name_for_grade="GAS Score",
    bundles={
        "coverage":        COVERAGE,
        "run_support":     RUN_SUPPORT,
        "ball_production": BALL_PRODUCTION,
        "playmaking":      PLAYMAKING,
    },
    bundle_weights={
        "coverage":        0.40,
        "run_support":     0.25,
        "ball_production": 0.25,
        "playmaking":      0.10,
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


def compute_safety_gas(df: pd.DataFrame,
                         spec: PositionGradeSpec = SAFETY_SPEC,
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

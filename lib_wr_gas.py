"""WR GAS Score — bundle-weighted, season-level WR grade.

Spec (v1) — six bundles. Same Path A discipline.

  Per-target efficiency (40%)
    The core WR job — produce on every target. SOS-adjusted versions
    used where available.
      adj_epa_per_target_z       0.30  (SOS-adjusted)
      adj_yards_per_target_z     0.25
      racr_z                     0.15  (RACR — air yards conversion)
      adj_success_rate_z         0.15  (SOS-adjusted)
      avg_cpoe_z                 0.15  (NOTE: this is QB CPOE on this
                                          WR's targets — high = QB
                                          throws catchable balls to him)

  Volume / role (20%)
    Where does this WR sit in his offense's pecking order?
      target_share_z      0.40
      air_yards_share_z   0.30
      wopr_z              0.30  (composite opportunity rating)

  Coverage-beating (10%)
    Pure player-isolated route-running ability (NGS). Same all year.
      avg_separation_z 1.00

  YAC (10%)
    What he does AFTER the catch. Player-isolated.
      yac_above_exp_z       0.60  (NGS — YAC above expected per catch)
      yac_per_reception_z   0.40

  Scoring + chains (15%)
    Touchdowns + first-down rate.
      rec_tds_z         0.55
      first_down_rate_z 0.45

  Catch quality (5%)
    Drop rate proxy. Noisier than other bundles.
      catch_rate_z 1.00
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


# ── Bundle definitions ────────────────────────────────────────────

PER_TARGET_EFFICIENCY = BundleSpec(
    name="Per-target efficiency",
    stats={
        # SOS-adjusted via PER-PLAYER leave-one-out (build_wr_sos_adjusted)
        # The opp baseline excludes ALL of this player's plays vs that
        # defense, breaking the chicken-and-egg of "elite WRs inflate
        # their own opponents' allowance baseline."
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

COVERAGE_BEATING = BundleSpec(
    name="Coverage-beating",
    stats={
        "avg_separation_z": 1.00,
    },
)

YAC = BundleSpec(
    name="YAC",
    stats={
        "yac_above_exp_z":     0.60,
        "yac_per_reception_z": 0.40,
    },
)

SCORING_CHAINS = BundleSpec(
    name="Scoring + chains",
    stats={
        "rec_tds_z":         0.55,
        "first_down_rate_z": 0.45,
    },
)

CATCH_QUALITY = BundleSpec(
    name="Catch quality",
    stats={
        "catch_rate_z": 1.00,
    },
)


WR_SPEC = PositionGradeSpec(
    position="WR",
    name_for_grade="GAS Score",
    bundles={
        "per_target_efficiency": PER_TARGET_EFFICIENCY,
        "volume_role":           VOLUME_ROLE,
        "coverage_beating":      COVERAGE_BEATING,
        "yac":                   YAC,
        "scoring_chains":        SCORING_CHAINS,
        "catch_quality":         CATCH_QUALITY,
    },
    # Bundle weights calibrated for football importance × YoY stability.
    # Per-bundle YoY (NFL pooled 2017-2025, ≥10g):
    #   Volume / role          0.81  ← extremely stable; targets-leader sticky
    #   Coverage-beating       0.55  (separation = real trait)
    #   Scoring + chains       0.49
    #   YAC                    0.37
    #   Per-target efficiency  0.35  (depends on QB throws, scheme, matchups)
    #   Catch quality          0.33
    #
    # Why volume_role weight is high (30%):
    # A WR's total value to his team scales with volume × efficiency. A
    # deep-specialist who's elite per-target on 60 targets isn't a top-5
    # NFL WR; a high-volume target hog with above-avg per-target IS.
    # Volume-role captures "is this player actually a #1 in his offense,"
    # which prevents low-volume efficient role players from ranking
    # above true alphas.
    bundle_weights={
        "per_target_efficiency": 0.30,
        "volume_role":           0.30,
        "coverage_beating":      0.10,
        "yac":                   0.10,
        "scoring_chains":        0.15,
        "catch_quality":         0.05,
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


def compute_wr_gas(df: pd.DataFrame,
                     spec: PositionGradeSpec = WR_SPEC,
                     min_games_full_grade: int = 10,
                     shrinkage_tau: float = 4.0,
                     ) -> pd.DataFrame:
    """Add bundle grades + composite GAS Score for each WR-season."""
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
                z_shrunk = shrunk_z(z, int(games),
                                       prior_z=0.0,
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

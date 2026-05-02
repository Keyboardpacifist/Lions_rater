"""QB GAS Score — bundle-weighted, season-level QB grade.

Spec
----
Six bundles, transparent weights:

  Per-dropback efficiency (40%)
    pass_epa_per_play_z, passing_cpoe_z, pass_success_rate_z, sack_rate_z
    (sack rate is "lower is better" — z is auto-flipped)

  Volume (15%)
    passing_yards_per_game_z, passing_tds_per_game_z

  Ball security (15%)
    int_rate_z, turnover_rate_z (both "lower is better" — z auto-flipped)

  Pressure performance (15%) — NEW vs. existing slider columns,
    computed in build_qb_pressure_clutch.py from qb_dropbacks
    epa_under_pressure_z, sack_avoided_rate_z

  Mobility (10%)
    rush_yards_per_game_z, rush_epa_per_carry_z

  Clutch / situational (5%) — NEW
    third_down_epa_z, red_zone_epa_z, late_close_epa_z

Public API
----------
    QB_SPEC                  — the PositionGradeSpec
    compute_qb_gas(df)       — takes a wide-format z-cols DataFrame,
                                 returns the same df with bundle grades
                                 + composite GAS Score columns added
    NEGATIVE_STATS           — set of stats where lower is better
                                 (z must be flipped before scoring)
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


# Stats where a NEGATIVE z is better than a positive z (lower stat
# value = better). NOTE: in the existing league_qb_all_seasons.parquet,
# the *_z columns are ALREADY direction-corrected (sign-flipped at
# build time), so higher z always = better player. We keep this set
# here for any future raw-z input source that doesn't pre-correct,
# but it's empty for the current pipeline.
NEGATIVE_STATS: set[str] = set()


# ── Bundle definitions ─────────────────────────────────────────────
# Each bundle's `stats` dict gives the relative weight of each stat-z
# WITHIN the bundle. The bundle's weight in the composite is set in
# `bundle_weights` below.

EFFICIENCY = BundleSpec(
    name="Per-dropback efficiency",
    stats={
        # SOS-adjusted (per-dropback opponent-defense subtraction).
        # Original unadjusted z-cols kept available in the master file
        # for the slider/community algorithms.
        "adj_pass_epa_per_play_z": 0.45,    # the single best QB stat
        "passing_cpoe_z":          0.20,    # already difficulty-adj at play level
        "adj_pass_success_rate_z": 0.20,    # consistency-of-positive plays
        "adj_sack_rate_z":         0.15,    # quick decisions / pocket awareness
    },
)

VOLUME = BundleSpec(
    name="Volume",
    stats={
        "passing_yards_per_game_z": 0.55,
        "passing_tds_per_game_z":   0.45,
    },
)

BALL_SECURITY = BundleSpec(
    name="Ball security",
    stats={
        "adj_int_rate_z":  0.55,    # SOS-adjusted INT rate
        "turnover_rate_z": 0.45,    # not yet SOS-adjusted (fumbles)
    },
)

PRESSURE = BundleSpec(
    name="Pressure performance",
    stats={
        # Computed by build_qb_pressure_clutch.py
        "epa_under_pressure_z":   0.60,
        "sack_avoided_rate_z":    0.40,
    },
)

MOBILITY = BundleSpec(
    name="Mobility",
    stats={
        "rush_yards_per_game_z": 0.55,
        "rush_epa_per_carry_z":  0.45,
    },
)

CLUTCH = BundleSpec(
    name="Clutch / situational",
    stats={
        "third_down_epa_z":  0.40,
        "red_zone_epa_z":    0.35,
        "late_close_epa_z":  0.25,
    },
)

QB_SPEC = PositionGradeSpec(
    position="QB",
    name_for_grade="GAS Score",
    bundles={
        "efficiency":     EFFICIENCY,
        "volume":         VOLUME,
        "ball_security":  BALL_SECURITY,
        "pressure":       PRESSURE,
        "mobility":       MOBILITY,
        "clutch":         CLUTCH,
    },
    # Bundle weights calibrated to balance two principles:
    # (a) football importance to winning football games
    # (b) the stat's actual season-to-season stability
    #
    # Per-bundle YoY (NFL pooled 2017-2025, ≥14 games):
    #   Mobility       0.71  ← very stable trait
    #   Volume         0.48
    #   Efficiency     0.34
    #   Pressure       0.29
    #   Clutch         0.16  ← sparse-situation noise
    #   Ball security  0.05  ← essentially random YoY (well-documented
    #                          in football analytics — INT rate is the
    #                          single noisiest core QB stat)
    #
    # We DON'T zero out ball security — it does affect this-season
    # outcomes, just doesn't predict next year. We weight it modestly
    # so noisy single-season fluctuations don't dominate. Same logic
    # for clutch: meaningful when it happens, but small-sample noisy.
    # Mobility lifted 12 → 21 (June 2026 rev): the modern game (Lamar,
    # Allen, Hurts, Daniels) makes rushing a primary QB skill, and 12%
    # was undergrading dual-threats vs pocket statues. The 9% comes from
    # volume (passing yards/TDs are partly captured in efficiency
    # already, so volume was double-counted at 17%).
    bundle_weights={
        "efficiency":     0.45,    # stable + decisive
        "volume":         0.08,    # trimmed to fund mobility increase
        "ball_security":  0.10,    # high noise — keep modest
        "pressure":       0.13,
        "mobility":       0.21,    # MODERN-GAME WEIGHTING (was 0.12)
        "clutch":         0.03,    # high noise — keep small
    },
)


# ── Compute ────────────────────────────────────────────────────────

def _stat_grade(z_value: float, stat_name: str) -> float:
    """Convert one z-value to a 0-100 grade. For negative-direction
    stats (sack rate, INT rate), flip the sign first."""
    if z_value is None:
        return 50.0
    try:
        if z_value != z_value:  # NaN check
            return 50.0
    except Exception:
        return 50.0
    z = float(z_value)
    if stat_name in NEGATIVE_STATS:
        z = -z
    return z_to_grade(z)


def compute_qb_gas(df: pd.DataFrame,
                     spec: PositionGradeSpec = QB_SPEC,
                     min_games_full_grade: int = 8,
                     shrinkage_tau: float = 4.0,
                     ) -> pd.DataFrame:
    """For a DataFrame of (player, season) rows with z-score columns,
    add bundle-grade columns and the composite GAS Score column.

    Sample-size shrinkage: each stat z is shrunk toward 0 (=league
    average) with weight `tau / (n + tau)` where n is games played.
    Tau=4 means a 17-game season gets 81% weight on the actual z, 19%
    on the league-mean prior. A 10-game season gets 71%/29%. This is
    Path A integrity — thin samples regress to mean honestly rather
    than producing wild grades that overreact to small-sample noise.

    Required input columns: every z-column referenced by any bundle in
    `spec.bundles`, plus `games`. Missing z-columns → grade of 50.

    Adds these columns:
        gas_efficiency_grade, gas_volume_grade, gas_ball_security_grade,
        gas_pressure_grade, gas_mobility_grade, gas_clutch_grade,
        gas_score, gas_label, gas_confidence
    """
    out = df.copy()

    # 1. Compute each stat's individual grade once.
    # Each stat z is shrunk by sample size before scoring.
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
                games = row.get("games", 16)
                if games is None or games != games:
                    games = 16
                z_shrunk = shrunk_z(z, int(games),
                                       prior_z=0.0,
                                       tau=shrinkage_tau)
                return _stat_grade(z_shrunk, _stat)
            out[col] = out.apply(_grade_one, axis=1)

    # 2. For each row, compute each bundle's grade
    bundle_grade_cols: list[str] = []
    for bundle_key, bundle in spec.bundles.items():
        col = f"gas_{bundle_key}_grade"
        bundle_grade_cols.append(col)
        out[col] = out.apply(
            lambda r: bundle_grade(
                stat_grades={s: r.get(f"_{s}_grade", 50.0)
                              for s in bundle.stats.keys()},
                weights=bundle.stats,
            ),
            axis=1,
        )

    # 3. Composite GAS Score
    out["gas_score"] = out.apply(
        lambda r: composite_grade(
            bundle_grades={k: r.get(f"gas_{k}_grade", 50.0)
                            for k in spec.bundles.keys()},
            bundle_weights=spec.bundle_weights,
        ),
        axis=1,
    )

    out["gas_label"] = out["gas_score"].apply(grade_label)

    # 4. Confidence — based on games played
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

    # Cleanup intermediate columns
    out = out.drop(columns=[c for c in out.columns
                             if c.startswith("_") and c.endswith("_grade")])
    return out

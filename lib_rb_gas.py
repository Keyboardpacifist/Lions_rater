"""RB GAS Score — bundle-weighted, season-level RB grade.

Spec (v1)
---------
Six bundles. Weights iterated against per-bundle YoY stability +
football importance, same Path A discipline as QB.

  Rushing efficiency (40%)
    The core RB job. Mix of player-isolated (RYOE, YAC/att) and
    SOS-adjusted (EPA, success rate, explosive rate).
      adj_epa_per_rush_z         0.30  (SOS-adjusted)
      ryoe_per_att_z             0.25  (NGS — already player-isolated)
      yards_after_contact_per_att_z 0.20
      adj_rush_success_rate_z    0.15  (SOS-adjusted)
      broken_tackles_per_att_z   0.10

  Receiving (20%)
    Modern RBs catch real volume. Not SOS-adjusted in v1
    (RB-receiving is dominated by QB quality, not opp pass def).
      rec_epa_per_target_z   0.30
      rec_yards_per_target_z 0.25
      targets_per_game_z     0.20
      yac_per_reception_z    0.15
      rec_tds_z              0.10

  Volume + durability (15%)
    Workload + games. Snap share captures three-down RB role.
      touches_per_game_z 0.55
      snap_share_z       0.45

  Explosiveness (10%)
    Big-play ability — separated for tunability since it's
    less stable than efficiency.
      adj_explosive_run_rate_z  0.55
      explosive_15_rate_z       0.45

  Red-zone production (10%)
    Goal-line conversion, RZ usage share — meaningful for both
    fantasy and game value.
      goal_line_td_rate_z 0.50
      rz_carry_share_z    0.30
      rush_tds_z          0.20

  Short-yardage power (5%)
      short_yardage_conv_rate_z 1.0

Public API
----------
    RB_SPEC                 — the PositionGradeSpec
    compute_rb_gas(df)      — same signature as compute_qb_gas
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


NEGATIVE_STATS: set[str] = set()  # master file already direction-corrected


# ── Bundle definitions ─────────────────────────────────────────────

RUSHING_EFFICIENCY = BundleSpec(
    name="Rushing efficiency",
    stats={
        "adj_epa_per_rush_z":             0.30,  # SOS-adjusted
        "ryoe_per_att_z":                 0.25,  # NGS player-isolated
        "yards_after_contact_per_att_z":  0.20,  # post-contact effort
        "adj_rush_success_rate_z":        0.15,  # SOS-adjusted
        "broken_tackles_per_att_z":       0.10,  # elusiveness
    },
)

RECEIVING = BundleSpec(
    name="Receiving",
    stats={
        "rec_epa_per_target_z":   0.30,
        "rec_yards_per_target_z": 0.25,
        "targets_per_game_z":     0.20,
        "yac_per_reception_z":    0.15,
        "rec_tds_z":              0.10,
    },
)

VOLUME_DURABILITY = BundleSpec(
    name="Volume + durability",
    stats={
        "touches_per_game_z": 0.55,
        "snap_share_z":       0.45,
    },
)

EXPLOSIVENESS = BundleSpec(
    name="Explosiveness",
    stats={
        "adj_explosive_run_rate_z": 0.55,  # SOS-adjusted
        "explosive_15_rate_z":      0.45,
    },
)

RED_ZONE = BundleSpec(
    name="Red-zone production",
    stats={
        "goal_line_td_rate_z": 0.50,
        "rz_carry_share_z":    0.30,
        "rush_tds_z":          0.20,
    },
)

SHORT_YARDAGE = BundleSpec(
    name="Short-yardage power",
    stats={
        "short_yardage_conv_rate_z": 1.00,
    },
)


RB_SPEC = PositionGradeSpec(
    position="RB",
    name_for_grade="GAS Score",
    bundles={
        "rushing_efficiency":  RUSHING_EFFICIENCY,
        "receiving":           RECEIVING,
        "volume_durability":   VOLUME_DURABILITY,
        "explosiveness":       EXPLOSIVENESS,
        "red_zone":            RED_ZONE,
        "short_yardage":       SHORT_YARDAGE,
    },
    # Bundle weights calibrated for football importance × YoY stability.
    # Per-bundle YoY (NFL pooled 2017-2025, ≥10 games):
    #   Volume + durability  0.70  ← very stable (workload is sticky)
    #   Receiving            0.29
    #   Explosiveness        0.22  ← noisy
    #   Rushing efficiency   0.21  ← noisier than expected (OL variance)
    #   Red zone             0.03  ← noise (TD vulturing)
    #   Short yardage        0.01  ← noise
    #
    # Volume + durability does the heavy lifting on YoY — and football-
    # wise it captures the "three-down workhorse" role which IS the most
    # valuable RB archetype. Rushing efficiency stays the largest per-
    # touch weight even though it's noisy (the metric matters; we just
    # accept that one season isn't always representative).
    bundle_weights={
        "rushing_efficiency":  0.35,    # per-touch quality
        "volume_durability":   0.25,    # up — stable + valuable
        "receiving":           0.22,    # modern RB job
        "explosiveness":       0.08,    # noise + already partially in efficiency
        "red_zone":            0.06,    # noise — small weight
        "short_yardage":       0.04,    # noise — small weight
    },
)


# ── Compute (mirrors QB version) ───────────────────────────────────

def _stat_grade(z_value: float, stat_name: str) -> float:
    if z_value is None or (isinstance(z_value, float)
                            and z_value != z_value):
        return 50.0
    z = float(z_value)
    if stat_name in NEGATIVE_STATS:
        z = -z
    return z_to_grade(z)


def compute_rb_gas(df: pd.DataFrame,
                     spec: PositionGradeSpec = RB_SPEC,
                     min_games_full_grade: int = 10,
                     shrinkage_tau: float = 4.0,
                     ) -> pd.DataFrame:
    """Add bundle grades + composite GAS Score for each RB-season row.
    Same Path A treatment as QB: shrinkage toward league mean for thin
    samples, transparent per-bundle subscores, no multi-year smoothing
    (so we don't artificially inflate YoY).
    """
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

"""OL GAS Score — bundle-weighted, season-level OL grade.

IMPORTANT CAVEATS — public-data limits.
1. PASS BLOCKING IS TEAM-LEVEL. nflverse / PFR public data does not
   attribute pressures/sacks to specific blockers. We use team_sack_rate
   and team_pressure_rate as the pass-block signal — meaning every
   starter on a team gets the same pass-block grade. This is a real
   floor on per-player precision; PFF would be needed to break it.
2. RUN BLOCKING IS POSITION-GROUP-LEVEL. pos_run_epa filters runs by
   target gap (LT/LG, C, RG/RT). More granular than team but still
   shares signal across the side of the line.
3. NO SOS ADJUSTMENT YET. A team running against soft fronts looks
   better. Future v1.1 could LOO-adjust opponent run-defense.

These limits matter for "Sewell vs Ragnow vs Decker" — within-Lions
ranking is mostly run-block + penalties + snaps. Across-team
ranking is more reliable.

Bundles
-------
  Run blocking (35%)
    pos_run_epa_z         0.40
    pos_run_success_z     0.35
    pos_run_explosive_z   0.25

  Pass blocking (35%)  [TEAM-LEVEL — flagged in UI]
    team_sack_rate_z      0.50  (already direction-corrected at pull)
    team_pressure_rate_z  0.50

  Discipline (15%)
    penalty_rate_z          0.50  (already direction-corrected)
    penalty_epa_per_game_z  0.50

  Durability / role (15%)
    snap_share_z  1.00
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


# IMPORTANT: the z-cols in league_ol_all_seasons.parquet are ALREADY
# direction-corrected. team_sack_rate_z high → LOW sack rate (good
# protection); penalty_rate_z high → LOW penalty rate (good discipline).
# So we DON'T flip signs again here — that would invert good and bad.
NEGATIVE_STATS: set[str] = set()


RUN_BLOCKING = BundleSpec(
    name="Run blocking",
    stats={
        # Gap-level (LT/LG, C, RG/RT) — preserves within-team
        # granularity. Confounded by RB talent but separates left
        # side / center / right side of the line.
        "pos_run_epa_z":       0.20,
        "pos_run_success_z":   0.15,
        "pos_run_explosive_z": 0.10,
        # Team-level + RB-talent-adjusted (ryoe-controlled,
        # build_team_run_block_adjusted.py). Cleaner causal signal
        # but same value across all team OL — pairs with gap-level
        # to give both granularity AND honest RB-context. (Option B
        # from the user discussion.)
        "adj_team_run_epa_z":       0.25,
        "adj_team_run_success_z":   0.15,
        "adj_team_run_explosive_z": 0.15,
    },
)

PASS_BLOCKING = BundleSpec(
    name="Pass blocking",
    stats={
        # SOS-adjusted via per-team leave-one-out (build_team_pass_block_
        # adjusted.py). Each team's pressure / sack rate is netted of
        # the opponent's typical pressure-generation, so a team that
        # faced soft rushes doesn't get free credit and a team that
        # faced elite rushes (Eagles seeing the NFC's best, etc.)
        # doesn't get over-penalized.
        "adj_team_sack_rate_z":     0.50,
        "adj_team_pressure_rate_z": 0.50,
    },
)

DISCIPLINE = BundleSpec(
    name="Discipline",
    stats={
        "penalty_rate_z":         0.50,
        "penalty_epa_per_game_z": 0.50,
    },
)

DURABILITY = BundleSpec(
    name="Durability / role",
    stats={
        "snap_share_z": 1.00,
    },
)


OL_SPEC = PositionGradeSpec(
    position="OL",
    name_for_grade="GAS Score",
    bundles={
        "run_blocking":  RUN_BLOCKING,
        "pass_blocking": PASS_BLOCKING,
        "discipline":    DISCIPLINE,
        "durability":    DURABILITY,
    },
    # Discipline trimmed 15 → 5: it's TEAM-level (every blocker on a
    # team gets the same penalty grade so it doesn't differentiate
    # players), and penalty rate is one of the noisiest YoY OL stats.
    # Keep it as a tiebreaker, fund the cut to the discriminating
    # bundles (run + pass blocking).
    bundle_weights={
        "run_blocking":  0.40,
        "pass_blocking": 0.40,
        "discipline":    0.05,
        "durability":    0.15,
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


def compute_ol_gas(df: pd.DataFrame,
                     spec: PositionGradeSpec = OL_SPEC,
                     min_snaps_full_grade: int = 300,
                     shrinkage_tau: float = 4.0,
                     ) -> pd.DataFrame:
    out = df.copy()

    # Use games_played as the n for shrinkage (snaps work too but
    # games is more comparable across positions and to other GAS specs).
    games_col = "games_played" if "games_played" in out.columns else "games"

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

    # Confidence based on snap count rather than games (OL play
    # all 70 snaps when in the lineup, so off_snaps is the right proxy)
    snaps_col = "off_snaps" if "off_snaps" in out.columns else None
    if snaps_col is not None:
        def _conf(n):
            n = int(n) if pd.notna(n) else 0
            if n >= 800:
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

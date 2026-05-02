"""DE + DT GAS Score — shared lib, position-specific weights.

Both positions use identical z-cols from PFR-derived stats. The
spec differences:

  DE_SPEC: pass rush is primary (50%), run defense secondary (30%)
  DT_SPEC: run defense primary (40%), pass rush secondary (35%)

CAVEATS:
1. NO PFF PASS-RUSH WIN RATE — we're using sacks/pressures/hurries
   from PFR, which are play-result stats. Doesn't separate "won
   the rep but blocked at the right time" from "didn't win." PFF
   pass-rush win rate would be the gold standard.
2. NO RUN-FIT / GAP DISCIPLINE charting — we use TFL and tackle
   rate as proxies but can't measure "did this DT eat the double
   team" or "did this DE set the edge."
3. SCHEME ABSTRACTION — 4-3 DE vs 3-4 DE play different roles. The
   master file groups them together.
4. NO SOS adjustment — opposing OL quality not netted out. Future
   v1.1 could adjust by opp OL GAS (we now have it).
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


# ── Bundle definitions (shared, weights vary by position) ────────

PASS_RUSH = BundleSpec(
    name="Pass rush",
    stats={
        "sacks_per_game_z":         0.30,
        "pressures_per_game_z":     0.20,
        "qb_hits_per_game_z":       0.20,
        "hurries_per_game_z":       0.15,
        "pressure_rate_z":          0.15,   # rate vs raw count
    },
)

RUN_DEFENSE = BundleSpec(
    name="Run defense",
    stats={
        "tfl_per_game_z":     0.40,
        "tackles_per_snap_z": 0.30,
        "solo_tackle_rate_z": 0.30,
    },
)

DISRUPTION = BundleSpec(
    name="Disruption",
    stats={
        "qb_knockdowns_per_game_z": 0.40,
        "forced_fumbles_per_game_z": 0.30,
        "missed_tackle_pct_z":      0.30,
    },
)

BALL_PRODUCTION = BundleSpec(
    name="Ball production",
    stats={
        "passes_defended_per_game_z": 0.55,
        "interceptions_per_game_z":   0.45,
    },
)


DE_SPEC = PositionGradeSpec(
    position="DE",
    name_for_grade="GAS Score",
    bundles={
        "pass_rush":       PASS_RUSH,
        "run_defense":     RUN_DEFENSE,
        "disruption":      DISRUPTION,
        "ball_production": BALL_PRODUCTION,
    },
    bundle_weights={
        "pass_rush":       0.50,   # edge primary job
        "run_defense":     0.30,
        "disruption":      0.15,
        "ball_production": 0.05,
    },
)

DT_SPEC = PositionGradeSpec(
    position="DT",
    name_for_grade="GAS Score",
    bundles={
        "pass_rush":       PASS_RUSH,
        "run_defense":     RUN_DEFENSE,
        "disruption":      DISRUPTION,
        "ball_production": BALL_PRODUCTION,
    },
    bundle_weights={
        "pass_rush":       0.35,   # secondary interior priority
        "run_defense":     0.40,   # primary interior job
        "disruption":      0.15,
        "ball_production": 0.10,
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


def compute_dl_gas(df: pd.DataFrame,
                     spec: PositionGradeSpec,
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
            if n >= 600:
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

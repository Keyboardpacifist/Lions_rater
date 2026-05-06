"""OC GAS Score — bundle-weighted composite grade for offensive coordinators.

Spec — four bundles:

  Efficiency (45%) — core production
    Roster-adjusted z-scores where available; raw z fallback otherwise.
    The OC's value-add over what the roster should produce.
      epa_per_play_adj_z       0.30
      pass_epa_per_play_adj_z  0.25
      rush_epa_per_play_adj_z  0.15
      success_rate_adj_z       0.30

  Explosiveness (15%) — big plays
      explosive_pass_rate_z 0.55
      explosive_rush_rate_z 0.45

  Situational (20%) — when it matters
      third_down_rate_z   0.40
      red_zone_td_rate_z  0.40
      win_pct_z           0.20  (modest; team-confounded)

  Clutch (20%) — fulcrum performance + elevation
      clutch_EPA_z         0.30  (raw clutch EPA, wp_volatility def)
      clutch_EPA_elev_z    0.40  (does this OC step up vs own baseline?)
      clutch_succ_z        0.10
      clutch_succ_elev_z   0.20
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


EFFICIENCY = BundleSpec(
    name="Efficiency",
    stats={
        # Raw z-scores. Roster adjustment is too punitive with
        # whole-team team_rating as the proxy (it already bakes in
        # offensive output). Roster-adjusted view is the optional
        # rater toggle, not the GAS default.
        "epa_per_play_z":      0.30,
        "pass_epa_per_play_z": 0.25,
        "rush_epa_per_play_z": 0.15,
        "success_rate_z":      0.30,
    },
)

EXPLOSIVENESS = BundleSpec(
    name="Explosiveness",
    stats={
        "explosive_pass_rate_z": 0.55,
        "explosive_rush_rate_z": 0.45,
    },
)

SITUATIONAL = BundleSpec(
    name="Situational",
    stats={
        "third_down_rate_z":  0.40,
        "red_zone_td_rate_z": 0.40,
        "win_pct_z":          0.20,
    },
)

CLUTCH = BundleSpec(
    name="Clutch",
    stats={
        "clutch_EPA_z":       0.30,
        "clutch_EPA_elev_z":  0.40,
        "clutch_succ_z":      0.10,
        "clutch_succ_elev_z": 0.20,
    },
)


OC_SPEC = PositionGradeSpec(
    position="OC",
    name_for_grade="OC GAS Score",
    bundles={
        "efficiency":    EFFICIENCY,
        "explosiveness": EXPLOSIVENESS,
        "situational":   SITUATIONAL,
        "clutch":        CLUTCH,
    },
    bundle_weights={
        "efficiency":    0.45,
        "explosiveness": 0.15,
        "situational":   0.20,
        "clutch":        0.20,
    },
)


def _stat_grade(z_value: float, stat_name: str) -> float:
    if z_value is None or (isinstance(z_value, float) and z_value != z_value):
        return 50.0
    z = float(z_value)
    if stat_name in NEGATIVE_STATS:
        z = -z
    return z_to_grade(z)


def compute_oc_gas(df: pd.DataFrame,
                   spec: PositionGradeSpec = OC_SPEC,
                   shrinkage_tau: float = 2.0,
                   ) -> pd.DataFrame:
    """Add bundle grades + composite GAS Score for each OC.

    Sample-size proxy is the `seasons` column (years of OC tenure) since
    these are career-aggregated rows.
    """
    out = df.copy()

    all_stats = set()
    for bundle in spec.bundles.values():
        all_stats.update(bundle.stats.keys())

    # Fall back from *_adj_z to *_z when adj is missing (and the
    # raw column does exist). Lets OCs without roster-proxy coverage
    # still grade against the raw league.
    for stat in list(all_stats):
        if stat.endswith("_adj_z") and stat not in out.columns:
            raw = stat.replace("_adj_z", "_z")
            if raw in out.columns:
                out[stat] = out[raw]
        if stat not in out.columns:
            out[stat] = float("nan")

    # Per-row: where adj is NaN but raw is not, fall back per-cell.
    for stat in all_stats:
        if stat.endswith("_adj_z"):
            raw = stat.replace("_adj_z", "_z")
            if raw in out.columns:
                fill = out[stat].isna() & out[raw].notna()
                out.loc[fill, stat] = out.loc[fill, raw]

    for stat in all_stats:
        col = f"_{stat}_grade"
        def _grade_one(row, _stat=stat):
            z = row.get(_stat)
            n = row.get("seasons", 1)
            if n is None or (isinstance(n, float) and n != n):
                n = 1
            z_shrunk = shrunk_z(z, int(n), prior_z=0.0, tau=shrinkage_tau)
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

    if "seasons" in out.columns:
        def _conf(n):
            n = int(n) if pd.notna(n) else 0
            if n >= 4:
                return "HIGH"
            if n >= 2:
                return "MEDIUM"
            return "LOW"
        out["gas_confidence"] = out["seasons"].apply(_conf)
    else:
        out["gas_confidence"] = "MEDIUM"

    out = out.drop(columns=[c for c in out.columns
                              if c.startswith("_") and c.endswith("_grade")])
    return out

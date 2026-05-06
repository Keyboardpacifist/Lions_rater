"""Aggregate per-team-season scheme signals into per-OC career profiles.

Inputs
------
- data/scheme/curation/oc_team_seasons.csv       (Tier 3 / Brett curation)
- data/scheme/curation/coaching_tree.csv         (Tier 3 / Brett curation)
- data/scheme/curation/oc_signature_concepts.csv (Tier 3 / Brett curation)
- data/scheme/team_scheme_fingerprint.parquet    (Tier 1 / situational)
- data/scheme/team_passing_fingerprint.parquet   (Tier 1 / route + form)
- data/scheme/team_philosophy_fit.parquet        (Tier 1 / archetype fit)

Outputs
-------
- data/scheme/oc_career_profile.parquet
    Long-format situational fingerprint per OC, weighted-averaged
    across their team-seasons. Schema:
        oc_name, dimension, category, n, value, league_value,
        value_delta, value_z_avg, n_seasons

- data/scheme/oc_career_philosophy.parquet
    Per-OC averaged philosophy fit. Schema:
        oc_name, philosophy, fit_avg, fit_z_avg, league_z_avg,
        n_seasons

Use
---
This is the canonical data layer the rebuilt OC page reads. It joins
the data-derivable signals (Tier 1) to OC names via Brett's curated
team-season mapping (Tier 3). The page can lay these aggregates next
to the curated school / mentor / signature-concepts strings.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CURATION = REPO / "data" / "scheme" / "curation"
SCHEME = REPO / "data" / "scheme"

OC_TEAMS = CURATION / "oc_team_seasons.csv"
FINGERPRINT = SCHEME / "team_scheme_fingerprint.parquet"
PHIL_FIT = SCHEME / "team_philosophy_fit.parquet"

OUT_PROFILE = SCHEME / "oc_career_profile.parquet"
OUT_PHIL = SCHEME / "oc_career_philosophy.parquet"


def _weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    w = weights.fillna(0).astype(float)
    if w.sum() == 0:
        return float(values.mean()) if len(values) else np.nan
    return float((values.astype(float) * w).sum() / w.sum())


def main() -> None:
    print("→ loading inputs...")
    oc_teams = pd.read_csv(OC_TEAMS)
    fp = pd.read_parquet(FINGERPRINT)
    pf = pd.read_parquet(PHIL_FIT)
    print(f"  oc_team_seasons: {len(oc_teams):,} rows")
    print(f"  team_scheme_fingerprint: {len(fp):,} rows")
    print(f"  team_philosophy_fit: {len(pf):,} rows")

    oc_teams["season"] = oc_teams["season"].astype(int)

    # ------------------------------------------------------------------
    # Career situational profile
    # ------------------------------------------------------------------
    print("→ building career situational profile...")
    joined = fp.merge(
        oc_teams[["oc_name", "team", "season"]],
        on=["team", "season"], how="inner",
    )
    print(f"  joined situational rows: {len(joined):,}")

    rows = []
    for (oc, dim, cat), g in joined.groupby(
            ["oc_name", "dimension", "category"]):
        rows.append({
            "oc_name": oc,
            "dimension": dim,
            "category": cat,
            "n": int(g["n"].sum()),
            "value": _weighted_avg(g["value"], g["n"]),
            "league_value": _weighted_avg(g["league_value"], g["n"]),
            "value_delta": _weighted_avg(g["value_delta"], g["n"]),
            "value_z_avg": _weighted_avg(g["value_z"], g["n"]),
            "n_seasons": g["season"].nunique(),
        })
    profile = pd.DataFrame(rows).sort_values(
        ["oc_name", "dimension", "category"])
    profile.to_parquet(OUT_PROFILE, index=False)
    print(f"  ✓ wrote {OUT_PROFILE.relative_to(REPO)}  rows={len(profile):,}")

    # ------------------------------------------------------------------
    # Career philosophy fit
    # ------------------------------------------------------------------
    print("→ building career philosophy fit...")
    pf_join = pf.merge(
        oc_teams[["oc_name", "team", "season"]],
        on=["team", "season"], how="inner",
    )
    phil_rows = []
    for (oc, phi), g in pf_join.groupby(["oc_name", "philosophy"]):
        phil_rows.append({
            "oc_name": oc,
            "philosophy": phi,
            "fit_avg": float(g["fit"].mean()),
            "fit_z_avg": float(g["fit_z"].mean()),
            "league_z_avg": float(g["league_z"].mean()),
            "n_seasons": int(g["season"].nunique()),
        })
    phil = pd.DataFrame(phil_rows)

    # Rank within each OC
    phil["rank"] = (
        phil.groupby("oc_name")["fit_z_avg"]
            .rank(method="min", ascending=False)
            .astype(int)
    )
    phil["top_match"] = phil["rank"] == 1
    phil = phil.sort_values(["oc_name", "rank"])
    phil.to_parquet(OUT_PHIL, index=False)
    print(f"  ✓ wrote {OUT_PHIL.relative_to(REPO)}  rows={len(phil):,}")
    print()

    # ------------------------------------------------------------------
    # Spot checks
    # ------------------------------------------------------------------
    print("=== Ben Johnson — career situational profile (top z signals) ===")
    bj = profile[profile["oc_name"] == "Ben Johnson"].copy()
    bj["abs_z"] = bj["value_z_avg"].abs()
    print(bj.nlargest(10, "abs_z")[
        ["dimension", "category", "n", "value", "league_value",
         "value_z_avg", "n_seasons"]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Ben Johnson — career philosophy fit ===")
    bjp = phil[phil["oc_name"] == "Ben Johnson"].sort_values("rank")
    print(bjp[["philosophy", "fit_avg", "fit_z_avg", "league_z_avg",
               "rank"]].to_string(index=False,
                                  float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Top OC by philosophy_top_match ===")
    tops = (phil[phil["top_match"]]
            .sort_values(["philosophy", "fit_z_avg"], ascending=[True, False]))
    print(tops[["philosophy", "oc_name", "fit_z_avg",
                "league_z_avg", "n_seasons"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

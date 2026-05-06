"""Score each team-season against the 5 coaching philosophy archetypes.

Inputs
------
- data/scheme/team_route_harmonized.parquet  — actual 12-route shares
- data/scheme/coaching_philosophies.json     — expected fingerprints

Output
------
- data/scheme/team_philosophy_fit.parquet
    team, season, philosophy, jsd, fit, rank, top_match

Method
------
- LEARN data-driven centroids: each philosophy's expected vector is the
  average observed route share across its known practitioner team-seasons
  (a seed list embedded below). Hand-seeded shares in the JSON were too
  coarse to discriminate; learning from real data lets us see which
  team's distribution actually looks like a Shanahan-tree distribution
  versus Reid versus Belichick.
- Compute Jensen-Shannon divergence (JSD, base 2) between the team's
  observed route share vector and each learned centroid.
  JSD ∈ [0, 1] — 0 means identical distributions, 1 means orthogonal.
- fit = 1 - JSD  → 1 means perfect philosophy match.
- For each (team, season), rank philosophies and flag the top.

Bridge
------
Combined with `data/scheme/curation/coaching_tree.csv` (which school
the OC came from), this tells you whether the offense actually plays
like its claimed lineage — or has drifted. Drift is the story.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ROUTES_PATH = REPO / "data" / "scheme" / "team_route_harmonized.parquet"
PHILO_PATH = REPO / "data" / "scheme" / "coaching_philosophies.json"
OUT_PATH = REPO / "data" / "scheme" / "team_philosophy_fit.parquet"

# 12-route harmonized taxonomy (alphabetical order — must match what's in
# both the parquet and the philosophies JSON)
ROUTE_KEYS = ["ANGLE", "CORNER", "CROSS", "FLAT", "GO", "HITCH",
              "IN", "OUT", "POST", "SCREEN", "SLANT", "WHEEL"]

# Seed practitioners — the team-seasons each philosophy's centroid is
# learned from. Picked to be unambiguous lineage examples (ignoring HC
# changes that re-defined identity mid-window). Refine over time.
SEED_PRACTITIONERS = {
    "WCO": [
        # Shanahan tree, well-established WCO years
        ("SF", 2019), ("SF", 2021), ("SF", 2022), ("SF", 2023),
        ("LA", 2018), ("LA", 2021), ("LA", 2023),
        ("GB",  2022), ("GB", 2023),
        ("MIA", 2022), ("MIA", 2023),
        ("DET", 2023),
    ],
    "Air Coryell": [
        # Vertical-leaning legacy practitioners
        ("LAC", 2018), ("LAC", 2019),  # Anthony Lynn / Ken Whisenhunt vertical
        ("BAL", 2019), ("BAL", 2020),  # Roman / Lamar vertical PA
        ("BUF", 2020), ("BUF", 2021),  # Daboll / Allen vertical era
        ("ARI", 2017),                  # Bruce Arians legacy
    ],
    "Erhardt-Perkins": [
        # Belichick lineage / EP concept-based teams
        ("NE",  2017), ("NE", 2018), ("NE", 2019),
        ("NYG", 2022), ("NYG", 2023),  # Daboll
        ("LV",  2022),                  # McDaniels
        ("HOU", 2020),                  # O'Brien
    ],
    "Spread/RPO": [
        # Reid / Sirianni / Pederson hybrid spread
        ("KC",  2018), ("KC", 2019), ("KC", 2020), ("KC", 2022),
        ("PHI", 2017), ("PHI", 2022), ("PHI", 2023),
    ],
    "Power Run / Vertical": [
        # Roman / heavy run + PA shots
        ("BAL", 2019), ("BAL", 2020),
        ("BUF", 2018), ("BUF", 2019),
        ("TEN", 2019), ("TEN", 2020),  # Henry-era run-first
    ],
}


def jensen_shannon_div(p: np.ndarray, q: np.ndarray) -> float:
    """JSD with log base 2 — output in [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def main() -> None:
    print("→ loading route shares + philosophies...")
    routes = pd.read_parquet(ROUTES_PATH)
    routes["season"] = routes["season"].astype(int)

    with open(PHILO_PATH, "r") as f:
        philos = json.load(f)
    philos = {k: v for k, v in philos.items() if not k.startswith("_")}
    print(f"  philosophies: {list(philos.keys())}")

    # Pivot to wide so we have one row per (team, season) with 12 route columns.
    wide = (
        routes.pivot_table(
            index=["team", "season"], columns="route",
            values="share", fill_value=0.0,
        )
        .reset_index()
    )
    # Ensure all 12 columns exist
    for r in ROUTE_KEYS:
        if r not in wide.columns:
            wide[r] = 0.0
    wide = wide[["team", "season"] + ROUTE_KEYS]
    print(f"  team-seasons: {len(wide):,}")

    # Build LEARNED centroid vectors from seed practitioners.
    exp_vecs = {}
    centroid_meta = {}
    for phi_name, seeds in SEED_PRACTITIONERS.items():
        seed_df = wide.merge(
            pd.DataFrame(seeds, columns=["team", "season"]),
            on=["team", "season"], how="inner",
        )
        if seed_df.empty:
            print(f"  ⚠ {phi_name}: no seed practitioners matched — "
                  f"falling back to JSON expected_route_shares")
            shares = philos.get(phi_name, {}).get("expected_route_shares", {})
            vec = np.array([shares.get(r, 0.0) for r in ROUTE_KEYS])
        else:
            vec = seed_df[ROUTE_KEYS].mean(axis=0).values
            centroid_meta[phi_name] = len(seed_df)
        if vec.sum() > 0:
            vec = vec / vec.sum()
        exp_vecs[phi_name] = vec
    print(f"  centroid sample sizes: {centroid_meta}")

    # Compute fit for every (team, season, philosophy)
    rows = []
    for _, r in wide.iterrows():
        observed = r[ROUTE_KEYS].values.astype(float)
        if observed.sum() <= 0:
            continue
        observed = observed / observed.sum()
        for phi_name, exp in exp_vecs.items():
            jsd = jensen_shannon_div(observed, exp)
            rows.append({
                "team": r["team"],
                "season": r["season"],
                "philosophy": phi_name,
                "jsd": jsd,
                "fit": 1.0 - jsd,
            })

    out = pd.DataFrame(rows)

    # Rank within each (team, season) — 1 is the best fit
    out["rank"] = (
        out.groupby(["team", "season"])["fit"]
           .rank(method="min", ascending=False)
           .astype(int)
    )
    out["top_match"] = out["rank"] == 1

    # Within-team-season z-score of fit across philosophies. Absolute fits
    # are saturated near 1.0 because the 12-route taxonomy is too coarse
    # to discriminate cleanly — but the *relative* pull is meaningful.
    # fit_z > +1: this philosophy is unusually pulling on this team-season.
    out["fit_z"] = (
        out.groupby(["team", "season"])["fit"]
           .transform(lambda s: (s - s.mean())
                                / (s.std(ddof=0) or np.nan))
           .fillna(0.0)
    )

    # Within-philosophy z-score of fit across the league for that season —
    # tells you "this team looks unusually like a WCO team this year".
    out["league_z"] = (
        out.groupby(["season", "philosophy"])["fit"]
           .transform(lambda s: (s - s.mean())
                                / (s.std(ddof=0) or np.nan))
           .fillna(0.0)
    )

    out = out.sort_values(["team", "season", "rank"]).reset_index(drop=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"  ✓ wrote {OUT_PATH.relative_to(REPO)}  rows={len(out):,}")
    print()

    # Spot checks
    print("=== DET 2024 — philosophy fit ===")
    det = out[(out["team"] == "DET") & (out["season"] == 2024)]
    print(det[["philosophy", "fit", "fit_z", "league_z", "rank"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Top 5 most-WCO-pulled team-seasons (league_z) ===")
    wcoz = (out[out["philosophy"] == "WCO"]
            .nlargest(8, "league_z"))
    print(wcoz[["team", "season", "fit", "league_z"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Most WCO-fitting team-seasons (across all years) ===")
    wco = out[(out["philosophy"] == "WCO") & (out["fit"] >= 0.85)]
    print(wco.nlargest(10, "fit")[["team", "season", "fit", "rank"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Top philosophy by team-season, 2024 ===")
    tops = out[(out["season"] == 2024) & (out["top_match"])]
    print(tops[["team", "philosophy", "fit"]]
          .sort_values("team")
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

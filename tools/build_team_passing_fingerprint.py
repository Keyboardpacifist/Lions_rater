"""Per-team passing-scheme fingerprint.

Output: data/scheme/team_passing_fingerprint.parquet

For every (team, season), describe HOW the offense throws the ball:
which routes, which depths, which locations, which personnel groupings.
The result is a multi-dimensional "passing DNA" that a player career
profile can be matched against.

Phase 1 of the cross-position Scheme Fit Engine. See
project_route_opportunity_feature.md memory for the full strategic
framing — this is the platform's biggest moat.

Output schema (long format)
---------------------------
    team, season, dimension, category, count, share,
    league_share, share_delta, share_z

  dimension is one of: route, depth, location, personnel
  category is the value within that dimension
    (e.g., dimension="route", category="IN/DIG")
  share = team's share of pass volume on that category
  league_share = the leaguewide share for the same season
  share_delta = team_share - league_share
  share_z = z-score of share within that (season, dimension)

So "DAL 2025, dimension=route, category=IN/DIG, share=0.10,
league_share=0.07, share_z=+1.2" means DAL ran 10% IN/DIG vs
the league's 7% — about 1.2σ above average. Dig-heavy team.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DROPBACKS = REPO / "data" / "qb_dropbacks.parquet"
OUT_DIR = REPO / "data" / "scheme"
OUT = OUT_DIR / "team_passing_fingerprint.parquet"


def _depth_bucket(air_yards: float) -> str:
    if pd.isna(air_yards):
        return "unknown"
    if air_yards < 0:
        return "behind_los"
    if air_yards < 10:
        return "short"
    if air_yards < 20:
        return "intermediate"
    return "deep"


def _personnel_simple(personnel: str) -> str:
    """Reduce 'offense_personnel' to a coarse bucket: 11 / 12 / 13 / OTHER.
    nflverse personnel strings look like '1 RB, 1 TE, 3 WR'."""
    if pd.isna(personnel):
        return "unknown"
    p = str(personnel)
    if "1 RB, 1 TE, 3 WR" in p or "1RB 1TE 3WR" in p:
        return "11"
    if "1 RB, 2 TE, 2 WR" in p or "1RB 2TE 2WR" in p:
        return "12"
    if "1 RB, 3 TE, 1 WR" in p or "1RB 3TE 1WR" in p:
        return "13"
    if "2 RB, 1 TE, 2 WR" in p or "2RB 1TE 2WR" in p:
        return "21"
    if "2 RB, 2 TE, 1 WR" in p or "2RB 2TE 1WR" in p:
        return "22"
    return "other"


def _build_dim(df: pd.DataFrame, col: str, dim_name: str
                 ) -> pd.DataFrame:
    """Compute team & league shares for one dimension column."""
    sub = df.dropna(subset=[col]).copy()
    sub = sub[sub[col] != ""]

    team_totals = (
        sub.groupby(["posteam", "season"]).size().rename("team_total")
    )
    cell_counts = (
        sub.groupby(["posteam", "season", col]).size()
           .rename("count").reset_index()
    )
    cell_counts = cell_counts.merge(team_totals, on=["posteam", "season"])
    cell_counts["share"] = cell_counts["count"] / cell_counts["team_total"]

    # League share (per season, per category)
    season_totals = sub.groupby("season").size().rename("season_total")
    season_cat = (
        sub.groupby(["season", col]).size()
           .rename("league_count").reset_index()
    )
    season_cat = season_cat.merge(season_totals, on="season")
    season_cat["league_share"] = (
        season_cat["league_count"] / season_cat["season_total"]
    )
    cell_counts = cell_counts.merge(
        season_cat[["season", col, "league_share"]],
        on=["season", col],
    )
    cell_counts["share_delta"] = (
        cell_counts["share"] - cell_counts["league_share"]
    )

    # Z-score the share within each (season, category)
    cell_counts["share_z"] = (
        cell_counts.groupby(["season", col])["share"]
                   .transform(lambda x: (x - x.mean())
                                          / (x.std(ddof=0) or np.nan))
                   .fillna(0)
    )

    cell_counts = cell_counts.rename(
        columns={"posteam": "team", col: "category"})
    cell_counts["dimension"] = dim_name
    return cell_counts[[
        "team", "season", "dimension", "category", "count",
        "share", "league_share", "share_delta", "share_z"
    ]]


def main() -> None:
    print("→ loading qb_dropbacks...")
    db = pd.read_parquet(DROPBACKS)
    db = db.dropna(subset=["posteam", "season"])
    db["season"] = db["season"].astype(int)
    print(f"  total dropbacks: {len(db):,}")

    # Build derived columns
    db["depth"] = db["air_yards"].apply(_depth_bucket)
    db["personnel"] = db["offense_personnel"].apply(_personnel_simple)

    print("→ building dimensions...")
    parts = []
    for col, dim in [
        ("route", "route"),
        ("depth", "depth"),
        ("pass_location", "location"),
        ("personnel", "personnel"),
        ("offense_formation", "formation"),
    ]:
        if col not in db.columns:
            continue
        parts.append(_build_dim(db, col, dim))
        print(f"  {dim}: {len(parts[-1]):,} cells")

    fingerprint = pd.concat(parts, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()

    # Spot check: DAL 2025 route DNA
    dal = fingerprint[
        (fingerprint["team"] == "DAL")
        & (fingerprint["season"] == 2025)
        & (fingerprint["dimension"] == "route")
    ].sort_values("share", ascending=False)
    print("=== DAL 2025 route DNA (top 10) ===")
    cols = ["category", "count", "share", "league_share",
            "share_delta", "share_z"]
    print(dal[cols].head(10).to_string(index=False))
    print()

    # Most "dig-heavy" team in 2024
    digs = fingerprint[
        (fingerprint["dimension"] == "route")
        & (fingerprint["category"] == "IN/DIG")
        & (fingerprint["season"] == 2024)
    ].nlargest(5, "share")
    print("=== Top 5 dig-heaviest teams 2024 ===")
    print(digs[["team"] + cols].to_string(index=False))


if __name__ == "__main__":
    main()

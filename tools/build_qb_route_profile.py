"""Per-(QB, season) route throwing profile + per-team primary-QB profile.

Outputs:
  data/scheme/qb_route_profile.parquet — per (passer, season) route distribution
  data/scheme/team_qb_profile.parquet  — per (team, season) using the team's
                                          primary QB

What this enables: when a team loses a deep-threat WR, we can ask
*does the QB even throw deep?* If the new QB's career GO-route share
is 4% (vs league 8%), the vacated GO targets won't reappear — they'll
redirect to the routes the QB actually prefers. This is the QB-tendency
cross-reference for the Usage Autopsy.

Schema (qb_route_profile.parquet)
---------------------------------
    passer_player_id, passer_player_name, season, route, throws,
    share, league_share, share_delta, share_z

Schema (team_qb_profile.parquet)
--------------------------------
    team, season, passer_player_id, passer_player_name,
    primary_qb_dropbacks, route, share, league_share, share_z

  "Primary QB" = passer with the most dropbacks for that team-season.
  We surface their throwing profile so we can compare to the team's
  vacated-route demand and project realistic redistribution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DROPBACKS = REPO / "data" / "qb_dropbacks.parquet"
OUT_DIR = REPO / "data" / "scheme"
QB_OUT = OUT_DIR / "qb_route_profile.parquet"
TEAM_QB_OUT = OUT_DIR / "team_qb_profile.parquet"

MIN_QB_THROWS = 80   # career-min for a QB to have a stable route profile


def main() -> None:
    print("→ loading qb_dropbacks...")
    db = pd.read_parquet(DROPBACKS)
    db = db.dropna(subset=["passer_player_id", "season", "route",
                              "posteam"])
    db = db[db["route"] != ""].copy()
    db["season"] = db["season"].astype(int)
    print(f"  qualifying dropbacks w/ route: {len(db):,}")

    # ── PER-(QB, SEASON) ROUTE PROFILE ─────────────────────────────
    qb_throws = (
        db.groupby(["passer_player_id", "passer_player_name", "season"])
          .size().rename("season_throws").reset_index()
    )
    qb_route = (
        db.groupby(["passer_player_id", "passer_player_name",
                       "season", "route"], as_index=False)
          .size().rename(columns={"size": "throws"})
    )
    qb_route = qb_route.merge(qb_throws,
                                  on=["passer_player_id",
                                       "passer_player_name", "season"])
    qb_route["share"] = qb_route["throws"] / qb_route["season_throws"]

    # League share per (season, route) for context
    season_total = (
        db.groupby("season").size().rename("season_total").reset_index()
    )
    season_route = (
        db.groupby(["season", "route"]).size().rename(
            "league_throws").reset_index()
    )
    season_route = season_route.merge(season_total, on="season")
    season_route["league_share"] = (
        season_route["league_throws"] / season_route["season_total"]
    )

    qb_route = qb_route.merge(
        season_route[["season", "route", "league_share"]],
        on=["season", "route"],
    )
    qb_route["share_delta"] = (
        qb_route["share"] - qb_route["league_share"]
    )

    # Z within (season, route)
    qb_route["share_z"] = (
        qb_route.groupby(["season", "route"])["share"]
                .transform(lambda x: (x - x.mean())
                                       / (x.std(ddof=0) or np.nan))
                .fillna(0)
    )

    qb_route = qb_route.rename(
        columns={"season_throws": "qb_season_throws"})

    # Filter to QBs with meaningful sample
    qb_route = qb_route[
        qb_route["qb_season_throws"] >= MIN_QB_THROWS
    ]
    print(f"  per-(QB, season, route) rows: {len(qb_route):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qb_route.to_parquet(QB_OUT, index=False)
    print(f"  ✓ wrote {QB_OUT.relative_to(REPO)}")

    # ── PER-(TEAM, SEASON) PRIMARY-QB PROFILE ──────────────────────
    print("→ identifying primary QB per (team, season)...")
    team_qb_throws = (
        db.groupby(["posteam", "season", "passer_player_id"])
          .size().rename("throws").reset_index()
    )
    primary = (
        team_qb_throws.sort_values("throws", ascending=False)
                       .drop_duplicates(["posteam", "season"])
                       [["posteam", "season", "passer_player_id",
                          "throws"]]
    )
    primary = primary.rename(columns={
        "posteam": "team",
        "throws": "primary_qb_dropbacks",
    })

    # Attach the primary QB's per-(season, route) share, so each
    # (team, season, route) row tells us how much that team's QB
    # threw on each route.
    team_qb_route = primary.merge(
        qb_route[["passer_player_id", "passer_player_name",
                   "season", "route", "share", "league_share",
                   "share_z"]],
        on=["passer_player_id", "season"],
        how="left",
    )
    team_qb_route = team_qb_route[[
        "team", "season", "passer_player_id", "passer_player_name",
        "primary_qb_dropbacks", "route", "share", "league_share",
        "share_z",
    ]]

    team_qb_route.to_parquet(TEAM_QB_OUT, index=False)
    print(f"  ✓ wrote {TEAM_QB_OUT.relative_to(REPO)}")
    print()

    # Spot check — Lamar's profile
    print("=== Lamar Jackson 2025 throwing profile ===")
    lamar = qb_route[
        (qb_route["passer_player_name"].str.contains("L.Jackson",
                                                          na=False))
        & (qb_route["season"] == 2025)
    ].sort_values("share", ascending=False)
    if not lamar.empty:
        print(lamar[["route", "throws", "share", "league_share",
                       "share_z"]].head(10).to_string(index=False))
    else:
        print("  no rows")


if __name__ == "__main__":
    main()

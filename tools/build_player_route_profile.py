"""Per-player career route profile.

Output: data/scheme/player_route_profile.parquet

For every receiver, describe WHAT KIND of receiver he is — by the
distribution of routes he's run and how well he's performed on each.
The other half of the Scheme Fit Engine: this is what we match
against the team passing fingerprint to compute scheme-fit scores.

Schema (long format)
--------------------
    player_id, player_display_name, position, dimension, category,
    targets, share, catch_rate, yards_per_target, epa_per_target,
    league_share, share_delta, share_z

  dimension is one of: route, depth, location, personnel
  Each player has one row per (dimension, category) for the routes /
  depths / locations / personnel groupings he's actually been targeted on.

Usage examples
--------------
- Find dig merchants: filter to dimension=route, category=IN/DIG,
  sort by share desc — top results are players who get a high % of
  their targets on dig routes.
- Find deep specialists: filter to dimension=depth, category=deep,
  sort by share desc.
- Find slot guys: filter to dimension=location, category=middle,
  sort by share desc.

Career-aggregated for the MVP. Per-season splits could be a
follow-up if the user wants to track route-profile evolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DROPBACKS = REPO / "data" / "qb_dropbacks.parquet"
ROSTERS = REPO / "data" / "nfl_rosters.parquet"
WEEKLY = REPO / "data" / "nfl_player_stats_weekly.parquet"
OUT_DIR = REPO / "data" / "scheme"
OUT = OUT_DIR / "player_route_profile.parquet"

MIN_TARGETS = 30   # career minimum to grade a player


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
    if pd.isna(personnel):
        return "unknown"
    p = str(personnel)
    if "1 RB, 1 TE, 3 WR" in p:
        return "11"
    if "1 RB, 2 TE, 2 WR" in p:
        return "12"
    if "1 RB, 3 TE, 1 WR" in p:
        return "13"
    if "2 RB, 1 TE, 2 WR" in p:
        return "21"
    if "2 RB, 2 TE, 1 WR" in p:
        return "22"
    return "other"


def _player_dimension(df: pd.DataFrame, col: str, dim_name: str
                       ) -> pd.DataFrame:
    """Compute per-(player, category) targets, share, and efficiency
    for one dimension."""
    sub = df.dropna(subset=[col, "receiver_player_id"]).copy()
    sub = sub[sub[col] != ""]

    player_totals = (
        sub.groupby("receiver_player_id").size().rename("player_total")
    )

    grp = sub.groupby(["receiver_player_id", col])
    rows = grp.agg(
        targets=("epa", "size"),
        catches=("complete_pass", "sum"),
        yards=("passing_yards", "sum"),
        epa=("epa", "sum"),
    ).reset_index()

    rows = rows.merge(player_totals, on="receiver_player_id")
    rows["share"] = rows["targets"] / rows["player_total"]
    rows["catch_rate"] = rows["catches"] / rows["targets"]
    rows["yards_per_target"] = rows["yards"] / rows["targets"]
    rows["epa_per_target"] = rows["epa"] / rows["targets"]

    # League share (across all qualifying targets, by category)
    league_total = len(sub)
    league_cat = sub.groupby(col).size().rename(
        "league_count").reset_index()
    league_cat["league_share"] = league_cat["league_count"] / league_total
    rows = rows.merge(league_cat[[col, "league_share"]], on=col)
    rows["share_delta"] = rows["share"] - rows["league_share"]

    # Z-score share across all PLAYERS within this category (so high
    # z = "this player is a specialist on this route relative to peers")
    rows["share_z"] = (
        rows.groupby(col)["share"]
            .transform(lambda x: (x - x.mean())
                                   / (x.std(ddof=0) or np.nan))
            .fillna(0)
    )

    rows = rows.rename(columns={col: "category"})
    rows["dimension"] = dim_name
    return rows[[
        "receiver_player_id", "dimension", "category",
        "targets", "share", "catch_rate", "yards_per_target",
        "epa_per_target", "league_share", "share_delta", "share_z",
    ]]


def main() -> None:
    print("→ loading qb_dropbacks...")
    db = pd.read_parquet(DROPBACKS)
    db = db.dropna(subset=["receiver_player_id", "epa"])
    db["season"] = db["season"].astype(int)

    db["depth"] = db["air_yards"].apply(_depth_bucket)
    db["personnel"] = db["offense_personnel"].apply(_personnel_simple)

    # Filter to players who hit the min-target threshold across career
    targets_per_player = db.groupby("receiver_player_id").size()
    qualified = set(
        targets_per_player[targets_per_player >= MIN_TARGETS].index
    )
    db = db[db["receiver_player_id"].isin(qualified)]
    print(f"  qualifying receivers (≥{MIN_TARGETS} career targets): "
          f"{len(qualified):,}")

    print("→ computing per-dimension profiles...")
    parts = []
    for col, dim in [
        ("route", "route"),
        ("depth", "depth"),
        ("pass_location", "location"),
        ("personnel", "personnel"),
    ]:
        parts.append(_player_dimension(db, col, dim))
        print(f"  {dim}: {len(parts[-1]):,} player-cells")

    profiles = pd.concat(parts, ignore_index=True)

    # Attach player display name + position via nfl_player_stats_weekly
    print("→ attaching player names + position...")
    pw = pd.read_parquet(WEEKLY,
                            columns=["player_id", "player_display_name",
                                       "position"])
    pw = pw.dropna(subset=["player_id"]).drop_duplicates(
        subset=["player_id"])
    profiles = profiles.merge(
        pw.rename(columns={"player_id": "receiver_player_id"}),
        on="receiver_player_id", how="left",
    )
    profiles = profiles.rename(
        columns={"receiver_player_id": "player_id"})
    profiles = profiles[[
        "player_id", "player_display_name", "position",
        "dimension", "category", "targets", "share",
        "catch_rate", "yards_per_target", "epa_per_target",
        "league_share", "share_delta", "share_z",
    ]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()

    # Spot check: top dig-route specialists
    digs = profiles[
        (profiles["dimension"] == "route")
        & (profiles["category"] == "IN/DIG")
    ].nlargest(10, "share")
    print("=== Top 10 IN/DIG specialists (career share) ===")
    cols = ["player_display_name", "position", "targets", "share",
            "league_share", "share_delta", "share_z",
            "yards_per_target", "epa_per_target"]
    print(digs[cols].to_string(index=False))
    print()

    # CeeDee Lamb route profile
    lamb_profile = profiles[
        (profiles["player_display_name"] == "CeeDee Lamb")
        & (profiles["dimension"] == "route")
    ].sort_values("share", ascending=False)
    if len(lamb_profile):
        print("=== CeeDee Lamb career route profile ===")
        print(lamb_profile[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

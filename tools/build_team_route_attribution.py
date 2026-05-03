"""Per (team, season, route, receiver) target attribution.

Output: data/scheme/team_route_attribution.parquet

For every (team, season), break down each route's targets by which
receiver got them. This is the foundation of the vacated-demand
calculator: when a player leaves, we can compute exactly how many
targets-per-route depart with him.

Schema
------
  team, season, route, receiver_player_id, player_display_name,
  position, targets, catches, yards, epa,
  share_of_team_route   ← receiver's share of THIS team-route
  share_of_player_routes ← this team-route's share of player's career

A row of "DET 2024 IN/DIG · Sam LaPorta · 18 targets, 0.43 share_of_team_route"
means LaPorta got 18 dig targets in Detroit 2024, which was 43% of
DET's IN/DIG volume that season. If he leaves, those 43% are vacated.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DROPBACKS = REPO / "data" / "qb_dropbacks.parquet"
WEEKLY = REPO / "data" / "nfl_player_stats_weekly.parquet"
OUT_DIR = REPO / "data" / "scheme"
OUT = OUT_DIR / "team_route_attribution.parquet"


def main() -> None:
    print("→ loading qb_dropbacks...")
    db = pd.read_parquet(DROPBACKS)
    db = db.dropna(subset=["receiver_player_id", "route", "posteam",
                              "season", "epa"])
    db = db[db["route"] != ""].copy()
    db["season"] = db["season"].astype(int)
    print(f"  qualifying targets: {len(db):,}")

    # Per (team, season, route, receiver) totals
    print("→ aggregating per-(team, season, route, receiver)...")
    # qb_dropbacks doesn't have a touchdown column directly, but we
    # can derive: a passing TD on the route was a (complete_pass=1
    # AND yards == yards-to-endzone). Simpler — pull TDs from pbp
    # via play_id join. But for performance, easier to scan pbp
    # for touchdown directly.
    db_with_td = db.copy()
    if "touchdown" in db.columns:
        db_with_td["td"] = db["touchdown"].fillna(0)
    else:
        # Fallback: join pbp for touchdown info
        from pathlib import Path
        PBP = REPO / "data" / "game_pbp.parquet"
        pbp_td = pd.read_parquet(PBP, columns=[
            "game_id", "play_id", "touchdown",
        ]).rename(columns={"touchdown": "td"})
        db_with_td = db.merge(pbp_td, on=["game_id", "play_id"],
                                  how="left")
        db_with_td["td"] = db_with_td["td"].fillna(0)

    attr = db_with_td.groupby(
        ["posteam", "season", "route", "receiver_player_id"],
        as_index=False,
    ).agg(
        targets=("epa", "size"),
        catches=("complete_pass", "sum"),
        yards=("passing_yards", "sum"),
        tds=("td", "sum"),
        epa=("epa", "sum"),
    )
    print(f"  team-season-route-receiver rows: {len(attr):,}")

    # Team-route totals (denominator for "share of team route")
    team_route_totals = (
        attr.groupby(["posteam", "season", "route"])["targets"]
            .sum().rename("team_route_total").reset_index()
    )
    attr = attr.merge(team_route_totals,
                          on=["posteam", "season", "route"])
    attr["share_of_team_route"] = (
        attr["targets"] / attr["team_route_total"]
    )

    # Player-career totals (denominator for "share of player's career routes")
    player_totals = (
        attr.groupby("receiver_player_id")["targets"]
            .sum().rename("player_total").reset_index()
    )
    attr = attr.merge(player_totals, on="receiver_player_id")
    attr["share_of_player_routes"] = (
        attr["targets"] / attr["player_total"]
    )

    # Attach display name + position
    print("→ attaching player names + positions...")
    pw = pd.read_parquet(WEEKLY,
                            columns=["player_id", "player_display_name",
                                       "position"])
    pw = pw.dropna(subset=["player_id"]).drop_duplicates(
        subset=["player_id"])
    attr = attr.merge(
        pw.rename(columns={"player_id": "receiver_player_id"}),
        on="receiver_player_id", how="left",
    )

    out = attr.rename(columns={"posteam": "team"})[[
        "team", "season", "route", "receiver_player_id",
        "player_display_name", "position",
        "targets", "catches", "yards", "tds", "epa",
        "team_route_total", "share_of_team_route",
        "share_of_player_routes",
    ]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()

    # Spot check: DET 2024 IN/DIG
    det_dig = out[
        (out["team"] == "DET") & (out["season"] == 2024)
        & (out["route"] == "IN/DIG")
    ].sort_values("targets", ascending=False)
    print("=== DET 2024 IN/DIG attribution ===")
    print(det_dig[["player_display_name", "position", "targets",
                     "share_of_team_route", "yards",
                     "share_of_player_routes"]
                  ].to_string(index=False))


if __name__ == "__main__":
    main()

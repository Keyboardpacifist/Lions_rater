"""Per (team, season, week) weapons-availability index.

Output: data/team_weapons_availability.parquet

For each team-game, compute how much of the team's expected receiving
threat was actually on the field. Used by QB SOS to credit QBs who
perform without their top weapons.

Method
------
1. Identify each team's top-N target hogs that season (by target share
   from pbp). Includes WR/TE/RB — anyone who catches passes.
2. For each (team, season, week), check snap_counts to see which
   top-N players actually played and how much.
3. weapons_strength = Σ(target_share × snap_pct_played)
                    / Σ(target_share)
   ∈ [0, 1] where 1.0 = all top-N played 100%; 0.5 = half the
   expected receiving threat was on the field.

Edge cases
----------
- Snap counts only available 2013+. Defaults to 1.0 if missing.
- Target share is computed against ALL pass attempts (not just
  WR/TE-targeted), so checkdowns to RBs count.

Output schema
-------------
team, season, week, weapons_strength, weapons_full_count
  weapons_full_count = # of top-N players who played ≥50% offense
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
SNAPS = REPO / "data" / "nfl_snap_counts.parquet"
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
OUT = REPO / "data" / "team_weapons_availability.parquet"

TOP_N = 5   # top 5 by season target share


def _normalize_name(s: pd.Series) -> pd.Series:
    """Normalize names for fuzzy join: lowercase, strip punct/spaces."""
    return (s.fillna("").str.lower()
              .str.replace(r"[\.\s\-']", "", regex=True))


def main() -> None:
    print("→ loading inputs...")
    pbp = pd.read_parquet(PBP)
    snaps = pd.read_parquet(SNAPS)

    # Build player_id → player_display_name lookup from weekly stats
    # (covers WR/TE/RB — every player who's recorded any stat).
    pw = pd.read_parquet(PLAYER_STATS,
                            columns=["player_id", "player_display_name"])
    name_lookup = (pw.dropna(subset=["player_id",
                                          "player_display_name"])
                       .drop_duplicates(subset=["player_id"]))
    print(f"  player name lookup: {len(name_lookup):,} unique gsis_ids")

    # ── Step 1: per-team-season targets (all receivers) ───────────
    pbp = pbp.dropna(subset=["receiver_player_id", "season",
                                "posteam", "game_id", "week"])
    pbp = pbp[pbp["play_type"] == "pass"].copy()
    pbp["season"] = pbp["season"].astype(int)
    pbp["week"] = pbp["week"].astype(int)
    print(f"  pass plays w/ receiver: {len(pbp):,}")

    team_total = pbp.groupby(["posteam", "season"]).size().rename(
        "team_total_targets").reset_index()
    player_team = pbp.groupby(["receiver_player_id", "posteam",
                                  "season"]).size().rename(
        "player_targets").reset_index()
    player_team = player_team.merge(team_total,
                                       on=["posteam", "season"])
    player_team["target_share"] = (player_team["player_targets"]
                                       / player_team["team_total_targets"])

    # Rank within team-season
    player_team["rank"] = player_team.groupby(
        ["posteam", "season"]
    )["target_share"].rank(ascending=False, method="first")
    top = player_team[player_team["rank"] <= TOP_N].copy()
    print(f"  top-{TOP_N} player-team-seasons: {len(top):,}")

    # Resolve player IDs to display names (season-independent)
    top = top.merge(
        name_lookup.rename(columns={"player_id": "receiver_player_id"}),
        on="receiver_player_id", how="left"
    )
    matched = top["player_display_name"].notna().sum()
    print(f"  name-matched: {matched}/{len(top)} "
          f"({matched/len(top):.0%})")

    # Drop unmatched (mostly RBs we don't have a name for)
    top = top.dropna(subset=["player_display_name"]).copy()

    # ── Step 2: snap counts join by (name, team, season, week) ────
    snaps = snaps[snaps["season"] >= 2016].copy()
    snaps["season"] = snaps["season"].astype(int)
    snaps["week"] = snaps["week"].astype(int)
    snaps["norm_name"] = _normalize_name(snaps["player"])
    top["norm_name"] = _normalize_name(top["player_display_name"])

    # snap_counts uses team abbreviations that should match posteam
    snaps_join = snaps[["norm_name", "team", "season", "week",
                          "offense_pct"]].copy()
    snaps_join["offense_pct"] = pd.to_numeric(snaps_join["offense_pct"],
                                                  errors="coerce").fillna(0)

    # Build the panel: every team-season-week × top-N receiver row
    # Each team has ~17 weeks per season (+ playoff).
    all_weeks = pbp[["posteam", "season", "week"]].drop_duplicates()
    all_weeks = all_weeks.rename(columns={"posteam": "team"})
    print(f"  team-game rows: {len(all_weeks):,}")

    panel = top.rename(columns={"posteam": "team"}).merge(
        all_weeks, on=["team", "season"], how="left"
    )
    panel = panel.merge(snaps_join,
                            on=["norm_name", "team", "season", "week"],
                            how="left")
    panel["offense_pct"] = panel["offense_pct"].fillna(0)
    panel["snap_pct"] = panel["offense_pct"].clip(0, 1)
    print(f"  panel rows: {len(panel):,}")

    # Weapons strength: weighted-by-target-share avg snap availability
    panel["contrib"] = panel["target_share"] * panel["snap_pct"]
    weapons = panel.groupby(["team", "season", "week"]).agg(
        contrib_sum=("contrib", "sum"),
        share_sum=("target_share", "sum"),
        weapons_full_count=("snap_pct", lambda x: (x >= 0.50).sum()),
    ).reset_index()
    weapons["weapons_strength"] = (weapons["contrib_sum"]
                                       / weapons["share_sum"]).clip(0, 1)
    weapons = weapons[["team", "season", "week",
                          "weapons_strength",
                          "weapons_full_count"]]
    print(f"  team-weeks output: {len(weapons):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    weapons.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== Distribution of weapons_strength ===")
    print(weapons["weapons_strength"].describe().to_string())
    print()
    print("=== Lowest 10 weapons-strength games (2024) ===")
    s24 = weapons[weapons["season"] == 2024].nsmallest(10,
                                                          "weapons_strength")
    print(s24.to_string(index=False))
    print()
    print("=== Lions weapons availability 2024 (Goff games) ===")
    det = weapons[(weapons["team"] == "DET")
                  & (weapons["season"] == 2024)].sort_values("week")
    print(det.to_string(index=False))


if __name__ == "__main__":
    main()

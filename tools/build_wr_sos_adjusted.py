"""SOS-adjusted WR per-season z-cols.

Output: data/wr_sos_adjusted_z.parquet

Per-target opponent pass-defense adjustment, FILTERED to WR
targets (so we're measuring how each defense handles WR-
specific routes, not RB/TE).

Stats adjusted (the highest-leverage per-target WR stats):
  yards_per_target  → adj_yards_per_target_z
  epa_per_target    → adj_epa_per_target_z
  success_rate      → adj_success_rate_z
  catch_rate        → adj_catch_rate_z

Stats SKIPPED for SOS (already player-isolated or not amenable):
  avg_separation       (NGS — already player skill, no adjustment)
  yac_above_exp        (NGS player-isolated)
  target_share         (player's role on his own offense, no opp adj)
  air_yards_share      (same)
  wopr                 (same)

Note: QB quality is a confound we DON'T try to normalize. A WR with
Mahomes will look better than a WR with a backup. We accept this.
The honest framing: GAS measures WR productivity in their actual
context — partially co-produced by their QB.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
ROSTERS = REPO / "data" / "nfl_rosters.parquet"
OUT = REPO / "data" / "wr_sos_adjusted_z.parquet"

MIN_TARGETS = 30


def _z_within_season(df: pd.DataFrame, col: str,
                       season_col: str = "season") -> pd.Series:
    means = df.groupby(season_col)[col].transform("mean")
    stds = df.groupby(season_col)[col].transform("std").replace(0,
                                                                  np.nan)
    return ((df[col] - means) / stds).fillna(0)


def main() -> None:
    print("→ loading pbp + rosters...")
    pbp = pd.read_parquet(PBP)
    rosters = pd.read_parquet(ROSTERS)

    # Filter to pass plays with a WR receiver
    pbp = pbp.dropna(subset=["receiver_player_id", "season",
                                "defteam", "epa", "receiving_yards"])
    pbp = pbp[pbp["play_type"] == "pass"].copy()
    pbp["season"] = pbp["season"].astype(int)
    print(f"  pass plays w/ receiver: {len(pbp):,}")

    wr_ids = set(rosters[rosters["position"] == "WR"][
        "gsis_id"
    ].dropna().tolist())
    print(f"  WR roster ids: {len(wr_ids):,}")

    # Filter to WR-targeted passes ONLY for the per-defense
    # allowance computation
    wr_pass = pbp[pbp["receiver_player_id"].isin(wr_ids)].copy()
    print(f"  WR-targeted pass plays: {len(wr_pass):,}")

    # League averages per season (WR-target subset only)
    league = wr_pass.groupby("season").agg(
        lg_epa=("epa", "mean"),
        lg_success=("success", "mean"),
        lg_yards=("receiving_yards", "mean"),
        lg_complete=("complete_pass", "mean"),
    ).reset_index()
    print(f"  league seasons: {len(league)}")

    # Per-(defteam, season) WR-target allowance
    opp_def = wr_pass.groupby(["defteam", "season"]).agg(
        opp_epa_allowed=("epa", "mean"),
        opp_success_allowed=("success", "mean"),
        opp_yards_allowed=("receiving_yards", "mean"),
        opp_complete_allowed=("complete_pass", "mean"),
    ).reset_index()
    print(f"  team-season WR-defense rows: {len(opp_def):,}")

    # Save the WR-defense quality table for reuse
    wr_def_path = REPO / "data" / "team_wr_def_quality.parquet"
    opp_def.rename(columns={"defteam": "team"}).to_parquet(
        wr_def_path, index=False)
    print(f"  ✓ side-effect: wrote {wr_def_path.relative_to(REPO)}")

    # Attach baselines + compute adjusted per-target values
    wr_pass = wr_pass.merge(opp_def, on=["defteam", "season"],
                              how="left")
    wr_pass = wr_pass.merge(league, on="season", how="left")

    wr_pass["adj_epa"] = (wr_pass["epa"]
                          - wr_pass["opp_epa_allowed"]
                          + wr_pass["lg_epa"])
    wr_pass["adj_success"] = (wr_pass["success"]
                                - wr_pass["opp_success_allowed"]
                                + wr_pass["lg_success"])
    wr_pass["adj_yards"] = (wr_pass["receiving_yards"]
                              - wr_pass["opp_yards_allowed"]
                              + wr_pass["lg_yards"])
    wr_pass["adj_complete"] = (wr_pass["complete_pass"]
                                 - wr_pass["opp_complete_allowed"]
                                 + wr_pass["lg_complete"])

    # Aggregate per (receiver, season)
    grp = wr_pass.groupby(["receiver_player_id", "season"])
    agg = grp.agg(
        targets=("epa", "size"),
        adj_epa_per_target=("adj_epa", "mean"),
        adj_success_rate=("adj_success", "mean"),
        adj_yards_per_target=("adj_yards", "mean"),
        adj_catch_rate=("adj_complete", "mean"),
    ).reset_index()
    agg = agg[agg["targets"] >= MIN_TARGETS].copy()
    print(f"  qualifying WR-seasons: {len(agg)}")

    agg = agg.rename(columns={
        "receiver_player_id": "player_id",
        "season": "season_year",
    })

    for col in ["adj_epa_per_target", "adj_success_rate",
                 "adj_yards_per_target", "adj_catch_rate"]:
        agg[f"{col}_z"] = _z_within_season(agg, col, "season_year")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    s24 = agg[agg["season_year"] == 2024].nlargest(8,
                                                      "adj_epa_per_target_z")
    print("=== 2024 WR top by adj EPA/target ===")
    print(s24[["player_id", "targets",
                "adj_epa_per_target", "adj_epa_per_target_z",
                "adj_yards_per_target", "adj_catch_rate"]
              ].to_string())


if __name__ == "__main__":
    main()

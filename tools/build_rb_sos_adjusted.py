"""SOS-adjusted RB per-season z-cols.

Output: data/rb_sos_adjusted_z.parquet

Per-rush opponent run-defense subtraction. ONLY the stats that aren't
already player-isolated get SOS-adjusted:

  epa_per_rush       → adj_epa_per_rush_z
  rush_success_rate  → adj_rush_success_rate_z
  explosive_run_rate → adj_explosive_run_rate_z

Stats SKIPPED for SOS (already opponent-controlled):
  ryoe_per_att          (NGS already controls for box / defender placement)
  yards_after_contact   (player-only effort, post-contact)
  broken_tackles        (player effort)

Receiving stats are NOT SOS-adjusted in v1 — for an RB receiving,
the QB's quality matters more than opp pass defense, and we don't
have a clean QB-adjustment substrate.

Methodology
-----------
1. Compute league per-rush mean (epa, success, explosive=run≥10) per season
2. Compute per-(team, season) defensive allowance (the same metrics
   averaged over all RUNS faced by that team's defense)
3. For each rush by an RB: adj = actual − opp_def_allowed + league_mean
4. Aggregate per (player, season) and z-score within season

Min 50 rushes per season for inclusion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
ROSTERS = REPO / "data" / "nfl_rosters.parquet"
OUT = REPO / "data" / "rb_sos_adjusted_z.parquet"

MIN_RUSHES = 50


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

    # Filter to RB rushes
    pbp = pbp.dropna(subset=["rusher_player_id", "season", "defteam",
                                "epa", "rushing_yards"])
    pbp = pbp[pbp["play_type"] == "run"].copy()
    pbp["season"] = pbp["season"].astype(int)
    pbp["explosive"] = (pbp["rushing_yards"] >= 10).astype(int)
    print(f"  total run plays: {len(pbp):,}")

    # Position lookup: only count RB rushes for the player aggregation
    rb_ids = set(rosters[rosters["position"].isin(["RB", "FB"])][
        "gsis_id"
    ].dropna().tolist())
    print(f"  RB/FB roster ids: {len(rb_ids):,}")

    # ── League averages per season
    league = pbp.groupby("season").agg(
        lg_epa=("epa", "mean"),
        lg_success=("success", "mean"),
        lg_explosive=("explosive", "mean"),
    ).reset_index()
    print(f"  league seasons: {len(league)}")

    # ── PER-PLAYER LEAVE-ONE-OUT SOS ──────────────────────────────
    # See build_wr_sos_adjusted.py for full docstring. Short version:
    # opp baseline = (D_total − player_v_D_total) / (D_count − player_v_D_count).
    print("→ computing per-defense + per-player season totals...")
    def_totals = pbp.groupby(["defteam", "season"]).agg(
        d_epa_total=("epa", "sum"),
        d_success_total=("success", "sum"),
        d_explosive_total=("explosive", "sum"),
        d_count=("epa", "size"),
    ).reset_index()
    player_v_def = pbp.groupby(
        ["rusher_player_id", "defteam", "season"]
    ).agg(
        p_v_d_epa=("epa", "sum"),
        p_v_d_success=("success", "sum"),
        p_v_d_explosive=("explosive", "sum"),
        p_v_d_count=("epa", "size"),
    ).reset_index()
    print(f"  team-season run-def rows: {len(def_totals):,}")
    print(f"  player-vs-def rows: {len(player_v_def):,}")

    run_def_path = REPO / "data" / "team_run_def_quality.parquet"
    save = def_totals.copy()
    save["opp_run_epa_allowed"] = save["d_epa_total"] / save["d_count"]
    save["opp_run_success_allowed"] = (save["d_success_total"]
                                          / save["d_count"])
    save["opp_run_explosive_allowed"] = (save["d_explosive_total"]
                                            / save["d_count"])
    save = save.rename(columns={"defteam": "team"})[[
        "team", "season", "opp_run_epa_allowed",
        "opp_run_success_allowed", "opp_run_explosive_allowed"
    ]]
    save.to_parquet(run_def_path, index=False)
    print(f"  ✓ side-effect: wrote {run_def_path.relative_to(REPO)}")

    pbp = pbp.merge(def_totals, on=["defteam", "season"], how="left")
    pbp = pbp.merge(player_v_def,
                      on=["rusher_player_id", "defteam", "season"],
                      how="left")
    pbp = pbp.merge(league, on="season", how="left")

    denom = (pbp["d_count"] - pbp["p_v_d_count"]).clip(lower=1)
    base_epa = ((pbp["d_epa_total"] - pbp["p_v_d_epa"]) / denom)
    base_success = ((pbp["d_success_total"] - pbp["p_v_d_success"]) / denom)
    base_explosive = ((pbp["d_explosive_total"]
                       - pbp["p_v_d_explosive"]) / denom)

    pbp["adj_epa"] = pbp["epa"] - base_epa + pbp["lg_epa"]
    pbp["adj_success"] = pbp["success"] - base_success + pbp["lg_success"]
    pbp["adj_explosive"] = (pbp["explosive"]
                              - base_explosive + pbp["lg_explosive"])

    # Filter to RB-only rushes
    pbp = pbp[pbp["rusher_player_id"].isin(rb_ids)]
    print(f"  RB rushes: {len(pbp):,}")

    # Aggregate per (rusher, season)
    grp = pbp.groupby(["rusher_player_id", "season"])
    agg = grp.agg(
        rushes=("epa", "size"),
        adj_epa_per_rush=("adj_epa", "mean"),
        adj_rush_success_rate=("adj_success", "mean"),
        adj_explosive_run_rate=("adj_explosive", "mean"),
    ).reset_index()
    agg = agg[agg["rushes"] >= MIN_RUSHES].copy()
    print(f"  qualifying RB-seasons: {len(agg)}")

    # Rename for join with master
    agg = agg.rename(columns={
        "rusher_player_id": "player_id",
        "season": "season_year",
    })

    # Z-score within season
    for col in ["adj_epa_per_rush", "adj_rush_success_rate",
                 "adj_explosive_run_rate"]:
        agg[f"{col}_z"] = _z_within_season(agg, col, "season_year")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    s24 = agg[agg["season_year"] == 2024].nlargest(8,
                                                      "adj_epa_per_rush_z")
    print("=== 2024 — SOS-adjusted EPA/rush leaders ===")
    print(s24[["player_id", "rushes",
                "adj_epa_per_rush", "adj_epa_per_rush_z",
                "adj_rush_success_rate", "adj_explosive_run_rate"]
              ].to_string())


if __name__ == "__main__":
    main()

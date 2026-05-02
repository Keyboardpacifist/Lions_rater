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

    # ── PER-PLAYER LEAVE-ONE-OUT SOS ──────────────────────────────
    # When grading player P against defense D, the opponent baseline
    # should EXCLUDE all of P's contribution to D's allowance — not
    # just one play, but every play P had vs D in that season.
    #
    # Math (per (player, season, defteam) cell):
    #   p_v_d_total = sum(player's stat vs D this season)
    #   p_v_d_count = count(player's plays vs D this season)
    #   D_excl_p_total = D_total - p_v_d_total
    #   D_excl_p_count = D_count - p_v_d_count
    #   adj_baseline = D_excl_p_total / D_excl_p_count
    #
    #   adj(play_i) = play_i - adj_baseline + league_mean
    #
    # This removes ~6-12% of D's volume for elite players (real
    # chicken-and-egg fix). Per-PLAY LOO only removed 1/600 — too
    # mild to matter.
    print("→ computing per-defense + per-player season totals...")
    def_totals = wr_pass.groupby(["defteam", "season"]).agg(
        d_epa_total=("epa", "sum"),
        d_success_total=("success", "sum"),
        d_yards_total=("receiving_yards", "sum"),
        d_complete_total=("complete_pass", "sum"),
        d_count=("epa", "size"),
    ).reset_index()
    player_v_def = wr_pass.groupby(
        ["receiver_player_id", "defteam", "season"]
    ).agg(
        p_v_d_epa=("epa", "sum"),
        p_v_d_success=("success", "sum"),
        p_v_d_yards=("receiving_yards", "sum"),
        p_v_d_complete=("complete_pass", "sum"),
        p_v_d_count=("epa", "size"),
    ).reset_index()
    print(f"  team-season WR-defense rows: {len(def_totals):,}")
    print(f"  player-vs-def rows: {len(player_v_def):,}")

    # Save the WR-defense quality table (mean form, for reuse)
    wr_def_path = REPO / "data" / "team_wr_def_quality.parquet"
    wr_def_save = def_totals.copy()
    wr_def_save["opp_epa_allowed"] = (wr_def_save["d_epa_total"]
                                          / wr_def_save["d_count"])
    wr_def_save["opp_success_allowed"] = (wr_def_save["d_success_total"]
                                              / wr_def_save["d_count"])
    wr_def_save["opp_yards_allowed"] = (wr_def_save["d_yards_total"]
                                            / wr_def_save["d_count"])
    wr_def_save["opp_complete_allowed"] = (wr_def_save["d_complete_total"]
                                                / wr_def_save["d_count"])
    wr_def_save = wr_def_save.rename(columns={"defteam": "team"})[[
        "team", "season", "opp_epa_allowed", "opp_success_allowed",
        "opp_yards_allowed", "opp_complete_allowed"
    ]]
    wr_def_save.to_parquet(wr_def_path, index=False)
    print(f"  ✓ side-effect: wrote {wr_def_path.relative_to(REPO)}")

    # Attach defense totals + player_v_def + league
    wr_pass = wr_pass.merge(def_totals, on=["defteam", "season"],
                              how="left")
    wr_pass = wr_pass.merge(player_v_def,
                              on=["receiver_player_id", "defteam",
                                  "season"], how="left")
    wr_pass = wr_pass.merge(league, on="season", how="left")

    # Per-player LOO baseline = (D − player_v_D) / (count − player_count)
    denom = (wr_pass["d_count"] - wr_pass["p_v_d_count"]).clip(lower=1)
    base_epa = ((wr_pass["d_epa_total"] - wr_pass["p_v_d_epa"]) / denom)
    base_success = ((wr_pass["d_success_total"]
                      - wr_pass["p_v_d_success"]) / denom)
    base_yards = ((wr_pass["d_yards_total"]
                    - wr_pass["p_v_d_yards"]) / denom)
    base_complete = ((wr_pass["d_complete_total"]
                       - wr_pass["p_v_d_complete"]) / denom)

    wr_pass["adj_epa"] = (wr_pass["epa"]
                          - base_epa + wr_pass["lg_epa"])
    wr_pass["adj_success"] = (wr_pass["success"]
                                - base_success + wr_pass["lg_success"])
    wr_pass["adj_yards"] = (wr_pass["receiving_yards"]
                              - base_yards + wr_pass["lg_yards"])
    wr_pass["adj_complete"] = (wr_pass["complete_pass"]
                                 - base_complete + wr_pass["lg_complete"])

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

"""SOS-adjusted RB per-season z-cols WITH OL-context adjustment.

Output: data/rb_sos_adjusted_z.parquet

Two adjustments per RB rush:

1. **Defense LOO (per-player).** Subtract the defense's typical
   run-allowance excluding this RB's contribution. Captures
   schedule-strength.

2. **OL-context credit removal.** A great OL makes any RB look
   better; a bad OL holds even Saquon back. We estimate the
   league-wide effect of OL run-block quality on per-rush stats
   (β > 0) and subtract that contribution. RB grade left over is
   what the RB did *above what his OL would predict*.
   OL quality = team-season top-5-starter avg `gas_run_blocking_grade`
   from ol_gas_seasons.parquet. Centered on 50 (grade midpoint).

Stats adjusted (rushing only):
  epa_per_rush       → adj_epa_per_rush_z
  rush_success_rate  → adj_rush_success_rate_z
  explosive_run_rate → adj_explosive_run_rate_z

Stats SKIPPED for adjustment (already player-isolated):
  ryoe_per_att          (NGS already controls for box / defender placement)
  yards_after_contact   (player-only effort, post-contact)
  broken_tackles        (player effort)

Receiving stats not adjusted in v1.

Min 50 rushes per season for inclusion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
ROSTERS = REPO / "data" / "nfl_rosters.parquet"
OL_GAS = REPO / "data" / "ol_gas_seasons.parquet"
OUT = REPO / "data" / "rb_sos_adjusted_z.parquet"

MIN_RUSHES = 50
TOP_N_OL = 5     # avg the team's top-5 starters by snap share


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

    pbp["def_adj_epa"] = pbp["epa"] - base_epa + pbp["lg_epa"]
    pbp["def_adj_success"] = (pbp["success"] - base_success
                                + pbp["lg_success"])
    pbp["def_adj_explosive"] = (pbp["explosive"]
                                  - base_explosive + pbp["lg_explosive"])

    # ── OL-CONTEXT ADJUSTMENT ──────────────────────────────────────
    # Compute team-season OL run-block quality (top-5-starter avg of
    # gas_run_blocking_grade). Then estimate β league-wide and net out
    # OL contribution from per-rush stats.
    print("→ computing team-season OL run-block quality...")
    ol = pd.read_parquet(OL_GAS)
    if "gas_run_blocking_grade" not in ol.columns:
        raise RuntimeError("ol_gas_seasons.parquet missing "
                              "gas_run_blocking_grade")
    # Top-5 starters by snap share per team-season
    ol_top5 = (ol.sort_values(["team", "season_year",
                                  "snap_share"], ascending=[True, True,
                                                              False])
                 .groupby(["team", "season_year"]).head(TOP_N_OL))
    team_ol = ol_top5.groupby(["team", "season_year"]).agg(
        ol_run_grade=("gas_run_blocking_grade", "mean"),
    ).reset_index().rename(columns={"team": "posteam",
                                        "season_year": "season"})
    print(f"  team-season OL run grades: {len(team_ol):,}")
    # League mean OL grade (centering)
    lg_ol = team_ol["ol_run_grade"].mean()
    team_ol["ol_run_grade_centered"] = (team_ol["ol_run_grade"] - lg_ol)
    print(f"  league mean OL run grade: {lg_ol:.2f}")

    pbp = pbp.merge(team_ol[["posteam", "season",
                                "ol_run_grade_centered"]],
                      on=["posteam", "season"], how="left")
    matched = pbp["ol_run_grade_centered"].notna().sum()
    print(f"  rushes w/ OL grade: {matched:,}/{len(pbp):,} "
          f"({matched/len(pbp):.0%})")
    pbp["ol_centered"] = pbp["ol_run_grade_centered"].fillna(0)

    # Estimate β on def-residuals (cleaner — opponent already removed)
    var_o = (pbp["ol_centered"] ** 2).mean()
    betas = {}
    for stat in ["epa", "success", "explosive"]:
        col = f"def_adj_{stat}"
        cov = ((pbp[col] - pbp[col].mean())
               * pbp["ol_centered"]).mean()
        betas[stat] = cov / var_o if var_o > 0 else 0.0
        print(f"  β_{stat} on OL-grade: {betas[stat]:+.5f}")

    # Final adjustment: defense + OL
    pbp["adj_epa"] = (pbp["def_adj_epa"]
                        - betas["epa"] * pbp["ol_centered"])
    pbp["adj_success"] = (pbp["def_adj_success"]
                            - betas["success"] * pbp["ol_centered"])
    pbp["adj_explosive"] = (pbp["def_adj_explosive"]
                              - betas["explosive"] * pbp["ol_centered"])

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

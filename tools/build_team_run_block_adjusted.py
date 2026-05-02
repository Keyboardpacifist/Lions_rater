"""SOS-adjusted team run-block stats with RB-talent credit removed.

Output: data/team_run_block_adjusted.parquet

Two adjustments applied per RB rush:

1. **Defense LOO (per-team).** Subtract the opposing run defense's
   typical EPA / success / explosive-rate allowance, EXCLUDING this
   team's plays vs that defense. Captures schedule-strength.

2. **RB-talent credit.** Saquon, Henry, McCaffrey make any OL look
   elite by breaking tackles and creating yards out of nothing. Bad
   RBs make good OL look pedestrian. Method: regress per-rush EPA
   league-wide on the rusher's `ryoe_per_att_z` (NGS Rush Yards Over
   Expected — already box-controlled, so the cleanest "RB skill above
   situation" signal we have). Subtract β × ryoe per rush.

   ryoe is the right metric here (vs raw yards/carry) because NGS
   already controls for the situation the OL created (box count,
   defender placement). What's left in ryoe is mostly RB skill.

Output schema (per (team, season)):
  team, season, rushes,
  adj_team_run_epa_z, adj_team_run_success_z,
  adj_team_run_explosive_z

Used by lib_ol_gas.py — run_blocking bundle adds these adj_team_*_z
columns ALONGSIDE the existing gap-level pos_run_*_z. Gap-level
preserves within-team granularity (LT/LG side vs C vs RG/RT side);
team-level RB-adjusted gives the cleaner causal signal. Both inform
the bundle (Option B from the user discussion).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
RB_GAS = REPO / "data" / "rb_gas_seasons.parquet"
ROSTERS = REPO / "data" / "nfl_rosters.parquet"
OUT = REPO / "data" / "team_run_block_adjusted.parquet"

MIN_RUSHES = 200   # team-season threshold


def _z_within_season(df: pd.DataFrame, col: str,
                       season_col: str = "season") -> pd.Series:
    means = df.groupby(season_col)[col].transform("mean")
    stds = df.groupby(season_col)[col].transform("std").replace(0,
                                                                  np.nan)
    return ((df[col] - means) / stds).fillna(0)


def main() -> None:
    print("→ loading pbp + rb_gas + rosters...")
    pbp = pd.read_parquet(PBP)
    rb_gas = pd.read_parquet(RB_GAS)
    rosters = pd.read_parquet(ROSTERS)

    pbp = pbp.dropna(subset=["rusher_player_id", "season", "defteam",
                                "posteam", "epa", "rushing_yards"])
    pbp = pbp[pbp["play_type"] == "run"].copy()
    pbp["season"] = pbp["season"].astype(int)
    pbp["explosive"] = (pbp["rushing_yards"] >= 10).astype(int)
    pbp["success"] = pbp["success"].fillna(0)
    print(f"  total run plays: {len(pbp):,}")

    # Filter to RB rushes
    rb_ids = set(rosters[rosters["position"].isin(["RB", "FB"])][
        "gsis_id"
    ].dropna().tolist())
    pbp = pbp[pbp["rusher_player_id"].isin(rb_ids)].copy()
    print(f"  RB rushes: {len(pbp):,}")

    # Attach rusher's ryoe_per_att_z (RB talent signal)
    rb_skill = rb_gas[["player_id", "season_year",
                          "ryoe_per_att_z"]].rename(
        columns={"player_id": "rusher_player_id",
                 "season_year": "season"})
    pbp = pbp.merge(rb_skill, on=["rusher_player_id", "season"],
                      how="left")
    matched = pbp["ryoe_per_att_z"].notna().sum()
    print(f"  rushes w/ RB ryoe-z: {matched:,}/{len(pbp):,} "
          f"({matched/len(pbp):.0%})")
    # Default missing to 0 (replacement-level RB)
    pbp["ryoe_z"] = pbp["ryoe_per_att_z"].fillna(0)

    # League means per season (centering)
    league = pbp.groupby("season").agg(
        lg_epa=("epa", "mean"),
        lg_success=("success", "mean"),
        lg_explosive=("explosive", "mean"),
    ).reset_index()
    print(f"  league seasons: {len(league)}")

    stats = ["epa", "success", "explosive"]

    # ── PER-DEFENSE LOO (per-team) ─────────────────────────────────
    print("→ computing per-defense + per-(team, defense) totals...")
    def_totals = pbp.groupby(["defteam", "season"]).agg(
        d_epa=("epa", "sum"),
        d_success=("success", "sum"),
        d_explosive=("explosive", "sum"),
        d_count=("epa", "size"),
    ).reset_index()
    team_v_def = pbp.groupby(["posteam", "defteam", "season"]).agg(
        t_v_d_epa=("epa", "sum"),
        t_v_d_success=("success", "sum"),
        t_v_d_explosive=("explosive", "sum"),
        t_v_d_count=("epa", "size"),
    ).reset_index()

    pbp = pbp.merge(def_totals, on=["defteam", "season"], how="left")
    pbp = pbp.merge(team_v_def,
                      on=["posteam", "defteam", "season"], how="left")
    pbp = pbp.merge(league, on="season", how="left")

    denom = (pbp["d_count"] - pbp["t_v_d_count"]).clip(lower=1)
    base_epa = (pbp["d_epa"] - pbp["t_v_d_epa"]) / denom
    base_success = (pbp["d_success"] - pbp["t_v_d_success"]) / denom
    base_explosive = ((pbp["d_explosive"] - pbp["t_v_d_explosive"])
                       / denom)

    # Defense-residual stats (used for β estimation + final adj)
    pbp["def_adj_epa"] = pbp["epa"] - (base_epa - pbp["lg_epa"])
    pbp["def_adj_success"] = (pbp["success"]
                                - (base_success - pbp["lg_success"]))
    pbp["def_adj_explosive"] = (pbp["explosive"]
                                  - (base_explosive
                                     - pbp["lg_explosive"]))

    # ── ESTIMATE RB-TALENT β ───────────────────────────────────────
    # Regress defense-residual run stat on RB ryoe_per_att_z.
    # β should be POSITIVE: high ryoe = better RB = higher per-rush
    # EPA. We then subtract β × ryoe per play to net out RB credit.
    print("→ estimating RB-talent β (on def-residual)...")
    var_r = (pbp["ryoe_z"] ** 2).mean()
    betas = {}
    for stat in stats:
        col = f"def_adj_{stat}"
        cov = ((pbp[col] - pbp[col].mean())
               * pbp["ryoe_z"]).mean()
        betas[stat] = cov / var_r if var_r > 0 else 0.0
        print(f"  β_{stat} on RB ryoe-z: {betas[stat]:+.4f}")

    # Apply RB credit subtraction
    pbp["adj_epa"] = (pbp["def_adj_epa"]
                        - betas["epa"] * pbp["ryoe_z"])
    pbp["adj_success"] = (pbp["def_adj_success"]
                            - betas["success"] * pbp["ryoe_z"])
    pbp["adj_explosive"] = (pbp["def_adj_explosive"]
                              - betas["explosive"] * pbp["ryoe_z"])

    # ── AGGREGATE PER (TEAM, SEASON) ───────────────────────────────
    grp = pbp.groupby(["posteam", "season"])
    agg = grp.agg(
        rushes=("epa", "size"),
        adj_team_run_epa=("adj_epa", "mean"),
        adj_team_run_success=("adj_success", "mean"),
        adj_team_run_explosive=("adj_explosive", "mean"),
    ).reset_index()
    agg = agg[agg["rushes"] >= MIN_RUSHES].copy()
    print(f"  qualifying team-seasons: {len(agg)}")

    agg = agg.rename(columns={"posteam": "team"})

    for col in ["adj_team_run_epa", "adj_team_run_success",
                 "adj_team_run_explosive"]:
        agg[f"{col}_z"] = _z_within_season(agg, col)

    out = agg[["team", "season", "rushes",
                "adj_team_run_epa", "adj_team_run_success",
                "adj_team_run_explosive",
                "adj_team_run_epa_z", "adj_team_run_success_z",
                "adj_team_run_explosive_z"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== 2024 best run-block (post RB-talent adjustment) ===")
    s24 = out[out["season"] == 2024].nlargest(8,
                                                  "adj_team_run_epa_z")
    print(s24[["team", "rushes", "adj_team_run_epa",
                "adj_team_run_epa_z", "adj_team_run_success_z",
                "adj_team_run_explosive_z"]].to_string(index=False))
    print()
    print("=== 2024 worst run-block ===")
    s24 = out[out["season"] == 2024].nsmallest(8,
                                                   "adj_team_run_epa_z")
    print(s24[["team", "rushes", "adj_team_run_epa",
                "adj_team_run_epa_z", "adj_team_run_success_z",
                "adj_team_run_explosive_z"]].to_string(index=False))
    print()
    print("=== Lions/Eagles/Ravens 2024 ===")
    focus = out[(out["season"] == 2024)
                 & out["team"].isin(["DET", "PHI", "BAL", "PHI",
                                       "BUF", "GB"])]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()

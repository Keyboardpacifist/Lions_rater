"""SOS-adjusted TE per-season z-cols — game-level QB + defense LOO.

Output: data/te_sos_adjusted_z.parquet

Stacked leave-one-out at GAME granularity. Mirrors WR builder
methodology with TE-targeted passes as the population:
  - Defense LOO at game level: subtract this defense's per-game
    allowance on TE-targeted passes, EXCLUDING this TE's contributions.
  - QB LOO at game level: subtract this QB's per-game level on
    TE-targeted passes, EXCLUDING this TE's contributions.

This isolates TE contribution from QB quality (Bowers shouldn't be
penalized for O'Connell; Kelce shouldn't be inflated by Mahomes)
AND defense quality.

Min 25 targets per season (lower than WR's 30 — TEs see fewer
targets, especially Y-TEs and rotational pieces).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
ROSTERS = REPO / "data" / "nfl_rosters.parquet"
OUT = REPO / "data" / "te_sos_adjusted_z.parquet"

MIN_TARGETS = 25
MIN_BASELINE_PLAYS = 6   # TEs see fewer targets per game; lower threshold

# QB-LOO strength. Matches WR builder. 0.75 leaves TEs with 25% of
# their QB's credit so elite-QB TEs (Kelce/Kittle) aren't over-penalized
# while bad-QB TEs (Bowers) still get most of the deserved boost.
QB_LOO_WEIGHT = 0.75


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

    pbp = pbp.dropna(subset=["receiver_player_id", "season",
                                "defteam", "epa", "passer_player_id",
                                "game_id"])
    pbp = pbp[pbp["play_type"] == "pass"].copy()
    pbp["receiving_yards"] = pbp["receiving_yards"].fillna(0)
    pbp["complete_pass"] = pbp["complete_pass"].fillna(0)
    pbp["season"] = pbp["season"].astype(int)
    print(f"  pass plays w/ receiver: {len(pbp):,}")

    te_ids = set(rosters[rosters["position"] == "TE"][
        "gsis_id"
    ].dropna().tolist())
    print(f"  TE roster ids: {len(te_ids):,}")

    te_pass = pbp[pbp["receiver_player_id"].isin(te_ids)].copy()
    print(f"  TE-targeted pass plays: {len(te_pass):,}")

    league = te_pass.groupby("season").agg(
        lg_epa=("epa", "mean"),
        lg_success=("success", "mean"),
        lg_yards=("receiving_yards", "mean"),
        lg_complete=("complete_pass", "mean"),
    ).reset_index()
    print(f"  league seasons: {len(league)}")

    stats = ["epa", "success", "receiving_yards", "complete_pass"]

    # ── GAME-LEVEL TOTALS ──────────────────────────────────────────
    print("→ computing per-game defense + QB totals...")
    def_g = te_pass.groupby(["defteam", "season",
                                "game_id"])[stats].agg(["sum", "size"])
    def_g.columns = [f"d_g_{s}" if a == "sum" else "d_g_count"
                       for s, a in def_g.columns]
    def_g = def_g.loc[:, ~def_g.columns.duplicated()].reset_index()

    qb_g = te_pass.groupby(["passer_player_id", "season",
                              "game_id"])[stats].agg(["sum", "size"])
    qb_g.columns = [f"q_g_{s}" if a == "sum" else "q_g_count"
                      for s, a in qb_g.columns]
    qb_g = qb_g.loc[:, ~qb_g.columns.duplicated()].reset_index()

    p_v_d_g = te_pass.groupby(["receiver_player_id", "defteam",
                                  "season", "game_id"])[stats].agg(
        ["sum", "size"])
    p_v_d_g.columns = [f"p_d_g_{s}" if a == "sum" else "p_d_g_count"
                          for s, a in p_v_d_g.columns]
    p_v_d_g = p_v_d_g.loc[:, ~p_v_d_g.columns.duplicated()].reset_index()

    p_v_q_g = te_pass.groupby(["receiver_player_id", "passer_player_id",
                                  "season", "game_id"])[stats].agg(
        ["sum", "size"])
    p_v_q_g.columns = [f"p_q_g_{s}" if a == "sum" else "p_q_g_count"
                          for s, a in p_v_q_g.columns]
    p_v_q_g = p_v_q_g.loc[:, ~p_v_q_g.columns.duplicated()].reset_index()
    print(f"  defense-game rows: {len(def_g):,}; "
          f"QB-game rows: {len(qb_g):,}")

    # ── SEASON-LEVEL FALLBACKS ─────────────────────────────────────
    print("→ computing season-level fallback baselines...")
    def_s = te_pass.groupby(["defteam", "season"])[stats].agg(
        ["sum", "size"])
    def_s.columns = [f"d_s_{s}" if a == "sum" else "d_s_count"
                       for s, a in def_s.columns]
    def_s = def_s.loc[:, ~def_s.columns.duplicated()].reset_index()
    qb_s = te_pass.groupby(["passer_player_id", "season"])[stats].agg(
        ["sum", "size"])
    qb_s.columns = [f"q_s_{s}" if a == "sum" else "q_s_count"
                      for s, a in qb_s.columns]
    qb_s = qb_s.loc[:, ~qb_s.columns.duplicated()].reset_index()
    p_v_d_s = te_pass.groupby(["receiver_player_id", "defteam",
                                  "season"])[stats].agg(["sum", "size"])
    p_v_d_s.columns = [f"p_d_s_{s}" if a == "sum" else "p_d_s_count"
                          for s, a in p_v_d_s.columns]
    p_v_d_s = p_v_d_s.loc[:, ~p_v_d_s.columns.duplicated()].reset_index()
    p_v_q_s = te_pass.groupby(["receiver_player_id", "passer_player_id",
                                  "season"])[stats].agg(["sum", "size"])
    p_v_q_s.columns = [f"p_q_s_{s}" if a == "sum" else "p_q_s_count"
                          for s, a in p_v_q_s.columns]
    p_v_q_s = p_v_q_s.loc[:, ~p_v_q_s.columns.duplicated()].reset_index()

    # Side-effect: TE pass-defense quality table (mean form)
    te_def_path = REPO / "data" / "team_te_def_quality.parquet"
    te_def_save = def_s.copy()
    te_def_save["opp_te_epa_allowed"] = (te_def_save["d_s_epa"]
                                            / te_def_save["d_s_count"])
    te_def_save["opp_te_success_allowed"] = (te_def_save["d_s_success"]
                                                / te_def_save["d_s_count"])
    te_def_save["opp_te_yards_allowed"] = (te_def_save["d_s_receiving_yards"]
                                              / te_def_save["d_s_count"])
    te_def_save["opp_te_complete_allowed"] = (te_def_save["d_s_complete_pass"]
                                                  / te_def_save["d_s_count"])
    te_def_save = te_def_save.rename(columns={"defteam": "team"})[[
        "team", "season", "opp_te_epa_allowed", "opp_te_success_allowed",
        "opp_te_yards_allowed", "opp_te_complete_allowed"
    ]]
    te_def_save.to_parquet(te_def_path, index=False)
    print(f"  ✓ side-effect: wrote {te_def_path.relative_to(REPO)}")

    # ── MERGE ──────────────────────────────────────────────────────
    print("→ merging baselines onto play-by-play...")
    te_pass = te_pass.merge(def_g, on=["defteam", "season", "game_id"],
                              how="left")
    te_pass = te_pass.merge(qb_g, on=["passer_player_id", "season",
                                         "game_id"], how="left")
    te_pass = te_pass.merge(p_v_d_g, on=["receiver_player_id", "defteam",
                                             "season", "game_id"],
                              how="left")
    te_pass = te_pass.merge(p_v_q_g,
                              on=["receiver_player_id", "passer_player_id",
                                  "season", "game_id"], how="left")
    te_pass = te_pass.merge(def_s, on=["defteam", "season"], how="left")
    te_pass = te_pass.merge(qb_s, on=["passer_player_id", "season"],
                              how="left")
    te_pass = te_pass.merge(p_v_d_s, on=["receiver_player_id", "defteam",
                                             "season"], how="left")
    te_pass = te_pass.merge(p_v_q_s,
                              on=["receiver_player_id", "passer_player_id",
                                  "season"], how="left")
    te_pass = te_pass.merge(league, on="season", how="left")
    print(f"  merged play rows: {len(te_pass):,}")

    g_def_denom = te_pass["d_g_count"] - te_pass["p_d_g_count"]
    g_qb_denom = te_pass["q_g_count"] - te_pass["p_q_g_count"]
    use_g_def = g_def_denom >= MIN_BASELINE_PLAYS
    use_g_qb = g_qb_denom >= MIN_BASELINE_PLAYS
    s_def_denom = (te_pass["d_s_count"] - te_pass["p_d_s_count"]).clip(lower=1)
    s_qb_denom = (te_pass["q_s_count"] - te_pass["p_q_s_count"]).clip(lower=1)
    print(f"  game-baseline used (defense): "
          f"{use_g_def.sum():,} / {len(te_pass):,} "
          f"({use_g_def.mean():.1%})")
    print(f"  game-baseline used (QB): "
          f"{use_g_qb.sum():,} / {len(te_pass):,} "
          f"({use_g_qb.mean():.1%})")

    stat_to_lg = {
        "epa": "lg_epa",
        "success": "lg_success",
        "receiving_yards": "lg_yards",
        "complete_pass": "lg_complete",
    }

    for stat in stats:
        g_def_base = ((te_pass[f"d_g_{stat}"] - te_pass[f"p_d_g_{stat}"])
                      / g_def_denom.clip(lower=1))
        s_def_base = ((te_pass[f"d_s_{stat}"] - te_pass[f"p_d_s_{stat}"])
                      / s_def_denom)
        def_base = g_def_base.where(use_g_def, s_def_base)

        g_qb_base = ((te_pass[f"q_g_{stat}"] - te_pass[f"p_q_g_{stat}"])
                     / g_qb_denom.clip(lower=1))
        s_qb_base = ((te_pass[f"q_s_{stat}"] - te_pass[f"p_q_s_{stat}"])
                     / s_qb_denom)
        qb_base = g_qb_base.where(use_g_qb, s_qb_base)

        lg = te_pass[stat_to_lg[stat]]
        te_pass[f"adj_{stat}"] = (te_pass[stat]
                                    - (def_base - lg)
                                    - QB_LOO_WEIGHT * (qb_base - lg))

    grp = te_pass.groupby(["receiver_player_id", "season"])
    agg = grp.agg(
        targets=("epa", "size"),
        adj_epa_per_target=("adj_epa", "mean"),
        adj_success_rate=("adj_success", "mean"),
        adj_yards_per_target=("adj_receiving_yards", "mean"),
        adj_catch_rate=("adj_complete_pass", "mean"),
    ).reset_index()
    agg = agg[agg["targets"] >= MIN_TARGETS].copy()
    print(f"  qualifying TE-seasons: {len(agg)}")

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
    print("=== 2024 TE top by game-LOO adj EPA/target ===")
    print(s24[["player_id", "targets",
                "adj_epa_per_target", "adj_epa_per_target_z",
                "adj_yards_per_target", "adj_catch_rate"]
              ].to_string())


if __name__ == "__main__":
    main()

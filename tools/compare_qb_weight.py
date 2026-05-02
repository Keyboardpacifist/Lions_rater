"""One-shot: WR/TE GAS top-10 at full QB-LOO vs 75% QB-LOO (2024).

Doesn't write parquets — runs both versions in memory, prints the
2024 top-10 leaderboard side by side. Used to evaluate whether
softening the QB adjustment to 75% produces more defensible rankings.

Math reminder. For each pass play to receiver P from QB Q vs def D:
  full:    adj = play − 1.00·(qb_base − lg) − (def_base − lg)
  q75:     adj = play − 0.75·(qb_base − lg) − (def_base − lg)

Defense LOO held constant (1.00) — only QB-LOO weight varies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PBP = REPO / "data" / "game_pbp.parquet"
ROSTERS = REPO / "data" / "nfl_rosters.parquet"

MIN_BASELINE_PLAYS_WR = 8
MIN_BASELINE_PLAYS_TE = 6
MIN_TARGETS_WR = 30
MIN_TARGETS_TE = 25


def _z_within_season(df, col, season_col="season_year"):
    means = df.groupby(season_col)[col].transform("mean")
    stds = df.groupby(season_col)[col].transform("std").replace(0,
                                                                  np.nan)
    return ((df[col] - means) / stds).fillna(0)


def _build_sos(rec_pass: pd.DataFrame, min_baseline: int,
               min_targets: int, qb_weight: float) -> pd.DataFrame:
    stats = ["epa", "success", "receiving_yards", "complete_pass"]
    league = rec_pass.groupby("season").agg(
        lg_epa=("epa", "mean"),
        lg_success=("success", "mean"),
        lg_yards=("receiving_yards", "mean"),
        lg_complete=("complete_pass", "mean"),
    ).reset_index()

    def _gb(df, keys):
        out = df.groupby(keys)[stats].agg(["sum", "size"])
        out.columns = [f"{s}" if a == "sum" else "_count"
                       for s, a in out.columns]
        out = out.loc[:, ~out.columns.duplicated()].reset_index()
        return out

    def_g = _gb(rec_pass, ["defteam", "season", "game_id"])
    def_g = def_g.rename(columns={c: f"d_g_{c}" if c not in
                                     ["defteam", "season", "game_id"]
                                     else c for c in def_g.columns})
    qb_g = _gb(rec_pass, ["passer_player_id", "season", "game_id"])
    qb_g = qb_g.rename(columns={c: f"q_g_{c}" if c not in
                                   ["passer_player_id", "season",
                                    "game_id"] else c
                                   for c in qb_g.columns})
    p_v_d_g = _gb(rec_pass, ["receiver_player_id", "defteam",
                                "season", "game_id"])
    p_v_d_g = p_v_d_g.rename(columns={c: f"p_d_g_{c}" if c not in
                                         ["receiver_player_id",
                                          "defteam", "season",
                                          "game_id"] else c
                                         for c in p_v_d_g.columns})
    p_v_q_g = _gb(rec_pass, ["receiver_player_id", "passer_player_id",
                                "season", "game_id"])
    p_v_q_g = p_v_q_g.rename(columns={c: f"p_q_g_{c}" if c not in
                                         ["receiver_player_id",
                                          "passer_player_id",
                                          "season", "game_id"] else c
                                         for c in p_v_q_g.columns})

    def_s = _gb(rec_pass, ["defteam", "season"])
    def_s = def_s.rename(columns={c: f"d_s_{c}" if c not in
                                     ["defteam", "season"] else c
                                     for c in def_s.columns})
    qb_s = _gb(rec_pass, ["passer_player_id", "season"])
    qb_s = qb_s.rename(columns={c: f"q_s_{c}" if c not in
                                   ["passer_player_id", "season"]
                                   else c for c in qb_s.columns})
    p_v_d_s = _gb(rec_pass, ["receiver_player_id", "defteam", "season"])
    p_v_d_s = p_v_d_s.rename(columns={c: f"p_d_s_{c}" if c not in
                                         ["receiver_player_id",
                                          "defteam", "season"] else c
                                         for c in p_v_d_s.columns})
    p_v_q_s = _gb(rec_pass, ["receiver_player_id", "passer_player_id",
                                "season"])
    p_v_q_s = p_v_q_s.rename(columns={c: f"p_q_s_{c}" if c not in
                                         ["receiver_player_id",
                                          "passer_player_id",
                                          "season"] else c
                                         for c in p_v_q_s.columns})

    rp = rec_pass.merge(def_g, on=["defteam", "season", "game_id"],
                          how="left")
    rp = rp.merge(qb_g, on=["passer_player_id", "season", "game_id"],
                    how="left")
    rp = rp.merge(p_v_d_g, on=["receiver_player_id", "defteam",
                                  "season", "game_id"], how="left")
    rp = rp.merge(p_v_q_g, on=["receiver_player_id", "passer_player_id",
                                  "season", "game_id"], how="left")
    rp = rp.merge(def_s, on=["defteam", "season"], how="left")
    rp = rp.merge(qb_s, on=["passer_player_id", "season"], how="left")
    rp = rp.merge(p_v_d_s, on=["receiver_player_id", "defteam",
                                  "season"], how="left")
    rp = rp.merge(p_v_q_s, on=["receiver_player_id", "passer_player_id",
                                  "season"], how="left")
    rp = rp.merge(league, on="season", how="left")

    g_def_denom = rp["d_g__count"] - rp["p_d_g__count"]
    g_qb_denom = rp["q_g__count"] - rp["p_q_g__count"]
    use_g_def = g_def_denom >= min_baseline
    use_g_qb = g_qb_denom >= min_baseline
    s_def_denom = (rp["d_s__count"] - rp["p_d_s__count"]).clip(lower=1)
    s_qb_denom = (rp["q_s__count"] - rp["p_q_s__count"]).clip(lower=1)

    stat_to_lg = {"epa": "lg_epa", "success": "lg_success",
                  "receiving_yards": "lg_yards",
                  "complete_pass": "lg_complete"}

    for stat in stats:
        g_def_base = ((rp[f"d_g_{stat}"] - rp[f"p_d_g_{stat}"])
                      / g_def_denom.clip(lower=1))
        s_def_base = ((rp[f"d_s_{stat}"] - rp[f"p_d_s_{stat}"])
                      / s_def_denom)
        def_base = g_def_base.where(use_g_def, s_def_base)

        g_qb_base = ((rp[f"q_g_{stat}"] - rp[f"p_q_g_{stat}"])
                     / g_qb_denom.clip(lower=1))
        s_qb_base = ((rp[f"q_s_{stat}"] - rp[f"p_q_s_{stat}"])
                     / s_qb_denom)
        qb_base = g_qb_base.where(use_g_qb, s_qb_base)

        lg = rp[stat_to_lg[stat]]
        rp[f"adj_{stat}"] = (rp[stat]
                              - (def_base - lg)
                              - qb_weight * (qb_base - lg))

    grp = rp.groupby(["receiver_player_id", "season"])
    agg = grp.agg(
        targets=("epa", "size"),
        adj_epa_per_target=("adj_epa", "mean"),
        adj_success_rate=("adj_success", "mean"),
        adj_yards_per_target=("adj_receiving_yards", "mean"),
        adj_catch_rate=("adj_complete_pass", "mean"),
    ).reset_index()
    agg = agg[agg["targets"] >= min_targets].copy()
    agg = agg.rename(columns={"receiver_player_id": "player_id",
                                  "season": "season_year"})
    for col in ["adj_epa_per_target", "adj_success_rate",
                 "adj_yards_per_target", "adj_catch_rate"]:
        agg[f"{col}_z"] = _z_within_season(agg, col)
    return agg


def _grade_position(position: str, master_path: Path,
                     sos_full: pd.DataFrame, sos_q75: pd.DataFrame):
    from lib_wr_gas import compute_wr_gas
    from lib_te_gas import compute_te_gas
    fn = compute_wr_gas if position == "WR" else compute_te_gas

    master = pd.read_parquet(master_path)
    if "position" in master.columns:
        master = master[master["position"] == position].copy()

    out = {}
    for label, sos in [("full", sos_full), ("q75", sos_q75)]:
        merged = master.merge(sos, on=["player_id", "season_year"],
                                how="left")
        out[label] = fn(merged)
    return out


def main() -> None:
    print("→ loading pbp + rosters...")
    pbp_full = pd.read_parquet(PBP)
    rosters = pd.read_parquet(ROSTERS)

    pbp = pbp_full.dropna(subset=["receiver_player_id", "season",
                                       "defteam", "epa",
                                       "passer_player_id", "game_id"])
    pbp = pbp[pbp["play_type"] == "pass"].copy()
    pbp["receiving_yards"] = pbp["receiving_yards"].fillna(0)
    pbp["complete_pass"] = pbp["complete_pass"].fillna(0)
    pbp["season"] = pbp["season"].astype(int)

    wr_ids = set(rosters[rosters["position"] == "WR"]["gsis_id"]
                  .dropna().tolist())
    te_ids = set(rosters[rosters["position"] == "TE"]["gsis_id"]
                  .dropna().tolist())

    wr_pass = pbp[pbp["receiver_player_id"].isin(wr_ids)].copy()
    te_pass = pbp[pbp["receiver_player_id"].isin(te_ids)].copy()

    print("→ building WR SOS at QB-weight 1.00...")
    wr_full = _build_sos(wr_pass, MIN_BASELINE_PLAYS_WR, MIN_TARGETS_WR,
                            qb_weight=1.00)
    print("→ building WR SOS at QB-weight 0.75...")
    wr_q75 = _build_sos(wr_pass, MIN_BASELINE_PLAYS_WR, MIN_TARGETS_WR,
                           qb_weight=0.75)

    print("→ building TE SOS at QB-weight 1.00...")
    te_full = _build_sos(te_pass, MIN_BASELINE_PLAYS_TE, MIN_TARGETS_TE,
                            qb_weight=1.00)
    print("→ building TE SOS at QB-weight 0.75...")
    te_q75 = _build_sos(te_pass, MIN_BASELINE_PLAYS_TE, MIN_TARGETS_TE,
                           qb_weight=0.75)

    print("→ grading WR...")
    wr_g = _grade_position("WR",
                              REPO / "data" / "league_te_all_seasons.parquet",
                              wr_full, wr_q75)
    print("→ grading TE...")
    te_g = _grade_position("TE",
                              REPO / "data" / "league_te_all_seasons.parquet",
                              te_full, te_q75)

    def _top10(df, season=2024, min_g=10):
        return (df[(df["season_year"] == season)
                   & (df["games"] >= min_g)]
                .nlargest(10, "gas_score")
                [["player_display_name", "gas_score"]]
                .reset_index(drop=True))

    print()
    print("=" * 70)
    print("WR 2024 — top 10")
    print("=" * 70)
    full = _top10(wr_g["full"]).rename(
        columns={"player_display_name": "player_full",
                 "gas_score": "score_full"})
    q75 = _top10(wr_g["q75"]).rename(
        columns={"player_display_name": "player_q75",
                 "gas_score": "score_q75"})
    print(pd.concat([full, q75], axis=1).to_string())
    print()
    print("=" * 70)
    print("TE 2024 — top 10")
    print("=" * 70)
    full = _top10(te_g["full"]).rename(
        columns={"player_display_name": "player_full",
                 "gas_score": "score_full"})
    q75 = _top10(te_g["q75"]).rename(
        columns={"player_display_name": "player_q75",
                 "gas_score": "score_q75"})
    print(pd.concat([full, q75], axis=1).to_string())


if __name__ == "__main__":
    main()

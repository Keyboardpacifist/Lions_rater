"""One-shot: QB GAS top-10 at full weapons-LOO vs 75% (2024).

Prints 2024 QB top-10 leaderboard side-by-side at WEAPONS_WEIGHT=1.0
and WEAPONS_WEIGHT=0.75. Hand-picked betas held constant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DROPBACKS = REPO / "data" / "qb_dropbacks.parquet"
WEAPONS = REPO / "data" / "team_weapons_availability.parquet"

MIN_DROPBACKS = 100

BETAS = {"epa": 0.05, "success": 0.04, "complete_pass": 0.04}


def _z_within_season(df, col, season_col="season_year"):
    means = df.groupby(season_col)[col].transform("mean")
    stds = df.groupby(season_col)[col].transform("std").replace(0,
                                                                  np.nan)
    return ((df[col] - means) / stds).fillna(0)


def _build(weapons_weight: float) -> pd.DataFrame:
    db = pd.read_parquet(DROPBACKS)
    weapons = pd.read_parquet(WEAPONS)

    db = db.dropna(subset=["passer_player_id", "season", "defteam",
                              "epa", "week", "posteam"])
    db["season"] = db["season"].astype(int)
    db["week"] = db["week"].astype(int)
    db["sack_f"] = db["sack"].fillna(0)
    db["int_f"] = db["interception"].fillna(0)
    db["complete_pass"] = db["complete_pass"].fillna(0)
    db["success"] = db["success"].fillna(0)
    db = db.merge(
        weapons.rename(columns={"team": "posteam"})[
            ["posteam", "season", "week", "weapons_strength"]],
        on=["posteam", "season", "week"], how="left")
    lg_w = db["weapons_strength"].mean()
    db["weapons_strength"] = db["weapons_strength"].fillna(lg_w)

    league = db.groupby("season").agg(
        lg_epa=("epa", "mean"), lg_success=("success", "mean"),
        lg_complete=("complete_pass", "mean"),
        lg_sack=("sack_f", "mean"), lg_int=("int_f", "mean"),
    ).reset_index()

    stats = ["epa", "success", "complete_pass", "sack_f", "int_f"]
    def_s = db.groupby(["defteam", "season"])[stats].agg(["sum", "size"])
    def_s.columns = [f"d_s_{s}" if a == "sum" else "d_s_count"
                       for s, a in def_s.columns]
    def_s = def_s.loc[:, ~def_s.columns.duplicated()].reset_index()
    p_v_d_s = db.groupby(["passer_player_id", "defteam",
                              "season"])[stats].agg(["sum", "size"])
    p_v_d_s.columns = [f"p_d_s_{s}" if a == "sum" else "p_d_s_count"
                          for s, a in p_v_d_s.columns]
    p_v_d_s = p_v_d_s.loc[:, ~p_v_d_s.columns.duplicated()].reset_index()
    db = db.merge(def_s, on=["defteam", "season"], how="left")
    db = db.merge(p_v_d_s, on=["passer_player_id", "defteam", "season"],
                    how="left")
    db = db.merge(league, on="season", how="left")

    s_def_denom = (db["d_s_count"] - db["p_d_s_count"]).clip(lower=1)
    weapons_delta = db["weapons_strength"] - lg_w
    stat_to_lg = {"epa": "lg_epa", "success": "lg_success",
                  "complete_pass": "lg_complete",
                  "sack_f": "lg_sack", "int_f": "lg_int"}

    for stat in stats:
        def_base = ((db[f"d_s_{stat}"] - db[f"p_d_s_{stat}"])
                    / s_def_denom)
        lg = db[stat_to_lg[stat]]
        adj = db[stat] - (def_base - lg)
        if stat in BETAS:
            adj = adj - weapons_weight * BETAS[stat] * weapons_delta
        db[f"adj_{stat}"] = adj

    grp = db.groupby(["passer_player_id", "season"])
    agg = grp.agg(
        dropbacks=("epa", "size"),
        adj_epa_per_play=("adj_epa", "mean"),
        adj_success_rate=("adj_success", "mean"),
        adj_completion_pct=("adj_complete_pass", "mean"),
        adj_sack_rate=("adj_sack_f", "mean"),
        adj_int_rate=("adj_int_f", "mean"),
        avg_weapons_strength=("weapons_strength", "mean"),
    ).reset_index()
    agg = agg[agg["dropbacks"] >= MIN_DROPBACKS].copy()
    agg = agg.rename(columns={"passer_player_id": "player_id",
                                  "season": "season_year"})
    agg["adj_sack_rate_neg"] = -agg["adj_sack_rate"]
    agg["adj_int_rate_neg"] = -agg["adj_int_rate"]
    agg["adj_pass_epa_per_play_z"] = _z_within_season(agg,
                                                          "adj_epa_per_play")
    agg["adj_pass_success_rate_z"] = _z_within_season(agg,
                                                          "adj_success_rate")
    agg["adj_completion_pct_z"] = _z_within_season(agg,
                                                       "adj_completion_pct")
    agg["adj_sack_rate_z"] = _z_within_season(agg, "adj_sack_rate_neg")
    agg["adj_int_rate_z"] = _z_within_season(agg, "adj_int_rate_neg")
    return agg


def _grade(sos: pd.DataFrame) -> pd.DataFrame:
    from lib_qb_gas import compute_qb_gas
    master = pd.read_parquet(REPO / "data" / "league_qb_all_seasons.parquet")
    merged = master.merge(sos, on=["player_id", "season_year"],
                            how="left")
    return compute_qb_gas(merged)


def main():
    print("→ building QB SOS at WEAPONS_WEIGHT 0.00 (def-LOO only)...")
    sos_zero = _build(0.00)
    print("→ building QB SOS at WEAPONS_WEIGHT 1.00...")
    sos_full = _build(1.00)
    print("→ building QB SOS at WEAPONS_WEIGHT 0.75...")
    sos_q75 = _build(0.75)
    print("→ grading...")
    g_zero = _grade(sos_zero)
    g_full = _grade(sos_full)
    g_q75 = _grade(sos_q75)

    def _top10(df, season=2024, min_g=12):
        return (df[(df["season_year"] == season)
                   & (df["games"] >= min_g)]
                .nlargest(10, "gas_score")
                [["player_display_name", "gas_score",
                  "gas_efficiency_grade"]]
                .reset_index(drop=True))

    print()
    print("=" * 90)
    print("QB 2024 — top 10  (efficiency_grade in parens)")
    print("=" * 90)
    z = _top10(g_zero).rename(
        columns={"player_display_name": "no_weapons",
                 "gas_score": "score_0", "gas_efficiency_grade": "eff_0"})
    full = _top10(g_full).rename(
        columns={"player_display_name": "weapons_full",
                 "gas_score": "score_full",
                 "gas_efficiency_grade": "eff_full"})
    q75 = _top10(g_q75).rename(
        columns={"player_display_name": "weapons_q75",
                 "gas_score": "score_q75",
                 "gas_efficiency_grade": "eff_q75"})
    print(pd.concat([z, full, q75], axis=1).to_string())

    # Show how much each canary moved due to weapons adj
    print()
    print("=" * 70)
    print("Weapons-adjustment impact on canaries (2024)")
    print("=" * 70)
    canaries = ["Jared Goff", "Patrick Mahomes", "Joe Burrow",
                "Justin Herbert", "Trevor Lawrence", "Jayden Daniels",
                "Lamar Jackson", "Jalen Hurts", "Brock Purdy",
                "Sam Darnold"]
    rows = []
    for name in canaries:
        z_row = g_zero[(g_zero["season_year"] == 2024)
                        & (g_zero["player_display_name"] == name)]
        f_row = g_full[(g_full["season_year"] == 2024)
                        & (g_full["player_display_name"] == name)]
        q_row = g_q75[(g_q75["season_year"] == 2024)
                       & (g_q75["player_display_name"] == name)]
        if len(z_row) and len(f_row) and len(q_row):
            rows.append({
                "QB": name,
                "score_no_weapons": z_row["gas_score"].iloc[0],
                "score_full": f_row["gas_score"].iloc[0],
                "score_q75": q_row["gas_score"].iloc[0],
                "delta_full": (f_row["gas_score"].iloc[0]
                               - z_row["gas_score"].iloc[0]),
                "delta_q75": (q_row["gas_score"].iloc[0]
                              - z_row["gas_score"].iloc[0]),
            })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

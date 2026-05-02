"""SOS-adjusted WR per-season z-cols — game-level QB + defense LOO.

Output: data/wr_sos_adjusted_z.parquet

Stacked leave-one-out at GAME granularity:
  - Defense LOO at game level: subtract this defense's per-game allowance
    on WR-targeted passes, EXCLUDING this receiver's contributions.
  - QB LOO at game level: subtract this QB's per-game level on
    WR-targeted passes, EXCLUDING this receiver's contributions.

This isolates receiver contribution from both QB quality (Mahomes
inflates his receivers' per-target stats; O'Connell deflates his)
AND defense quality (per the prior season-level fix).

Game-level matters because:
  - Multi-QB seasons (McLaurin: Mariota games vs Daniels games)
  - Within-QB game variance (Mahomes hot/cold day = different baseline)
  - Backup spot starts get graded against THAT game's actual play

Math (per pass play in game G to receiver P from QB Q vs defense D):
  def_baseline_G_excl_P = (D_g_total − P_v_D_g_total) / (D_g_count − P_v_D_g_count)
  qb_baseline_G_excl_P  = (Q_g_total − P_v_Q_g_total) / (Q_g_count − P_v_Q_g_count)

  adj_play = play_stat
             − (def_baseline_G_excl_P − league_mean)
             − (qb_baseline_G_excl_P  − league_mean)

Falls back to season-level baseline if a game's LOO denominator
drops below MIN_BASELINE_PLAYS — protects against tiny-sample noise
in mop-up / spot-start games.

Stats adjusted (per-target receiver leverage):
  yards_per_target  → adj_yards_per_target_z
  epa_per_target    → adj_epa_per_target_z
  success_rate      → adj_success_rate_z
  catch_rate        → adj_catch_rate_z

Skipped (already player-isolated):
  avg_separation, yac_above_exp, target_share, air_yards_share, wopr
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
MIN_BASELINE_PLAYS = 8   # below this, fall back to season-level baseline

# QB-LOO strength. 1.00 = fully back out QB contribution. 0.75 leaves
# receivers with 25% of their QB's credit, which empirically lands on
# more defensible top-of-leaderboard ordering (e.g., Kittle ≥ Kraft in
# 2024) without giving Mahomes/Burrow receivers a free ride.
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

    # NOTE: don't drop on receiving_yards — NaN on incompletions, would
    # silently filter to completions only and turn per-target metrics
    # into per-reception. Fill with 0 instead.
    pbp = pbp.dropna(subset=["receiver_player_id", "season",
                                "defteam", "epa", "passer_player_id",
                                "game_id"])
    pbp = pbp[pbp["play_type"] == "pass"].copy()
    pbp["receiving_yards"] = pbp["receiving_yards"].fillna(0)
    pbp["complete_pass"] = pbp["complete_pass"].fillna(0)
    pbp["season"] = pbp["season"].astype(int)
    print(f"  pass plays w/ receiver: {len(pbp):,}")

    wr_ids = set(rosters[rosters["position"] == "WR"][
        "gsis_id"
    ].dropna().tolist())
    print(f"  WR roster ids: {len(wr_ids):,}")

    wr_pass = pbp[pbp["receiver_player_id"].isin(wr_ids)].copy()
    print(f"  WR-targeted pass plays: {len(wr_pass):,}")

    # League means per season (target population: WR-targeted passes)
    league = wr_pass.groupby("season").agg(
        lg_epa=("epa", "mean"),
        lg_success=("success", "mean"),
        lg_yards=("receiving_yards", "mean"),
        lg_complete=("complete_pass", "mean"),
    ).reset_index()
    print(f"  league seasons: {len(league)}")

    stats = ["epa", "success", "receiving_yards", "complete_pass"]

    # ── GAME-LEVEL DEFENSE TOTALS ──────────────────────────────────
    print("→ computing per-game defense totals...")
    def_g = wr_pass.groupby(["defteam", "season",
                                "game_id"])[stats].agg(
        ["sum", "size"]
    )
    def_g.columns = [f"d_g_{s}" if a == "sum" else "d_g_count"
                       for s, a in def_g.columns]
    # Collapse duplicate count cols
    def_g = def_g.loc[:, ~def_g.columns.duplicated()].reset_index()
    print(f"  defense-game rows: {len(def_g):,}")

    # ── GAME-LEVEL QB TOTALS ───────────────────────────────────────
    print("→ computing per-game QB totals (WR-targets only)...")
    qb_g = wr_pass.groupby(["passer_player_id", "season",
                              "game_id"])[stats].agg(
        ["sum", "size"]
    )
    qb_g.columns = [f"q_g_{s}" if a == "sum" else "q_g_count"
                      for s, a in qb_g.columns]
    qb_g = qb_g.loc[:, ~qb_g.columns.duplicated()].reset_index()
    print(f"  QB-game rows: {len(qb_g):,}")

    # ── PLAYER-VS-DEFENSE-IN-GAME ───────────────────────────────────
    print("→ computing receiver-vs-defense per-game contributions...")
    p_v_d_g = wr_pass.groupby(["receiver_player_id", "defteam",
                                  "season", "game_id"])[stats].agg(
        ["sum", "size"]
    )
    p_v_d_g.columns = [f"p_d_g_{s}" if a == "sum" else "p_d_g_count"
                          for s, a in p_v_d_g.columns]
    p_v_d_g = p_v_d_g.loc[:, ~p_v_d_g.columns.duplicated()].reset_index()
    print(f"  receiver-defense-game rows: {len(p_v_d_g):,}")

    # ── PLAYER-VS-QB-IN-GAME ────────────────────────────────────────
    print("→ computing receiver-vs-QB per-game contributions...")
    p_v_q_g = wr_pass.groupby(["receiver_player_id", "passer_player_id",
                                  "season", "game_id"])[stats].agg(
        ["sum", "size"]
    )
    p_v_q_g.columns = [f"p_q_g_{s}" if a == "sum" else "p_q_g_count"
                          for s, a in p_v_q_g.columns]
    p_v_q_g = p_v_q_g.loc[:, ~p_v_q_g.columns.duplicated()].reset_index()
    print(f"  receiver-QB-game rows: {len(p_v_q_g):,}")

    # ── SEASON-LEVEL FALLBACK BASELINES ─────────────────────────────
    # For thin-sample games we fall back to season totals.
    print("→ computing season-level fallback baselines...")
    def_s = wr_pass.groupby(["defteam", "season"])[stats].agg(
        ["sum", "size"]
    )
    def_s.columns = [f"d_s_{s}" if a == "sum" else "d_s_count"
                       for s, a in def_s.columns]
    def_s = def_s.loc[:, ~def_s.columns.duplicated()].reset_index()
    qb_s = wr_pass.groupby(["passer_player_id", "season"])[stats].agg(
        ["sum", "size"]
    )
    qb_s.columns = [f"q_s_{s}" if a == "sum" else "q_s_count"
                      for s, a in qb_s.columns]
    qb_s = qb_s.loc[:, ~qb_s.columns.duplicated()].reset_index()

    p_v_d_s = wr_pass.groupby(["receiver_player_id", "defteam",
                                  "season"])[stats].agg(["sum", "size"])
    p_v_d_s.columns = [f"p_d_s_{s}" if a == "sum" else "p_d_s_count"
                          for s, a in p_v_d_s.columns]
    p_v_d_s = p_v_d_s.loc[:, ~p_v_d_s.columns.duplicated()].reset_index()
    p_v_q_s = wr_pass.groupby(["receiver_player_id", "passer_player_id",
                                  "season"])[stats].agg(["sum", "size"])
    p_v_q_s.columns = [f"p_q_s_{s}" if a == "sum" else "p_q_s_count"
                          for s, a in p_v_q_s.columns]
    p_v_q_s = p_v_q_s.loc[:, ~p_v_q_s.columns.duplicated()].reset_index()

    # Side-effect WR-defense table (season form, for reuse) — unchanged
    wr_def_path = REPO / "data" / "team_wr_def_quality.parquet"
    wr_def_save = def_s.copy()
    wr_def_save["opp_epa_allowed"] = (wr_def_save["d_s_epa"]
                                          / wr_def_save["d_s_count"])
    wr_def_save["opp_success_allowed"] = (wr_def_save["d_s_success"]
                                              / wr_def_save["d_s_count"])
    wr_def_save["opp_yards_allowed"] = (wr_def_save["d_s_receiving_yards"]
                                            / wr_def_save["d_s_count"])
    wr_def_save["opp_complete_allowed"] = (wr_def_save["d_s_complete_pass"]
                                                / wr_def_save["d_s_count"])
    wr_def_save = wr_def_save.rename(columns={"defteam": "team"})[[
        "team", "season", "opp_epa_allowed", "opp_success_allowed",
        "opp_yards_allowed", "opp_complete_allowed"
    ]]
    wr_def_save.to_parquet(wr_def_path, index=False)
    print(f"  ✓ side-effect: wrote {wr_def_path.relative_to(REPO)}")

    # ── MERGE EVERYTHING INTO PBP ──────────────────────────────────
    print("→ merging baselines onto play-by-play...")
    wr_pass = wr_pass.merge(def_g, on=["defteam", "season", "game_id"],
                              how="left")
    wr_pass = wr_pass.merge(qb_g, on=["passer_player_id", "season",
                                         "game_id"], how="left")
    wr_pass = wr_pass.merge(p_v_d_g, on=["receiver_player_id", "defteam",
                                             "season", "game_id"],
                              how="left")
    wr_pass = wr_pass.merge(p_v_q_g,
                              on=["receiver_player_id", "passer_player_id",
                                  "season", "game_id"], how="left")
    wr_pass = wr_pass.merge(def_s, on=["defteam", "season"], how="left")
    wr_pass = wr_pass.merge(qb_s, on=["passer_player_id", "season"],
                              how="left")
    wr_pass = wr_pass.merge(p_v_d_s, on=["receiver_player_id", "defteam",
                                             "season"], how="left")
    wr_pass = wr_pass.merge(p_v_q_s,
                              on=["receiver_player_id", "passer_player_id",
                                  "season"], how="left")
    wr_pass = wr_pass.merge(league, on="season", how="left")
    print(f"  merged play rows: {len(wr_pass):,}")

    # ── COMPUTE BASELINES PER PLAY ─────────────────────────────────
    # Game-level LOO denominators
    g_def_denom = wr_pass["d_g_count"] - wr_pass["p_d_g_count"]
    g_qb_denom = wr_pass["q_g_count"] - wr_pass["p_q_g_count"]
    # Use game baseline only if denom ≥ MIN_BASELINE_PLAYS, else season fallback
    use_g_def = g_def_denom >= MIN_BASELINE_PLAYS
    use_g_qb = g_qb_denom >= MIN_BASELINE_PLAYS
    # Season-level fallback denominators
    s_def_denom = (wr_pass["d_s_count"] - wr_pass["p_d_s_count"]).clip(lower=1)
    s_qb_denom = (wr_pass["q_s_count"] - wr_pass["p_q_s_count"]).clip(lower=1)
    fallback_g_def = pd.Series(False, index=wr_pass.index)
    fallback_g_qb = pd.Series(False, index=wr_pass.index)
    fallback_g_def[~use_g_def] = True
    fallback_g_qb[~use_g_qb] = True
    print(f"  game-baseline used (defense): "
          f"{use_g_def.sum():,} / {len(wr_pass):,} "
          f"({use_g_def.mean():.1%})")
    print(f"  game-baseline used (QB): "
          f"{use_g_qb.sum():,} / {len(wr_pass):,} "
          f"({use_g_qb.mean():.1%})")

    stat_to_lg = {
        "epa": "lg_epa",
        "success": "lg_success",
        "receiving_yards": "lg_yards",
        "complete_pass": "lg_complete",
    }

    for stat in stats:
        # Baselines
        g_def_base = ((wr_pass[f"d_g_{stat}"] - wr_pass[f"p_d_g_{stat}"])
                      / g_def_denom.clip(lower=1))
        s_def_base = ((wr_pass[f"d_s_{stat}"] - wr_pass[f"p_d_s_{stat}"])
                      / s_def_denom)
        def_base = g_def_base.where(use_g_def, s_def_base)

        g_qb_base = ((wr_pass[f"q_g_{stat}"] - wr_pass[f"p_q_g_{stat}"])
                     / g_qb_denom.clip(lower=1))
        s_qb_base = ((wr_pass[f"q_s_{stat}"] - wr_pass[f"p_q_s_{stat}"])
                     / s_qb_denom)
        qb_base = g_qb_base.where(use_g_qb, s_qb_base)

        lg = wr_pass[stat_to_lg[stat]]
        wr_pass[f"adj_{stat}"] = (wr_pass[stat]
                                    - (def_base - lg)
                                    - QB_LOO_WEIGHT * (qb_base - lg))

    # ── AGGREGATE PER (RECEIVER, SEASON) ───────────────────────────
    grp = wr_pass.groupby(["receiver_player_id", "season"])
    agg = grp.agg(
        targets=("epa", "size"),
        adj_epa_per_target=("adj_epa", "mean"),
        adj_success_rate=("adj_success", "mean"),
        adj_yards_per_target=("adj_receiving_yards", "mean"),
        adj_catch_rate=("adj_complete_pass", "mean"),
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
    print("=== 2024 WR top by game-LOO adj EPA/target ===")
    print(s24[["player_id", "targets",
                "adj_epa_per_target", "adj_epa_per_target_z",
                "adj_yards_per_target", "adj_catch_rate"]
              ].to_string())


if __name__ == "__main__":
    main()

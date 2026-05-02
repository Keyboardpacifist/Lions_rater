"""SOS-adjusted QB per-season z-cols — season-level defense LOO +
weapons-availability adjustment.

Output: data/qb_sos_adjusted_z.parquet

Two adjustments per pass play:

1. **Defense LOO at SEASON level.** Subtract the defense's per-play
   allowance, EXCLUDING this QB's contributions across the season.
   (Game-level LOO is useless for QB — only one QB faces a defense
   per game, so the game-LOO denominator collapses to 0.)

2. **Weapons-availability adjustment.** Credit QBs for performing
   without their top weapons. Method:
     a. Apply defense-LOO to each play first (residual stat).
     b. Regress residual on team_weapons_strength league-wide to
        estimate β (cleaner than raw OLS — defense confound removed).
     c. Per play, subtract: WEAPONS_WEIGHT × β × (w_G − w̄)
        — QBs in low-weapons games get a bonus, high-weapons get a
        discount.
   WEAPONS_WEIGHT = 0.75 mirrors WR/TE LOO weight.

Stats adjusted (the 5 highest-leverage QB stats):
  pass_epa_per_play    (def + weapons)
  pass_success_rate    (def + weapons)
  completion_pct       (def + weapons)
  sack_rate            (def only — weapons effect unclear)
  int_rate             (def only — weapons effect unclear)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DROPBACKS = REPO / "data" / "qb_dropbacks.parquet"
WEAPONS = REPO / "data" / "team_weapons_availability.parquet"
OUT = REPO / "data" / "qb_sos_adjusted_z.parquet"

MIN_DROPBACKS = 100
WEAPONS_WEIGHT = 0.75


def _z_within_season(df: pd.DataFrame, col: str,
                       season_col: str = "season_year") -> pd.Series:
    means = df.groupby(season_col)[col].transform("mean")
    stds = df.groupby(season_col)[col].transform("std").replace(0,
                                                                  np.nan)
    return ((df[col] - means) / stds).fillna(0)


def main() -> None:
    print("→ loading inputs...")
    db = pd.read_parquet(DROPBACKS)
    weapons = pd.read_parquet(WEAPONS)
    print(f"  dropbacks: {len(db):,}")
    print(f"  weapons rows: {len(weapons):,}")

    db = db.dropna(subset=["passer_player_id", "season", "defteam",
                              "epa", "week", "posteam"])
    db["season"] = db["season"].astype(int)
    db["week"] = db["week"].astype(int)
    db["sack_f"] = db["sack"].fillna(0)
    db["int_f"] = db["interception"].fillna(0)
    db["complete_pass"] = db["complete_pass"].fillna(0)
    db["success"] = db["success"].fillna(0)

    # Attach weapons_strength per (team, season, week)
    db = db.merge(
        weapons.rename(columns={"team": "posteam"})[
            ["posteam", "season", "week", "weapons_strength"]],
        on=["posteam", "season", "week"], how="left")
    matched = db["weapons_strength"].notna().sum()
    print(f"  plays w/ weapons data: {matched:,} / {len(db):,} "
          f"({matched/len(db):.0%})")
    league_mean_weapons = db["weapons_strength"].mean()
    db["weapons_strength"] = db["weapons_strength"].fillna(
        league_mean_weapons)
    print(f"  league mean weapons_strength: {league_mean_weapons:.3f}")

    # League per-play means per season
    league = db.groupby("season").agg(
        lg_epa=("epa", "mean"),
        lg_success=("success", "mean"),
        lg_complete=("complete_pass", "mean"),
        lg_sack=("sack_f", "mean"),
        lg_int=("int_f", "mean"),
    ).reset_index()

    # ── SEASON-LEVEL DEFENSE LOO ───────────────────────────────────
    stats_def = ["epa", "success", "complete_pass", "sack_f", "int_f"]
    print("→ computing season-level defense LOO baselines...")
    def_s = db.groupby(["defteam", "season"])[stats_def].agg(
        ["sum", "size"])
    def_s.columns = [f"d_s_{s}" if a == "sum" else "d_s_count"
                       for s, a in def_s.columns]
    def_s = def_s.loc[:, ~def_s.columns.duplicated()].reset_index()
    p_v_d_s = db.groupby(["passer_player_id", "defteam",
                              "season"])[stats_def].agg(["sum", "size"])
    p_v_d_s.columns = [f"p_d_s_{s}" if a == "sum" else "p_d_s_count"
                          for s, a in p_v_d_s.columns]
    p_v_d_s = p_v_d_s.loc[:, ~p_v_d_s.columns.duplicated()].reset_index()

    db = db.merge(def_s, on=["defteam", "season"], how="left")
    db = db.merge(p_v_d_s, on=["passer_player_id", "defteam",
                                   "season"], how="left")
    db = db.merge(league, on="season", how="left")

    s_def_denom = (db["d_s_count"] - db["p_d_s_count"]).clip(lower=1)

    stat_to_lg = {"epa": "lg_epa", "success": "lg_success",
                  "complete_pass": "lg_complete",
                  "sack_f": "lg_sack", "int_f": "lg_int"}

    # First pass: defense-only adjustment (residuals for β estimation)
    for stat in stats_def:
        def_base = ((db[f"d_s_{stat}"] - db[f"p_d_s_{stat}"])
                    / s_def_denom)
        lg = db[stat_to_lg[stat]]
        db[f"def_adj_{stat}"] = db[stat] - (def_base - lg)

    # ── WEAPONS β: HAND-PICKED FOOTBALL PRIORS ─────────────────────
    # Observational β estimation (raw OLS, def-residual, even within-QB
    # fixed effects) consistently produces negative betas, because of
    # a deep confound: when weapons are out, playcallers shift to safer
    # plays (checkdowns, runs) which inflates per-play efficiency for
    # non-QB-skill reasons. Within-QB FE doesn't fix it because the
    # play mix changes inside-QB too.
    #
    # The observational data CANNOT cleanly isolate "QB skill above
    # what his weapons would predict" without play-mix controls
    # (formation, down-distance, depth-of-target) — way out of scope
    # for v1.
    #
    # Instead: use modest positive priors. These encode football
    # intuition that weapons help, at conservative magnitude. A QB
    # in a -0.20 weapons-strength game (ex: Goff w/o LaPorta) gets
    # ≈ +0.0075 EPA/play credit, ~+0.4 EPA/game, ~+6 EPA over a
    # season of similarly-diminished games. Visible but not extreme.
    print("→ using hand-picked weapons β (football priors):")
    betas = {
        "epa":            0.05,
        "success":        0.04,
        "complete_pass":  0.04,
    }
    db["weapons_centered"] = db["weapons_strength"] - league_mean_weapons
    for stat, b in betas.items():
        print(f"  β_{stat}: {b:+.4f}")

    # ── FINAL ADJUSTMENT: def + weapons ────────────────────────────
    weapons_delta = db["weapons_centered"]   # already (w - lg_mean)
    for stat in stats_def:
        adj = db[f"def_adj_{stat}"]
        if stat in ("epa", "success", "complete_pass"):
            adj = adj - WEAPONS_WEIGHT * betas[stat] * weapons_delta
        db[f"adj_{stat}"] = adj

    # ── AGGREGATE PER (QB, SEASON) ─────────────────────────────────
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
    print(f"  qualified QB-seasons: {len(agg)}")

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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    s24 = agg[agg["season_year"] == 2024].nlargest(8,
                                                      "adj_pass_epa_per_play_z")
    print("=== 2024 — def + weapons-adj EPA leaders ===")
    print(s24[["player_id", "dropbacks", "avg_weapons_strength",
                "adj_epa_per_play", "adj_pass_epa_per_play_z",
                "adj_completion_pct"]].to_string())


if __name__ == "__main__":
    main()

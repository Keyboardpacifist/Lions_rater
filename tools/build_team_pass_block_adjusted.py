"""SOS-adjusted team pass-block stats (sack rate + pressure rate),
with QB sack-avoidance credit removed.

Output: data/team_pass_block_adjusted.parquet

Two adjustments applied per pass play:

1. **Defense LOO (per-team).** Subtract the opposing defense's
   typical pressure / sack rate excluding this team's plays vs that
   defense. Captures schedule-strength.

2. **QB sack-avoidance credit.** Some QBs save their OL from sacks
   even when pressured (Allen 1.78, Lamar 1.10 sack-avoided z). Some
   don't despite being mobile (Fields −0.52, Wilson −0.73). Without
   this adjustment, BUF/BAL OL look better than they are because the
   QB bailed them out, while the OL of QBs who hold the ball get
   undeserved credit. We use sack_avoided_rate_z (NOT raw mobility)
   because that's the QB skill specifically relevant to converting
   pressures into non-sacks.

   Method: regress per-play sack outcome on QB's sack_avoided_rate_z
   league-wide → estimate β. Subtract β × (Q_avoid_z − 0) per play
   to net out QB contribution.

Aggregates per (team, season). Lower adj_rate = better OL than its
schedule + QB would predict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DROPBACKS = REPO / "data" / "qb_dropbacks.parquet"
QB_PRESSURE = REPO / "data" / "qb_pressure_clutch_z.parquet"
OUT = REPO / "data" / "team_pass_block_adjusted.parquet"

MIN_DROPBACKS = 200   # team-season threshold


def _z_within_season(df: pd.DataFrame, col: str,
                       season_col: str = "season") -> pd.Series:
    means = df.groupby(season_col)[col].transform("mean")
    stds = df.groupby(season_col)[col].transform("std").replace(0,
                                                                  np.nan)
    return ((df[col] - means) / stds).fillna(0)


def main() -> None:
    print("→ loading qb_dropbacks...")
    db = pd.read_parquet(DROPBACKS)
    db = db.dropna(subset=["posteam", "defteam", "season"])
    db["season"] = db["season"].astype(int)
    db["sack_f"] = db["sack"].fillna(0)
    # was_pressure has 92-100% coverage. Fill missing with column mean
    # (per season) so we don't drop those plays.
    db["pressure_f"] = db.groupby("season")["was_pressure"].transform(
        lambda x: x.fillna(x.mean()))
    db["pressure_f"] = db["pressure_f"].fillna(
        db["was_pressure"].mean())
    print(f"  dropbacks: {len(db):,}")

    # League means per season (centering point)
    league = db.groupby("season").agg(
        lg_sack=("sack_f", "mean"),
        lg_pressure=("pressure_f", "mean"),
    ).reset_index()
    print(f"  league seasons: {len(league)}")

    # ── PER-DEFENSE TOTALS + PER-(TEAM, DEFENSE) TOTALS ────────────
    # opp baseline excluding T = (D_total − T_v_D_total)
    #                          / (D_count − T_v_D_count)
    print("→ computing per-defense + per-(team, defense) totals...")
    def_totals = db.groupby(["defteam", "season"]).agg(
        d_sack=("sack_f", "sum"),
        d_pressure=("pressure_f", "sum"),
        d_count=("sack_f", "size"),
    ).reset_index()
    team_v_def = db.groupby(["posteam", "defteam", "season"]).agg(
        t_v_d_sack=("sack_f", "sum"),
        t_v_d_pressure=("pressure_f", "sum"),
        t_v_d_count=("sack_f", "size"),
    ).reset_index()
    print(f"  defense-season rows: {len(def_totals):,}")
    print(f"  team-vs-def rows: {len(team_v_def):,}")

    db = db.merge(def_totals, on=["defteam", "season"], how="left")
    db = db.merge(team_v_def, on=["posteam", "defteam", "season"],
                    how="left")
    db = db.merge(league, on="season", how="left")

    # ── ATTACH QB SACK-AVOIDANCE Z PER PLAY ────────────────────────
    print("→ attaching QB sack-avoidance z-cols...")
    qb_pc = pd.read_parquet(QB_PRESSURE)[
        ["player_id", "season_year", "sack_avoided_rate_z"]
    ].rename(columns={"player_id": "passer_player_id",
                          "season_year": "season"})
    db = db.merge(qb_pc, on=["passer_player_id", "season"], how="left")
    qb_avoid_z = db["sack_avoided_rate_z"].fillna(0)
    matched = db["sack_avoided_rate_z"].notna().sum()
    print(f"  plays w/ QB sack-avoidance z: {matched:,}/{len(db):,} "
          f"({matched/len(db):.0%})")

    denom = (db["d_count"] - db["t_v_d_count"]).clip(lower=1)
    base_sack = (db["d_sack"] - db["t_v_d_sack"]) / denom
    base_pressure = (db["d_pressure"] - db["t_v_d_pressure"]) / denom

    # ── ESTIMATE QB SACK-AVOIDANCE β ───────────────────────────────
    # On defense-residual sack outcomes, regress on QB sack_avoided z.
    # β should be NEGATIVE: high z = QB avoids sacks → fewer sacks.
    db["def_adj_sack"] = db["sack_f"] - (base_sack - db["lg_sack"])
    db["def_adj_pressure"] = (db["pressure_f"]
                                - (base_pressure - db["lg_pressure"]))
    var_q = (qb_avoid_z ** 2).mean()
    cov_sack = ((db["def_adj_sack"] - db["def_adj_sack"].mean())
                * qb_avoid_z).mean()
    cov_press = ((db["def_adj_pressure"]
                  - db["def_adj_pressure"].mean())
                 * qb_avoid_z).mean()
    beta_sack_qb = cov_sack / var_q if var_q > 0 else 0.0
    beta_press_qb = cov_press / var_q if var_q > 0 else 0.0
    print(f"  β_sack on QB-sack-avoidance:     {beta_sack_qb:+.4f}")
    print(f"  β_pressure on QB-sack-avoidance: {beta_press_qb:+.4f}")

    # Subtract QB contribution. Negative β + positive QB-z → adj
    # is HIGHER (OL gets less credit). For pressure, the β may be
    # ~0 since sack-avoidance is post-pressure — that's fine.
    db["adj_sack"] = (db["def_adj_sack"]
                        - beta_sack_qb * qb_avoid_z)
    db["adj_pressure"] = (db["def_adj_pressure"]
                            - beta_press_qb * qb_avoid_z)

    # ── AGGREGATE PER (TEAM, SEASON) ───────────────────────────────
    grp = db.groupby(["posteam", "season"])
    agg = grp.agg(
        dropbacks=("sack_f", "size"),
        adj_sack_rate=("adj_sack", "mean"),
        adj_pressure_rate=("adj_pressure", "mean"),
    ).reset_index()
    agg = agg[agg["dropbacks"] >= MIN_DROPBACKS].copy()
    print(f"  qualifying team-seasons: {len(agg)}")

    agg = agg.rename(columns={"posteam": "team"})

    # Negative-direction → flip so high z = good OL
    agg["adj_sack_neg"] = -agg["adj_sack_rate"]
    agg["adj_pressure_neg"] = -agg["adj_pressure_rate"]
    agg["adj_team_sack_rate_z"] = _z_within_season(agg, "adj_sack_neg")
    agg["adj_team_pressure_rate_z"] = _z_within_season(agg,
                                                          "adj_pressure_neg")

    out = agg[["team", "season", "dropbacks",
                "adj_sack_rate", "adj_pressure_rate",
                "adj_team_sack_rate_z", "adj_team_pressure_rate_z"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== 2024 best OL by adj_team_sack_rate_z ===")
    s24 = out[out["season"] == 2024].nlargest(8,
                                                  "adj_team_sack_rate_z")
    print(s24.to_string(index=False))
    print()
    print("=== 2024 worst OL by adj_team_sack_rate_z ===")
    s24 = out[out["season"] == 2024].nsmallest(8,
                                                   "adj_team_sack_rate_z")
    print(s24.to_string(index=False))
    print()
    print("=== Lions/Eagles/Chiefs/Ravens 2024 ===")
    focus = out[(out["season"] == 2024)
                 & out["team"].isin(["DET", "PHI", "KC", "BAL", "BUF"])]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()

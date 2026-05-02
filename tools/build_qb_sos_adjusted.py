"""SOS-adjusted QB per-season z-cols.

Output: data/qb_sos_adjusted_z.parquet

Per-dropback opponent adjustment: for each pass attempt, subtract
the opponent defense's typical allowance from the QB's actual
performance, recentered on league mean. Then aggregate per
(player, season) and z-score within season.

Stats adjusted (the 5 highest-leverage QB stats):
  pass_epa_per_play  →  adj_pass_epa_per_play_z
  pass_success_rate  →  adj_pass_success_rate_z
  completion_pct     →  adj_completion_pct_z
  sack_rate          →  adj_sack_rate_z
  int_rate           →  adj_int_rate_z

These slot into the QB GAS spec to replace the unadjusted versions.
Volume stats (yards/g, TDs/g) and rushing stats keep their original
z-cols for v1 — schedule-strength matters less for those.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DROPBACKS = REPO / "data" / "qb_dropbacks.parquet"
TEAM_DEF = REPO / "data" / "team_pass_def_quality.parquet"
OUT = REPO / "data" / "qb_sos_adjusted_z.parquet"

MIN_DROPBACKS = 100   # same threshold as the unadjusted version


def _z_within_season(df: pd.DataFrame, col: str) -> pd.Series:
    """Z-score within each season's QB population. Higher is better
    for all output cols (we sign-flip negative-direction stats so
    callers always treat z>0 as good)."""
    means = df.groupby("season_year")[col].transform("mean")
    stds = df.groupby("season_year")[col].transform("std").replace(0, np.nan)
    return ((df[col] - means) / stds).fillna(0)


def main() -> None:
    print("→ loading inputs...")
    db = pd.read_parquet(DROPBACKS)
    td = pd.read_parquet(TEAM_DEF)
    print(f"  dropbacks: {len(db):,}")
    print(f"  team-def rows: {len(td):,}")

    db = db.dropna(subset=["passer_player_id", "season",
                              "defteam", "epa"])
    db["season"] = db["season"].astype(int)

    # League averages per season (centering point)
    league = db.groupby("season").agg(
        lg_epa=("epa", "mean"),
        lg_success=("success", "mean"),
        lg_complete=("complete_pass", "mean"),
        lg_sack=("sack", "mean"),
        lg_int=("interception", "mean"),
    ).reset_index()
    print(f"  league seasons: {len(league)}")

    # Opp pass-defense allowance per (team, season).
    # NOTE: completion_pct_allowed, sack_rate_forced are 0-1 scaled
    # in the team_def file. pass_epa_allowed_per_play is the EPA value.
    td_lite = td[["team", "season", "pass_epa_allowed_per_play",
                   "completion_pct_allowed", "sack_rate_forced"]].copy()

    # Compute per-team per-season opp success_rate_allowed and
    # int_rate_forced from raw db (team_def doesn't have these).
    opp_aux = (db.groupby(["defteam", "season"])
               .agg(opp_success_allowed=("success", "mean"),
                    opp_int_forced=("interception", "mean"))
               .reset_index()
               .rename(columns={"defteam": "team"}))
    opp = td_lite.merge(opp_aux, on=["team", "season"], how="outer")
    print(f"  opp def rows: {len(opp)}")

    # Attach opp + league baselines to each dropback
    db = db.merge(opp.rename(columns={"team": "defteam"}),
                   on=["defteam", "season"], how="left")
    db = db.merge(league, on="season", how="left")

    # Per-dropback adjusted values:
    #   adj = actual - opp_allowance + league_mean
    # so the centered scale is league mean. A QB carving a top-5 D
    # at +0.05 EPA/play (when that D allows -0.05 per play league-
    # wide) gets credited more than the same +0.05 against a soft D.
    db["adj_epa"] = (db["epa"]
                      - db["pass_epa_allowed_per_play"]
                      + db["lg_epa"])
    db["adj_success"] = (db["success"]
                          - db["opp_success_allowed"]
                          + db["lg_success"])
    db["adj_complete"] = (db["complete_pass"]
                           - db["completion_pct_allowed"]
                           + db["lg_complete"])
    # sacks/INTs: opp's sack_rate_forced is "% they FORCE". For QB
    # POV, lower opp sack rate forced = easier defense, so adj_sack
    # = actual_sack - opp_forced + league_mean keeps the convention.
    db["adj_sack"] = (db["sack"].fillna(0)
                       - db["sack_rate_forced"].fillna(db["lg_sack"])
                       + db["lg_sack"])
    db["adj_int"] = (db["interception"].fillna(0)
                      - db["opp_int_forced"].fillna(db["lg_int"])
                      + db["lg_int"])

    # Aggregate per (passer, season)
    grp = db.groupby(["passer_player_id", "season"])
    agg = grp.agg(
        dropbacks=("epa", "size"),
        adj_epa_per_play=("adj_epa", "mean"),
        adj_success_rate=("adj_success", "mean"),
        adj_completion_pct=("adj_complete", "mean"),
        adj_sack_rate=("adj_sack", "mean"),
        adj_int_rate=("adj_int", "mean"),
    ).reset_index()
    agg = agg[agg["dropbacks"] >= MIN_DROPBACKS].copy()
    print(f"  qualified player-seasons: {len(agg)}")

    # Rename for join with master + flip sign on negative-direction stats
    # so HIGHER z always = better.
    agg = agg.rename(columns={
        "passer_player_id": "player_id",
        "season": "season_year",
    })

    # Negative-direction stats — flip so high z = better
    # (low sack rate = good, low INT rate = good)
    agg["adj_sack_rate_neg"] = -agg["adj_sack_rate"]
    agg["adj_int_rate_neg"]  = -agg["adj_int_rate"]

    # Z-score within season
    agg["adj_pass_epa_per_play_z"] = _z_within_season(agg, "adj_epa_per_play")
    agg["adj_pass_success_rate_z"] = _z_within_season(agg, "adj_success_rate")
    agg["adj_completion_pct_z"]    = _z_within_season(agg, "adj_completion_pct")
    agg["adj_sack_rate_z"]         = _z_within_season(agg, "adj_sack_rate_neg")
    agg["adj_int_rate_z"]          = _z_within_season(agg, "adj_int_rate_neg")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    # Spot check: 2024 SOS-adjusted EPA leaders
    s24 = agg[agg["season_year"] == 2024].nlargest(8,
                                                      "adj_pass_epa_per_play_z")
    print("=== 2024 — SOS-adjusted EPA/play leaders ===")
    print(s24[["player_id", "dropbacks",
                "adj_epa_per_play",
                "adj_pass_epa_per_play_z",
                "adj_completion_pct",
                "adj_completion_pct_z"]].to_string())


if __name__ == "__main__":
    main()

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

    # ── PER-PLAYER LEAVE-ONE-OUT SOS ──────────────────────────────
    # opp baseline = (D_total − player_v_D_total) / (D_count − player_v_D_count)
    # — the right way to break the chicken-and-egg.
    print("→ computing per-defense + per-player season totals...")
    db["sack_f"] = db["sack"].fillna(0)
    db["int_f"] = db["interception"].fillna(0)

    def_totals = db.groupby(["defteam", "season"]).agg(
        d_epa_total=("epa", "sum"),
        d_success_total=("success", "sum"),
        d_complete_total=("complete_pass", "sum"),
        d_sack_total=("sack_f", "sum"),
        d_int_total=("int_f", "sum"),
        d_count=("epa", "size"),
    ).reset_index()
    player_v_def = db.groupby(
        ["passer_player_id", "defteam", "season"]
    ).agg(
        p_v_d_epa=("epa", "sum"),
        p_v_d_success=("success", "sum"),
        p_v_d_complete=("complete_pass", "sum"),
        p_v_d_sack=("sack_f", "sum"),
        p_v_d_int=("int_f", "sum"),
        p_v_d_count=("epa", "size"),
    ).reset_index()
    print(f"  team-season pass-def rows: {len(def_totals)}")
    print(f"  player-vs-def rows: {len(player_v_def):,}")

    db = db.merge(def_totals, on=["defteam", "season"], how="left")
    db = db.merge(player_v_def,
                    on=["passer_player_id", "defteam", "season"],
                    how="left")
    db = db.merge(league, on="season", how="left")

    denom = (db["d_count"] - db["p_v_d_count"]).clip(lower=1)
    base_epa = ((db["d_epa_total"] - db["p_v_d_epa"]) / denom)
    base_success = ((db["d_success_total"] - db["p_v_d_success"]) / denom)
    base_complete = ((db["d_complete_total"] - db["p_v_d_complete"]) / denom)
    base_sack = ((db["d_sack_total"] - db["p_v_d_sack"]) / denom)
    base_int = ((db["d_int_total"] - db["p_v_d_int"]) / denom)

    db["adj_epa"] = db["epa"] - base_epa + db["lg_epa"]
    db["adj_success"] = db["success"] - base_success + db["lg_success"]
    db["adj_complete"] = (db["complete_pass"] - base_complete
                            + db["lg_complete"])
    db["adj_sack"] = db["sack_f"] - base_sack + db["lg_sack"]
    db["adj_int"] = db["int_f"] - base_int + db["lg_int"]

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

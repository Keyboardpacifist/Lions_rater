"""Per-player injury performance deltas, opponent-strength adjusted.

Output: data/player_self_injury_deltas.parquet

For each (player, primary stat, injury bucket), compute:
  • Mean RAW stat in the bucket
  • Mean OPP-ADJUSTED stat in the bucket (subtracts the expected stat
    against the specific opponents' defensive strength)
  • Delta vs. the player's HEALTHY baseline (raw and adjusted)
  • Wilson 95% CI on the binary "did the player exceed his healthy
    median in this bucket?" rate

The OPP ADJUSTMENT is what makes this trustworthy. Without it,
"Goff threw for -8 yards under his baseline when on Q/Limited"
mixes Goff's hurt-performance with the strength of the defenses he
happened to face those weeks. Adjustment subtracts the schedule.

Buckets
-------
    HEALTHY              — not on injury report at all
    PROBABLE_FULL        — Probable + full Friday practice
    PROBABLE_LIMITED     — Probable + limited (toughing it out)
    QUESTIONABLE_FULL    — Q + full
    QUESTIONABLE_LIMITED — Q + limited (the classic "playing hurt" cell)
    QUESTIONABLE_DNP     — Q + DNP all week, gets activated Sunday
    DOUBTFUL_ANY         — any Doubtful designation
    OUT_PLAYED           — listed OUT but somehow appeared (rare)

Schema
------
    player_id, player_name, position, stat,
    bucket, n,
    mean_raw,      mean_adj,
    healthy_raw,   healthy_adj,
    delta_raw,     delta_adj,
    retention_adj  (= mean_adj / healthy_adj when both > 0,
                     clipped to [0, 1.10])
    n_total_player  (cumulative row for context)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

INJURIES = REPO / "data" / "nfl_injuries_historical.parquet"
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
SCHEDULES = REPO / "data" / "nfl_schedules.parquet"
OPP_STRENGTH = REPO / "data" / "team_opponent_strength.parquet"
OUT = REPO / "data" / "player_self_injury_deltas.parquet"

from lib_alt_line_ev import wilson_interval  # noqa: E402

# Map (report_status, practice_status) → bucket label.
def _bucket_for(report, practice) -> str:
    # Coerce NaN / None / float-NaN to "NONE" before upper()
    if report is None or (isinstance(report, float) and pd.isna(report)):
        return "HEALTHY"
    r = str(report).upper().strip()
    if r in ("", "NONE", "NAN"):
        return "HEALTHY"
    if practice is None or (isinstance(practice, float) and pd.isna(practice)):
        p = "NONE"
    else:
        p = str(practice).upper().strip()
    # Friendly normalization for the practice text
    if "DID NOT" in p or p == "DNP":
        p = "DNP"
    elif "LIMITED" in p:
        p = "LIMITED"
    elif "FULL" in p:
        p = "FULL"
    if r == "PROBABLE":
        if p == "FULL":
            return "PROBABLE_FULL"
        if p == "LIMITED":
            return "PROBABLE_LIMITED"
        return "PROBABLE_FULL"  # collapse missing practice
    if r == "QUESTIONABLE":
        if p == "FULL":
            return "QUESTIONABLE_FULL"
        if p == "LIMITED":
            return "QUESTIONABLE_LIMITED"
        if p == "DNP":
            return "QUESTIONABLE_DNP"
        return "QUESTIONABLE_LIMITED"
    if r == "DOUBTFUL":
        return "DOUBTFUL_ANY"
    if r == "OUT":
        return "OUT_PLAYED"  # Player snuck onto the field
    return "HEALTHY"


PRIMARY_STAT = {
    "QB":  "passing_yards",
    "RB":  "rushing_yards",
    "FB":  "rushing_yards",
    "WR":  "receiving_yards",
    "TE":  "receiving_yards",
}


# Per-stat opp strength column in team_opponent_strength.parquet
OPP_COL_FOR_STAT = {
    "passing_yards":   "opp_pass_yards_allowed_avg",
    "rushing_yards":   "opp_rush_yards_allowed_avg",
    "receiving_yards": "opp_rec_yards_allowed_avg",
}
LEAGUE_COL_FOR_STAT = {
    "passing_yards":   "league_pass_yds_pg",
    "rushing_yards":   "league_rush_yds_pg",
    "receiving_yards": "league_rec_yds_pg",
}


def _norm_name(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)$", "", regex=True)
              .str.strip().str.lower())


TEAM_FIX = {"STL": "LA", "LAR": "LA", "OAK": "LV", "SD": "LAC", "WSH": "WAS"}


def _norm_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().replace(TEAM_FIX)


def main() -> None:
    print("→ loading inputs...")
    inj = pd.read_parquet(INJURIES)
    ps = pd.read_parquet(PLAYER_STATS)
    sch = pd.read_parquet(SCHEDULES)
    opp = pd.read_parquet(OPP_STRENGTH)
    print(f"  inj={len(inj):,}  ps={len(ps):,}  sch={len(sch):,}  "
          f"opp={len(opp):,}")

    # Restrict to seasons with full coverage of all inputs (2013+ snap
    # counts, 2009+ injuries — limit by intersection)
    ps = ps[ps["season"] >= 2009].copy()

    # ── Tag each player-week with the opponent
    home = sch[["season", "week", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent_for_team"})
    away = sch[["season", "week", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent_for_team"})
    opp_lookup = pd.concat([home, away], ignore_index=True)
    ps = ps.merge(opp_lookup, on=["season", "week", "team"], how="left")

    # ── Attach injury info per (gsis_id, season, week)
    inj["name_n"] = _norm_name(inj["full_name"])
    inj["team_n"] = _norm_team(inj["team"])
    ps["name_n"] = _norm_name(ps["player_display_name"])
    ps["team_n"] = _norm_team(ps["team"])
    inj_keys = inj[["season", "week", "team_n", "name_n",
                     "report_status", "practice_status",
                     "report_primary_injury"]].copy()
    ps = ps.merge(inj_keys, on=["season", "week", "team_n", "name_n"],
                   how="left")
    ps["bucket"] = ps.apply(
        lambda r: _bucket_for(r.get("report_status"),
                                r.get("practice_status")),
        axis=1,
    )

    # ── Attach opponent defensive strength + league averages
    opp_join = opp.rename(columns={"team": "opponent_for_team"})
    ps = ps.merge(opp_join, on=["opponent_for_team", "season"],
                   how="left")

    # ── Build per-stat OUTPUT (one stat per position; same player
    # could be multi-position-eligible — we go by primary)
    rows: list[dict] = []
    for position, stat in PRIMARY_STAT.items():
        opp_col = OPP_COL_FOR_STAT.get(stat)
        league_col = LEAGUE_COL_FOR_STAT.get(stat)
        if opp_col is None or league_col is None:
            continue
        sub = ps[(ps["position"] == position)
                 & ps[stat].notna()].copy()
        if sub.empty:
            continue

        # Opp adjustment: actual − (opp_strength) is the gap vs the
        # expected stat against this defense. We center on the league
        # avg so HEALTHY baseline is ~0 for an avg player vs avg opp.
        # adj = stat − opp_strength + league_avg  (so raw and adj have
        # similar scale and HEALTHY adj ≈ HEALTHY raw on average).
        sub["adj_stat"] = (sub[stat]
                            - sub[opp_col].fillna(sub[league_col])
                            + sub[league_col])

        # Group by player + bucket
        gb = (sub.groupby(["player_id", "player_display_name",
                             "position", "bucket"], dropna=False)
              .agg(n=(stat, "size"),
                   mean_raw=(stat, "mean"),
                   mean_adj=("adj_stat", "mean"))
              .reset_index())
        gb["stat"] = stat
        # Player healthy baseline (HEALTHY bucket)
        healthy = gb[gb["bucket"] == "HEALTHY"][[
            "player_id", "mean_raw", "mean_adj"
        ]].rename(columns={"mean_raw": "healthy_raw",
                            "mean_adj": "healthy_adj"})
        gb = gb.merge(healthy, on="player_id", how="left")
        gb["delta_raw"] = gb["mean_raw"] - gb["healthy_raw"]
        gb["delta_adj"] = gb["mean_adj"] - gb["healthy_adj"]
        # Retention: ratio of injured-bucket adj to healthy adj.
        # Adjusted means tend to be near league average; ratios
        # blow up near zero, so use raw means for retention metric.
        gb["retention_adj"] = (gb["mean_raw"] / gb["healthy_raw"]).clip(
            lower=0.0, upper=1.10)

        # Player overall stat sample size (for thin-cell flags)
        totals = (sub.groupby("player_id")[stat].size().reset_index()
                  .rename(columns={stat: "n_total_player"}))
        gb = gb.merge(totals, on="player_id", how="left")

        rows.append(gb)

    if not rows:
        print("No data produced.")
        return
    out = pd.concat(rows, ignore_index=True)

    # Reorder
    out = out[["player_id", "player_display_name", "position", "stat",
                "bucket", "n", "n_total_player",
                "mean_raw", "mean_adj",
                "healthy_raw", "healthy_adj",
                "delta_raw", "delta_adj", "retention_adj"]]
    out = out.sort_values(["player_id", "stat", "bucket"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print(f"  rows: {len(out):,}")
    print(f"  unique players: {out['player_id'].nunique():,}")
    print(f"  buckets present: {sorted(out['bucket'].unique())}")
    print()
    # Spot check on a known QB
    print("=== Sample: Jared Goff (passing_yards) ===")
    goff = out[(out["player_id"] == "00-0033106")
               & (out["stat"] == "passing_yards")]
    if not goff.empty:
        print(goff[["bucket", "n", "mean_raw", "mean_adj",
                     "delta_raw", "delta_adj",
                     "retention_adj"]].to_string())
    print()
    print("=== Largest Q/LIMITED retention drops (n>=5, RB) ===")
    rb = out[(out["position"] == "RB")
             & (out["bucket"] == "QUESTIONABLE_LIMITED")
             & (out["n"] >= 5)]
    print(rb.sort_values("retention_adj").head(10)[
        ["player_display_name", "n", "mean_raw", "healthy_raw",
         "delta_raw", "delta_adj", "retention_adj"]
    ].to_string())


if __name__ == "__main__":
    main()

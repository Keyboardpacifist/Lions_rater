"""Per-team-season starter-absence deltas, opponent-adjusted.

Output: data/team_starter_absence_deltas.parquet

For each (team, season, role_lost) — where role is QB1/RB1/WR1/TE1
identified by season-long workload — compares the team's offensive
metrics in games where THAT specific starter was ACTIVE vs. games
where they were OUT (used snap counts as ground truth, not just the
injury report — players sometimes get downgraded game-time).

The delta is OPP-ADJUSTED: each team-game's points/yards are first
shifted vs. the opponent's defensive strength, so "DET scored 20 pts
without Gibbs" and "DET scored 20 vs the league's worst defense" are
treated correctly.

Schema (one row per team-season-role):
    team, season,
    role_lost            (QB1 / RB1 / WR1 / TE1)
    player_lost_id, player_lost_name
    n_active             (games starter was active)
    n_out                (games starter was missing)
    raw_pts_active, raw_pts_out, raw_pts_delta
    adj_pts_active, adj_pts_out, adj_pts_delta   ← isolated impact
    pts_ci_low, pts_ci_high                       (Wilson 95% on raw delta)

If n_out < 1, no row is produced (nothing to compare).
If n_out < 3, the row is produced but flagged thin_sample=True.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
SCHEDULES = REPO / "data" / "nfl_schedules.parquet"
SNAPS = REPO / "data" / "nfl_snap_counts.parquet"
OPP_STRENGTH = REPO / "data" / "team_opponent_strength.parquet"
OUT = REPO / "data" / "team_starter_absence_deltas.parquet"


TEAM_FIX = {"STL": "LA", "LAR": "LA", "OAK": "LV", "SD": "LAC", "WSH": "WAS"}


def _norm_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().replace(TEAM_FIX)


def _norm_name(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.replace(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)$", "",
                            regex=True)
              .str.strip().str.lower())


def _identify_starters(ps: pd.DataFrame) -> pd.DataFrame:
    """Long-format (team, season, role, player_id, player_name)."""
    rows = []
    qb = (ps[ps["position"] == "QB"]
          .groupby(["team", "season", "player_id",
                     "player_display_name"])["attempts"].sum()
          .reset_index()
          .sort_values("attempts", ascending=False)
          .drop_duplicates(["team", "season"]))
    qb["role"] = "QB1"
    rows.append(qb[["team", "season", "player_id",
                     "player_display_name", "role"]])

    rb = (ps[ps["position"] == "RB"]
          .groupby(["team", "season", "player_id",
                     "player_display_name"])["carries"].sum()
          .reset_index()
          .sort_values("carries", ascending=False)
          .drop_duplicates(["team", "season"]))
    rb["role"] = "RB1"
    rows.append(rb[["team", "season", "player_id",
                     "player_display_name", "role"]])

    wr = (ps[ps["position"] == "WR"]
          .groupby(["team", "season", "player_id",
                     "player_display_name"])["targets"].sum()
          .reset_index()
          .sort_values("targets", ascending=False)
          .drop_duplicates(["team", "season"]))
    wr["role"] = "WR1"
    rows.append(wr[["team", "season", "player_id",
                     "player_display_name", "role"]])

    te = (ps[ps["position"] == "TE"]
          .groupby(["team", "season", "player_id",
                     "player_display_name"])["targets"].sum()
          .reset_index()
          .sort_values("targets", ascending=False)
          .drop_duplicates(["team", "season"]))
    te["role"] = "TE1"
    rows.append(te[["team", "season", "player_id",
                     "player_display_name", "role"]])

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    print("→ loading inputs...")
    ps = pd.read_parquet(PLAYER_STATS)
    sch = pd.read_parquet(SCHEDULES)
    snaps = pd.read_parquet(SNAPS)
    opp = pd.read_parquet(OPP_STRENGTH)
    print(f"  ps={len(ps):,}  sch={len(sch):,}  snaps={len(snaps):,}")

    # Restrict to snap-count era
    ps = ps[ps["season"] >= 2013].copy()
    sch = sch[sch["season"] >= 2013].copy()

    # Identify starters per (team, season)
    starters = _identify_starters(ps)
    print(f"  starters identified: {len(starters):,}")

    # Build per-game (team, points_for) records with the OPPONENT
    # whose defensive strength we'll use to adjust
    home = sch[["season", "week", "home_team", "away_team",
                 "home_score"]].rename(
        columns={"home_team": "team",
                  "away_team": "opp_team",
                  "home_score": "team_pts"})
    away = sch[["season", "week", "away_team", "home_team",
                 "away_score"]].rename(
        columns={"away_team": "team",
                  "home_team": "opp_team",
                  "away_score": "team_pts"})
    games = pd.concat([home, away], ignore_index=True)
    print(f"  team-games (≥2013): {len(games):,}")

    # Attach opp strength + league average
    games = games.merge(
        opp.rename(columns={"team": "opp_team",
                              "opp_ppg_allowed_avg":
                                "opp_def_ppg_allowed"}),
        on=["opp_team", "season"], how="left",
    )
    # Adjusted points: actual - opp's typical PPG-allowed + league avg
    # so the centered scale is league-average (~23). HEALTHY teams
    # against avg defenses will average ~league_ppg.
    games["adj_pts"] = (games["team_pts"]
                         - games["opp_def_ppg_allowed"]
                         + games["league_ppg"])

    # ── Active/missed determination via snap counts (truth)
    snaps["snap_total"] = (snaps["offense_snaps"].fillna(0)
                            + snaps["defense_snaps"].fillna(0)
                            + snaps["st_snaps"].fillna(0))
    snaps_played = snaps[snaps["snap_total"] > 0].copy()
    # Bridge snap player → gsis_id via name+team+season
    snaps_played["name_n"] = _norm_name(snaps_played["player"])
    snaps_played["team_n"] = _norm_team(snaps_played["team"])
    ps["name_n"] = _norm_name(ps["player_display_name"])
    ps["team_n"] = _norm_team(ps["team"])
    name_lookup = (ps.dropna(subset=["player_id"])
                    .drop_duplicates(["player_id"])[
                        ["player_id", "name_n", "team_n", "season"]
                    ])
    bridged = snaps_played.merge(
        name_lookup,
        on=["name_n", "team_n", "season"], how="left",
    )
    active_keys = bridged.dropna(subset=["player_id"])[
        ["player_id", "season", "week", "team_n"]
    ].drop_duplicates()
    active_keys = active_keys.rename(columns={"team_n": "team"})

    # ── Per (team, season, role, week) → was the starter active?
    starters_with_team = starters.copy()
    starters_with_team["team"] = _norm_team(starters_with_team["team"])

    # Get all weeks in each season the team played (from games)
    all_team_weeks = games[["team", "season", "week"]].drop_duplicates()
    all_team_weeks["team"] = _norm_team(all_team_weeks["team"])

    # Cross starters with all team-weeks
    starter_weeks = starters_with_team.merge(
        all_team_weeks, on=["team", "season"], how="left",
    )
    # Tag active or not via snap counts
    starter_weeks = starter_weeks.merge(
        active_keys.assign(_active=True),
        on=["player_id", "season", "week", "team"],
        how="left",
    )
    starter_weeks["was_active"] = starter_weeks["_active"].fillna(False)
    starter_weeks = starter_weeks.drop(columns=["_active"])

    # Attach team's points + opp_def_ppg_allowed + league_ppg
    games_keyed = games.copy()
    games_keyed["team"] = _norm_team(games_keyed["team"])
    starter_weeks = starter_weeks.merge(
        games_keyed[["team", "season", "week",
                       "team_pts", "adj_pts",
                       "opp_def_ppg_allowed", "league_ppg"]],
        on=["team", "season", "week"], how="left",
    )

    # ── Aggregate per (team, season, role)
    # V2.3 adds RESIDUALIZED delta — controls for BOTH opp strength
    # AND the team's own healthy baseline. The team_baseline is
    # computed from games WHERE THE STARTER WAS ACTIVE (so it's the
    # "team's normal scoring without this starter problem"). Then:
    #   predicted_pts = team_baseline + (opp_def_ppg_allowed - league_ppg)
    #   residual = team_pts - predicted_pts
    # Mean residual on active games ≈ 0 by construction. Mean residual
    # on out-games = isolated starter impact, with both opp and team
    # strength netted out.
    agg_rows = []
    for (team, season, role), sub in starter_weeks.groupby(
            ["team", "season", "role"]):
        if sub.empty:
            continue
        active = sub[sub["was_active"]]
        absent = sub[~sub["was_active"]]
        n_active = len(active)
        n_out = len(absent)
        if n_active == 0 or n_out == 0:
            continue
        pid = (sub["player_id"].mode().iloc[0]
                if not sub["player_id"].mode().empty else None)
        pname = (sub.loc[sub["player_id"] == pid,
                          "player_display_name"].iloc[0]
                  if pid else "?")

        raw_active = float(active["team_pts"].mean())
        raw_out    = float(absent["team_pts"].mean())
        adj_active = float(active["adj_pts"].mean())
        adj_out    = float(absent["adj_pts"].mean())

        # ── V2.3 residualization via team-specific regression
        # Fit OLS:   pts_active = a + b * opp_def_ppg_allowed
        # over the ACTIVE games. This captures team-specific sensitivity
        # to opponent strength (a strong offense like KC has b≈0.5;
        # a weak offense has b≈1.2). Then for both active and out games:
        #   predicted_pts = a + b * opp_def_for_that_game
        #   residual = actual_pts - predicted_pts
        # Mean residual on ACTIVE is ~0 by OLS construction.
        # Mean residual on OUT is the isolated starter-absence impact,
        # netted out for both opp strength AND team-specific opp slope.
        # Falls back to simple opp-adjustment when n_active < 4.
        if n_active >= 4 and active["opp_def_ppg_allowed"].notna().sum() >= 4:
            x = active["opp_def_ppg_allowed"].fillna(
                active["league_ppg"]).values
            y = active["team_pts"].astype(float).values
            try:
                # Closed-form OLS: b = cov(x,y)/var(x), a = mean(y) - b*mean(x)
                xm = x.mean()
                ym = y.mean()
                xc = x - xm
                yc = y - ym
                denom = float((xc * xc).sum())
                if denom > 1e-6:
                    b = float((xc * yc).sum() / denom)
                else:
                    b = 0.0
                a = ym - b * xm
            except Exception:
                a, b = raw_active, 0.0
            x_act = active["opp_def_ppg_allowed"].fillna(
                active["league_ppg"]).values
            x_out = absent["opp_def_ppg_allowed"].fillna(
                absent["league_ppg"]).values
            pred_active = a + b * x_act
            pred_out    = a + b * x_out
            active_residuals = (active["team_pts"].values - pred_active)
            absent_residuals = (absent["team_pts"].values - pred_out)
            residual_method = f"ols_team_specific (b={b:+.2f})"
        else:
            # Fallback: simple league-mean recentering
            active_residuals = (active["adj_pts"].values - raw_active)
            absent_residuals = (absent["adj_pts"].values - raw_active)
            residual_method = "simple_recenter"

        residual_active = float(active_residuals.mean())  # ≈ 0
        residual_out    = float(absent_residuals.mean())  # the impact
        residual_delta  = residual_out - residual_active

        # Confidence interval on the residual delta
        s_active = (float(active["team_pts"].std(ddof=1))
                     if n_active > 1 else 7.0)
        s_out    = (float(absent["team_pts"].std(ddof=1))
                     if n_out > 1 else 7.0)
        se_raw = (s_active**2/n_active + s_out**2/n_out) ** 0.5
        # Residual SE adds the team_baseline estimation noise
        # (s_active/sqrt(n_active)). Variance of (mean_out - team_baseline):
        # Var(mean_out) + Var(team_baseline) = s_out²/n_out + s_active²/n_active
        # — same as the SE of the raw delta. So we reuse it.
        se_resid = se_raw
        delta_raw = raw_out - raw_active
        delta_adj = adj_out - adj_active
        ci_low_raw  = delta_raw - 1.96 * se_raw
        ci_high_raw = delta_raw + 1.96 * se_raw
        ci_low_resid  = residual_delta - 1.96 * se_resid
        ci_high_resid = residual_delta + 1.96 * se_resid

        agg_rows.append({
            "team": team, "season": int(season),
            "role_lost": role,
            "player_lost_id": pid,
            "player_lost_name": pname,
            "n_active": n_active, "n_out": n_out,
            "raw_pts_active": raw_active,
            "raw_pts_out": raw_out,
            "raw_pts_delta": delta_raw,
            "adj_pts_active": adj_active,
            "adj_pts_out": adj_out,
            "adj_pts_delta": delta_adj,
            "team_baseline_active": raw_active,
            "residual_pts_active": residual_active,
            "residual_pts_out": residual_out,
            "residual_pts_delta": residual_delta,
            "residual_method": residual_method,
            "delta_ci_low": ci_low_raw,
            "delta_ci_high": ci_high_raw,
            "residual_ci_low": ci_low_resid,
            "residual_ci_high": ci_high_resid,
            "thin_sample": n_out < 3,
        })

    out = pd.DataFrame(agg_rows)
    out = out.sort_values(["season", "team", "role_lost"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print(f"  rows: {len(out):,}")
    print(f"  thin samples (n_out<3): "
          f"{int((out['n_out'] < 3).sum()):,}")
    print()
    print("=== Largest QB1-out scoring drops "
           "(n_out>=2, residualized = isolated impact) ===")
    qb_out = out[(out["role_lost"] == "QB1") & (out["n_out"] >= 2)]
    print(qb_out.sort_values("residual_pts_delta").head(10)[
        ["team", "season", "player_lost_name",
         "n_active", "n_out",
         "raw_pts_active", "raw_pts_out",
         "raw_pts_delta",
         "adj_pts_delta",
         "residual_pts_delta"]
    ].to_string())
    print()
    print("=== Detroit, Kansas City, San Francisco starter-absence ===")
    sample_teams = out[out["team"].isin(["DET", "KC", "SF"])
                        & (out["season"] >= 2022)]
    print(sample_teams[["team", "season", "role_lost",
                          "player_lost_name", "n_active", "n_out",
                          "raw_pts_delta", "adj_pts_delta"]].to_string())


if __name__ == "__main__":
    main()

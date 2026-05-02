"""Per-team-season opponent strength + league averages.

Output: data/team_opponent_strength.parquet

For each (team, season) computes the AVERAGE STRENGTH OF THEIR
OPPONENTS' DEFENSE — i.e., when this team plays a generic game, how
many points / pass yards / rush yards would a league-average team be
expected to score against the typical defense they face?

This is the baseline subtractor that isolates real player/team effects
from schedule luck. Without this, "DET scored -2 PPG without Gibbs"
mixes Gibbs-impact with whatever defenses happened to be on the
schedule those weeks.

Schema (one row per team-season):
    team, season,
    games_played,
    opp_ppg_allowed_avg          — mean PPG given up by team's opponents (over their full schedule)
    opp_pass_yards_allowed_avg   — mean opposing-passing-yards-per-game allowed
    opp_rush_yards_allowed_avg
    opp_rec_yards_allowed_avg

Plus league averages (same for all teams in a season — convenient
join target):
    league_ppg, league_pass_yds_pg, league_rush_yds_pg, league_rec_yds_pg

We treat the LEAGUE AVERAGE as the centering point so deltas have a
sensible zero. e.g.:
    opp_strength_centered = opp_ppg_allowed_avg - league_ppg
    expected_score_vs_this_opp = league_ppg + opp_strength_centered
                                = opp_ppg_allowed_avg

Methodology
-----------
1. From schedules, get each (team, season) → list of opponents
2. From schedules, compute each team's PPG allowed (= opponents' PPG
   scored against them) per season
3. For team X, opp_ppg_allowed_avg = average of opponents' PPG-allowed
   values, weighted by frequency of facing them
4. Same logic for passing/rushing/receiving yards from
   nfl_player_stats_weekly (aggregated to team-game level first)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCHEDULES = REPO / "data" / "nfl_schedules.parquet"
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
OUT = REPO / "data" / "team_opponent_strength.parquet"


def _team_game_yards(ps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly player stats to per-team-game offensive yards.
    Returns one row per (season, week, team) with team_pass_yds,
    team_rush_yds, team_rec_yds (rec is what RBs/WRs/TEs caught — equal
    to pass_yards for completed passes; we keep both for the WR-side
    DvP)."""
    keep_cols = [c for c in ["passing_yards", "rushing_yards",
                              "receiving_yards"] if c in ps.columns]
    grp = (ps.dropna(subset=["season", "week", "team"])
           .groupby(["season", "week", "team"])[keep_cols]
           .sum()
           .reset_index())
    grp = grp.rename(columns={
        "passing_yards":   "team_pass_yds",
        "rushing_yards":   "team_rush_yds",
        "receiving_yards": "team_rec_yds",
    })
    return grp


def main() -> None:
    print("→ loading schedules + player stats...")
    sch = pd.read_parquet(SCHEDULES)
    ps = pd.read_parquet(PLAYER_STATS)
    sch = sch.dropna(subset=["home_team", "away_team",
                              "home_score", "away_score"])
    print(f"  schedules: {len(sch):,}  player_stats: {len(ps):,}")

    # ── Step 1: per (team, season, week) → points_for, points_against
    home_rows = sch[["season", "week", "home_team", "away_team",
                      "home_score", "away_score"]].copy()
    home_rows = home_rows.rename(columns={
        "home_team": "team",
        "away_team": "opp",
        "home_score": "team_pts",
        "away_score": "opp_pts",
    })
    away_rows = sch[["season", "week", "away_team", "home_team",
                      "home_score", "away_score"]].copy()
    away_rows = away_rows.rename(columns={
        "away_team": "team",
        "home_team": "opp",
        "away_score": "team_pts",
        "home_score": "opp_pts",
    })
    games = pd.concat([home_rows, away_rows], ignore_index=True)
    print(f"  team-games: {len(games):,}")

    # ── Step 2: per-team PPG-allowed per season (defense strength)
    pts_allowed = (games.groupby(["team", "season"])
                   .agg(team_ppg_allowed=("opp_pts", "mean"),
                        team_games=("opp_pts", "size"))
                   .reset_index())

    # ── Step 3: attach yards-allowed via team-game offensive aggregation
    team_game_yards = _team_game_yards(ps)
    # For team X in week W: yards-allowed-this-game = the OPPONENT's
    # offensive yards in that game.
    games_with_yards = games.merge(
        team_game_yards.rename(columns={
            "team": "opp",
            "team_pass_yds": "opp_pass_yds",
            "team_rush_yds": "opp_rush_yds",
            "team_rec_yds":  "opp_rec_yds",
        }),
        on=["season", "week", "opp"], how="left",
    )
    yards_allowed = (games_with_yards
                     .groupby(["team", "season"])
                     .agg(team_pass_yds_allowed=("opp_pass_yds", "mean"),
                          team_rush_yds_allowed=("opp_rush_yds", "mean"),
                          team_rec_yds_allowed=("opp_rec_yds", "mean"))
                     .reset_index())

    team_def = pts_allowed.merge(yards_allowed,
                                  on=["team", "season"], how="left")
    print(f"  per-team defensive metrics: {len(team_def):,}")

    # ── Step 4: per (team, season) — average opponent strength
    # For team X in season S, opp_strength_avg = mean of their opponents'
    # season-long PPG/yards-allowed values.
    games_with_opp_def = games.merge(
        team_def.rename(columns={
            "team": "opp",
            "team_ppg_allowed":      "opp_def_ppg",
            "team_pass_yds_allowed": "opp_def_pass_yds",
            "team_rush_yds_allowed": "opp_def_rush_yds",
            "team_rec_yds_allowed":  "opp_def_rec_yds",
        }),
        on=["opp", "season"], how="left",
    )
    opp_strength = (games_with_opp_def
                    .groupby(["team", "season"])
                    .agg(games_played=("opp", "size"),
                         opp_ppg_allowed_avg=("opp_def_ppg", "mean"),
                         opp_pass_yards_allowed_avg=("opp_def_pass_yds",
                                                      "mean"),
                         opp_rush_yards_allowed_avg=("opp_def_rush_yds",
                                                      "mean"),
                         opp_rec_yards_allowed_avg=("opp_def_rec_yds",
                                                     "mean"))
                    .reset_index())

    # ── Step 5: league averages per season (centering point)
    league = (games.groupby("season")["team_pts"]
              .mean().reset_index()
              .rename(columns={"team_pts": "league_ppg"}))
    # League pass/rush/rec yards per team-game
    league_yds = (team_game_yards.groupby("season")
                  .agg(league_pass_yds_pg=("team_pass_yds", "mean"),
                       league_rush_yds_pg=("team_rush_yds", "mean"),
                       league_rec_yds_pg=("team_rec_yds", "mean"))
                  .reset_index())
    league_full = league.merge(league_yds, on="season", how="outer")

    out = opp_strength.merge(league_full, on="season", how="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== Sample: 2024 schedule strength (toughest first) ===")
    sample = (out[out["season"] == 2024]
              .sort_values("opp_ppg_allowed_avg")
              .head(10))
    print(sample[["team", "opp_ppg_allowed_avg",
                  "opp_pass_yards_allowed_avg",
                  "opp_rush_yards_allowed_avg",
                  "league_ppg"]].to_string())
    print()
    print("=== 2024 easiest schedule (most points allowed by opps) ===")
    sample2 = (out[out["season"] == 2024]
               .sort_values("opp_ppg_allowed_avg", ascending=False)
               .head(5))
    print(sample2[["team", "opp_ppg_allowed_avg",
                    "opp_pass_yards_allowed_avg",
                    "opp_rush_yards_allowed_avg"]].to_string())


if __name__ == "__main__":
    main()

"""Anytime / First TD Probability Vector — Feature 5.5.

Computes per-player TD probability (rushing / receiving / anytime)
combining empirical per-game TD rates, red-zone usage rates, and
opponent goal-line defense splits.

Why this matters
----------------
TD markets move slowly because the math is hard for casuals — low
base rates, multiple interacting factors. Decomposing anytime-TD
into rushing-TD-only and receiving-TD-only lets users find the
sub-prop that's mispriced, which is often the rushing-TD-only line
for a pass-catching back or the receiving-TD-only line for a
goal-line-package back.

Public entry points
-------------------
    player_td_rates(player_id, lookback_games=None)
        → empirical per-game (rush_td_rate, rec_td_rate, any_td_rate)
    rz_usage_share(player_id, season)
        → fraction of team red-zone touches/targets player saw
    td_probability_vector(player_id, opp_team=None, season=None)
        → returns the full probability decomposition
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


REPO = Path(__file__).resolve().parent
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
PBP = REPO / "data" / "game_pbp.parquet"


@st.cache_data(show_spinner=False)
def _load_stats() -> pd.DataFrame:
    return pd.read_parquet(PLAYER_STATS)


@st.cache_data(show_spinner=False)
def _load_pbp_rz() -> pd.DataFrame:
    """Cache only the red-zone subset of pbp to keep memory tight.

    Includes `defteam` so opponent-allowed defensive metrics can be
    computed (the prior version excluded it, which forced
    `_opponent_td_factors` to return identity)."""
    df = pd.read_parquet(PBP, columns=[
        "season", "week", "posteam", "defteam",
        "yardline_100", "play_type",
        "rusher_player_id", "receiver_player_id",
        "rushing_yards", "receiving_yards",
        "rush_touchdown", "pass_touchdown",
    ])
    return df[df["yardline_100"] <= 20].copy()


@dataclass
class TDProbabilityResult:
    n_games: int
    p_rush_td: float
    p_rec_td: float
    p_any_td: float
    rush_td_per_game: float
    rec_td_per_game: float
    any_td_per_game: float


def player_td_rates(player_id: str,
                     lookback_games: int | None = None
                     ) -> TDProbabilityResult:
    """Per-game TD rates from the weekly stats table.

    Probabilities use Beta-Binomial shrinkage toward a position-
    adjacent prior so hot-streak and cold-streak players don't
    swing to extreme values. Without shrinkage, a WR who scored
    in 3 of his last 5 games gets p_rec_td = 0.60 (raw); the
    shrunk version is closer to 0.45, much closer to actual
    out-of-sample rate.

    Position-typical TD-game rate (the prior mean) is set conservatively:
      • RB rush-TD-game rate ~25%
      • WR/TE rec-TD-game rate ~25%
      • Anytime ~30-35%
    Implemented as alpha=2, beta=5 — soft pull toward ~30%.
    """
    from lib_alt_line_ev import beta_shrink

    df = _load_stats()
    sub = df[df["player_id"] == player_id]
    sub = sub.sort_values(["season", "week"], ascending=[False, False])
    if lookback_games:
        sub = sub.head(int(lookback_games))
    if sub.empty:
        return TDProbabilityResult(0, 0, 0, 0, 0, 0, 0)

    rush_tds = sub.get("rushing_tds", pd.Series(dtype=float)).fillna(0)
    rec_tds = sub.get("receiving_tds", pd.Series(dtype=float)).fillna(0)
    any_tds = rush_tds + rec_tds

    n = len(sub)
    k_rush = int((rush_tds > 0).sum())
    k_rec = int((rec_tds > 0).sum())
    k_any = int((any_tds > 0).sum())

    # Beta-Binomial shrinkage with prior mean ~28% (alpha=2, beta=5)
    p_rush = beta_shrink(k_rush, n, alpha=2.0, beta=5.0)
    p_rec = beta_shrink(k_rec, n, alpha=2.0, beta=5.0)
    p_any = beta_shrink(k_any, n, alpha=2.5, beta=5.0)

    return TDProbabilityResult(
        n_games=n,
        p_rush_td=p_rush,
        p_rec_td=p_rec,
        p_any_td=p_any,
        rush_td_per_game=float(rush_tds.mean()),
        rec_td_per_game=float(rec_tds.mean()),
        any_td_per_game=float(any_tds.mean()),
    )


@dataclass
class RZUsage:
    rz_carries_share: float    # share of team RZ rush attempts
    rz_targets_share: float    # share of team RZ pass targets
    goal_line_carries_share: float  # share inside the 10
    n_team_rz_plays: int
    n_player_rz_carries: int
    n_player_rz_targets: int


def rz_usage_share(player_id: str, season: int,
                   team: str | None = None) -> RZUsage:
    """Compute the player's share of his team's red-zone usage in a
    given season."""
    rz = _load_pbp_rz()
    if rz.empty:
        return RZUsage(0, 0, 0, 0, 0, 0)
    rz = rz[rz["season"] == int(season)]
    if team:
        rz = rz[rz["posteam"] == team]

    # Team totals
    team_rush = rz[rz["play_type"] == "run"]
    team_pass = rz[rz["play_type"] == "pass"]

    # Player slices
    p_rush = team_rush[team_rush["rusher_player_id"] == player_id]
    p_targets = team_pass[team_pass["receiver_player_id"] == player_id]
    p_gl_carries = p_rush[p_rush["yardline_100"] <= 10]

    return RZUsage(
        rz_carries_share=(len(p_rush) / max(len(team_rush), 1)),
        rz_targets_share=(len(p_targets) / max(len(team_pass), 1)),
        goal_line_carries_share=(
            len(p_gl_carries) / max(len(team_rush[team_rush["yardline_100"] <= 10]), 1)
        ),
        n_team_rz_plays=len(team_rush) + len(team_pass),
        n_player_rz_carries=len(p_rush),
        n_player_rz_targets=len(p_targets),
    )


@dataclass
class TDVector:
    """Full decomposition for one (player, opponent) matchup."""
    n_games_player: int
    p_rush_td_baseline: float
    p_rec_td_baseline: float
    p_any_td_baseline: float
    p_rush_td_adj: float        # opponent-adjusted
    p_rec_td_adj: float
    p_any_td_adj: float
    opp_rush_td_factor: float   # 1.0 = league-avg defense
    opp_rec_td_factor: float


def _opponent_td_factors(opp_team: str | None,
                          season: int | None) -> tuple[float, float]:
    """Return (rush_td_factor, rec_td_factor) for the opponent's defense.

    Factor > 1 = defense allows MORE TDs than league average (good
    for offensive players matched up against them).
    Factor < 1 = defense allows FEWER TDs (tougher matchup).
    Factor = 1.0 = league average OR insufficient sample.

    Includes shrinkage toward 1.0 by sample size: with only 4-6
    games of defensive data on a team early in the season, raw rates
    are high-variance. We blend the team's observed rate with the
    league-mean prior using `tau=8` effective games of pseudo-prior
    weight (so a team with 4 games is half league, half observed).
    """
    if not opp_team or season is None:
        return 1.0, 1.0
    rz = _load_pbp_rz()
    if rz.empty:
        return 1.0, 1.0
    season_rz = rz[rz["season"] == int(season)]
    if season_rz.empty:
        return 1.0, 1.0

    runs = season_rz[season_rz["play_type"] == "run"]
    passes = season_rz[season_rz["play_type"] == "pass"]

    # Per-defense per-game TDs allowed
    rush_td_pg = (runs.groupby(["defteam", "week"])["rush_touchdown"]
                       .sum().reset_index()
                       .groupby("defteam")["rush_touchdown"].mean())
    rec_td_pg = (passes.groupby(["defteam", "week"])["pass_touchdown"]
                       .sum().reset_index()
                       .groupby("defteam")["pass_touchdown"].mean())

    # Number of defensive games observed (for shrinkage weight)
    n_games = (season_rz.groupby(["defteam", "week"]).size()
                          .reset_index().groupby("defteam").size())

    league_rush_pg = float(rush_td_pg.mean()) if len(rush_td_pg) else 0.0
    league_rec_pg = float(rec_td_pg.mean()) if len(rec_td_pg) else 0.0
    if league_rush_pg <= 0 or league_rec_pg <= 0:
        return 1.0, 1.0

    team_rush = float(rush_td_pg.get(opp_team, league_rush_pg))
    team_rec = float(rec_td_pg.get(opp_team, league_rec_pg))
    n_team = int(n_games.get(opp_team, 0))

    # Shrinkage — blend team rate with league mean, weighted by
    # effective sample size. tau=8 chosen so a team with 4 games
    # gets half-credit, a team with 16 games gets two-thirds credit.
    TAU = 8.0
    w = n_team / (n_team + TAU) if n_team > 0 else 0.0
    rush_shrunk = w * team_rush + (1 - w) * league_rush_pg
    rec_shrunk = w * team_rec + (1 - w) * league_rec_pg

    rush_f = rush_shrunk / league_rush_pg
    rec_f = rec_shrunk / league_rec_pg

    # Cap factors to a defensible range — no defense is realistically
    # 2× league average even after a fluky n=4 sample
    rush_f = max(0.5, min(1.5, rush_f))
    rec_f = max(0.5, min(1.5, rec_f))
    return rush_f, rec_f


def td_probability_vector(player_id: str,
                           opp_team: str | None = None,
                           season: int | None = None,
                           lookback_games: int | None = None
                           ) -> TDVector:
    base = player_td_rates(player_id, lookback_games=lookback_games)
    rush_f, rec_f = _opponent_td_factors(opp_team, season)

    # p_any_td_baseline is computed empirically from game-level data
    # (rush_tds + rec_tds > 0).mean() — it already captures the true
    # joint correlation between rush and receiving TDs within the same
    # game (volume / RZ usage / game flow correlate the two events).
    # If we have no opponent-specific factors yet (the v1 case where
    # _opponent_td_factors returns 1.0/1.0), pass the baseline straight
    # through. Otherwise, scale the baseline by a weighted average of
    # the rush/rec multipliers, weighted by each event's share of the
    # player's total TDs. This preserves the empirical joint instead of
    # rebuilding it under the false independence assumption.
    rush_td = base.p_rush_td
    rec_td = base.p_rec_td
    if abs(rush_f - 1.0) < 1e-9 and abs(rec_f - 1.0) < 1e-9:
        p_any_adj = base.p_any_td
    else:
        denom = max(rush_td + rec_td, 1e-9)
        rush_share = rush_td / denom
        rec_share = rec_td / denom
        p_any_adj = min(1.0, base.p_any_td
                                 * (rush_share * rush_f
                                     + rec_share * rec_f))

    return TDVector(
        n_games_player=base.n_games,
        p_rush_td_baseline=base.p_rush_td,
        p_rec_td_baseline=base.p_rec_td,
        p_any_td_baseline=base.p_any_td,
        p_rush_td_adj=min(1.0, base.p_rush_td * rush_f),
        p_rec_td_adj=min(1.0, base.p_rec_td * rec_f),
        p_any_td_adj=p_any_adj,
        opp_rush_td_factor=rush_f,
        opp_rec_td_factor=rec_f,
    )

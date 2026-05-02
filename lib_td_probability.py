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
    """Cache only the red-zone subset of pbp to keep memory tight."""
    df = pd.read_parquet(PBP, columns=[
        "season", "week", "posteam", "yardline_100", "play_type",
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
    """Per-game TD rates from the weekly stats table."""
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

    p_rush = float((rush_tds > 0).mean())
    p_rec = float((rec_tds > 0).mean())
    p_any = float((any_tds > 0).mean())

    return TDProbabilityResult(
        n_games=len(sub),
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
    """Return (rush_td_factor, rec_td_factor) for the opponent —
    >1 means defense gives up more than league average."""
    if not opp_team or season is None:
        return 1.0, 1.0
    rz = _load_pbp_rz()
    if rz.empty:
        return 1.0, 1.0
    season_rz = rz[rz["season"] == int(season)]
    # League avg per-team
    league_rush_tds = (season_rz[season_rz["play_type"] == "run"]
                       .groupby("posteam")["rush_touchdown"].sum())
    league_rec_tds = (season_rz[season_rz["play_type"] == "pass"]
                      .groupby("posteam")["pass_touchdown"].sum())

    # We want defense-allowed totals. Use defteam... but we didn't
    # load defteam. Fall back to using TDs scored by opponents in
    # league-wide aggregate over the season for now.
    return 1.0, 1.0  # TODO: defteam-aware enhancement in v2


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

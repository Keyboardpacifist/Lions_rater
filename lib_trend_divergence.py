"""Snap-Share / Target-Share Trend Divergence — Feature 5.6.

Flags players whose recent (last-N-week) usage has decoupled from
their season baseline. Books often anchor to season averages and
miss recent role expansions or contractions — that gap is where
prop-bet edge lives.

Public entry points
-------------------
    compute_player_window(player_id, season, week, lookback=3)
        → returns (recent, season, divergence) for usage stats
    league_divergence_today(season, week, position, min_z=0.5)
        → top divergence flags across the league this week
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


REPO = Path(__file__).resolve().parent
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"

# Stats we track for divergence — usage indicators that drive props.
USAGE_STATS = [
    "targets", "receptions", "receiving_yards",
    "carries", "rushing_yards",
    "attempts", "completions", "passing_yards",
    "fantasy_points_ppr",
]


@st.cache_data(show_spinner=False)
def _load_stats() -> pd.DataFrame:
    if not PLAYER_STATS.exists():
        return pd.DataFrame()
    return pd.read_parquet(PLAYER_STATS)


@dataclass
class DivergenceRow:
    player_id: str
    player_display_name: str
    team: str
    position: str
    season: int
    week: int
    stat: str
    recent_avg: float       # avg over last N weeks
    season_avg: float       # avg over the rest of the season
    delta: float            # recent - season
    delta_z: float          # standardized vs the player's own variance
    n_recent: int
    n_season: int


def compute_player_window(player_id: str, season: int, week: int,
                           lookback: int = 3
                           ) -> list[DivergenceRow]:
    """For one (player, season, week) snapshot, compute recent vs.
    season divergence on each usage stat. The player's own historical
    variance defines what counts as a *meaningful* shift (>0.5σ)."""
    df = _load_stats()
    if df.empty:
        return []

    season_df = df[(df["player_id"] == player_id)
                   & (df["season"] == int(season))
                   & (df["week"] < int(week))].sort_values("week")
    if len(season_df) < lookback + 1:
        return []

    recent = season_df.tail(lookback)
    earlier = season_df.iloc[:-lookback]

    rows: list[DivergenceRow] = []
    for stat in USAGE_STATS:
        if stat not in season_df.columns:
            continue
        rec_vals = recent[stat].fillna(0).astype(float)
        ear_vals = earlier[stat].fillna(0).astype(float)
        if len(rec_vals) == 0 or len(ear_vals) == 0:
            continue
        rec_avg = float(rec_vals.mean())
        sea_avg = float(ear_vals.mean())
        # Use the union variance to z-score the delta. If everything
        # is 0 (e.g., this stat doesn't apply to this player), skip.
        all_vals = pd.concat([rec_vals, ear_vals])
        std = float(all_vals.std(ddof=0))
        if std < 1e-6:
            continue
        delta = rec_avg - sea_avg
        z = delta / std
        rows.append(DivergenceRow(
            player_id=player_id,
            player_display_name=str(season_df["player_display_name"].iloc[-1]),
            team=str(season_df["team"].iloc[-1]),
            position=str(season_df["position"].iloc[-1]),
            season=int(season), week=int(week),
            stat=stat,
            recent_avg=rec_avg, season_avg=sea_avg,
            delta=delta, delta_z=z,
            n_recent=len(rec_vals), n_season=len(ear_vals),
        ))
    return rows


def league_divergence_today(season: int, week: int,
                              position: str | None = None,
                              min_z: float = 0.5,
                              lookback: int = 3,
                              min_season_games: int = 4
                              ) -> pd.DataFrame:
    """For every player who has played this season, compute divergence
    and return only the rows where |z| ≥ min_z."""
    df = _load_stats()
    if df.empty:
        return pd.DataFrame()

    base = df[(df["season"] == int(season))
              & (df["week"] < int(week))]
    if position:
        base = base[base["position"] == position]
    # Require min_season_games of season history before considering
    counts = base.groupby("player_id").size().reset_index(name="n")
    eligible = counts[counts["n"] >= max(min_season_games, lookback + 1)]
    out: list[DivergenceRow] = []
    for pid in eligible["player_id"]:
        rows = compute_player_window(pid, season, week, lookback=lookback)
        for r in rows:
            if abs(r.delta_z) >= min_z:
                out.append(r)
    if not out:
        return pd.DataFrame()
    return pd.DataFrame([r.__dict__ for r in out])

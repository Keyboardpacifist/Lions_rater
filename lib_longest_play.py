"""Longest-Play Edge Finder — Feature 5.7.

Models the empirical distribution of a player's per-game longest
single play (rush or reception) and converts target yards X into
a probability via the historical longest-per-game distribution,
plus an opportunity-aware bootstrap.

Why this matters
----------------
Books model "longest reception" and "longest rush" props on smooth
distributions, but reality is bimodal — most plays are short, then
a heavy tail. Players with elite explosive rates have structurally
undervalued longest-play markets.

Public entry points
-------------------
    longest_play_distribution(player_id, kind="reception")
        → DataFrame with one row per (season, week) and the longest
          single play of that game
    p_longest_at_least(player_id, threshold, kind, n_games_min=10)
        → empirical Pr(longest single play in next game ≥ threshold)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


REPO = Path(__file__).resolve().parent
PBP = REPO / "data" / "game_pbp.parquet"


@st.cache_data(show_spinner=False)
def _load_pbp() -> pd.DataFrame:
    if not PBP.exists():
        return pd.DataFrame()
    cols = ["game_id", "season", "week", "play_type",
            "receiver_player_id", "receiving_yards",
            "rusher_player_id", "rushing_yards"]
    return pd.read_parquet(PBP, columns=cols)


@dataclass
class LongestPlayResult:
    n_games: int
    threshold: float
    p_at_least: float       # historical Pr(longest ≥ threshold)
    median_longest: float
    p10_longest: float
    p90_longest: float
    kind: str               # "reception" / "rush"


def longest_play_distribution(player_id: str,
                              kind: str = "reception"
                              ) -> pd.DataFrame:
    """Per-game longest play DataFrame for a player."""
    pbp = _load_pbp()
    if pbp.empty:
        return pd.DataFrame()
    if kind == "reception":
        plays = pbp[(pbp["receiver_player_id"] == player_id)
                    & pbp["receiving_yards"].notna()]
        yards_col = "receiving_yards"
    else:
        plays = pbp[(pbp["rusher_player_id"] == player_id)
                    & pbp["rushing_yards"].notna()]
        yards_col = "rushing_yards"
    if plays.empty:
        return plays
    longest = (plays.groupby(["game_id", "season", "week"])[yards_col]
               .max()
               .reset_index()
               .rename(columns={yards_col: "longest_play"}))
    return longest.sort_values(["season", "week"])


def p_longest_at_least(player_id: str, threshold: float,
                       kind: str = "reception",
                       n_games_min: int = 10) -> LongestPlayResult:
    dist = longest_play_distribution(player_id, kind=kind)
    if dist.empty or len(dist) < n_games_min:
        return LongestPlayResult(
            n_games=len(dist), threshold=threshold,
            p_at_least=float("nan"),
            median_longest=float("nan"),
            p10_longest=float("nan"),
            p90_longest=float("nan"),
            kind=kind,
        )
    vals = dist["longest_play"].astype(float)
    return LongestPlayResult(
        n_games=len(vals),
        threshold=float(threshold),
        p_at_least=float((vals >= threshold).mean()),
        median_longest=float(np.percentile(vals, 50)),
        p10_longest=float(np.percentile(vals, 10)),
        p90_longest=float(np.percentile(vals, 90)),
        kind=kind,
    )


def player_options(kind: str = "reception",
                    min_games: int = 16) -> pd.DataFrame:
    """Return players with enough samples to build a reasonable
    longest-play model."""
    pbp = _load_pbp()
    if pbp.empty:
        return pd.DataFrame()
    if kind == "reception":
        plays = pbp[(pbp["receiver_player_id"].notna())
                    & pbp["receiving_yards"].notna()]
        # Need names — pull from player_stats for join
        players = (plays.groupby(["receiver_player_id"])
                   ["game_id"].nunique()
                   .reset_index()
                   .rename(columns={"receiver_player_id": "player_id",
                                    "game_id": "n_games"}))
    else:
        plays = pbp[(pbp["rusher_player_id"].notna())
                    & pbp["rushing_yards"].notna()]
        players = (plays.groupby(["rusher_player_id"])
                   ["game_id"].nunique()
                   .reset_index()
                   .rename(columns={"rusher_player_id": "player_id",
                                    "game_id": "n_games"}))
    players = players[players["n_games"] >= min_games]

    # Attach name + most-recent team from player_stats
    ps = pd.read_parquet(REPO / "data" / "nfl_player_stats_weekly.parquet",
                          columns=["player_id", "player_display_name",
                                   "position", "team", "season", "week"])
    ps_sorted = (ps.dropna(subset=["player_id"])
                  .sort_values(["season", "week"],
                                 ascending=[False, False]))
    name_lookup = (ps_sorted.drop_duplicates("player_id")[
        ["player_id", "player_display_name", "position", "team"]
    ])
    return (players.merge(name_lookup, on="player_id", how="left")
                  .sort_values("n_games", ascending=False)
                  .reset_index(drop=True))

"""Accessor functions for the scheme-delta table.

Built by `tools/build_scheme_deltas.py`. Contains per-(team, season,
side) scheme metrics plus league-relative delta columns.

Public entry points:
    load_scheme_deltas()                      — cached parquet load
    get_team_scheme(team, season, side)        — one row as dict
    rank_teams(season, side, metric, top_n)    — leaderboard
    season_year_over_year(team, side, metric)  — track scheme change
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
SCHEME = DATA / "scheme_deltas.parquet"


# Pretty labels for the lab UI.
METRIC_LABELS = {
    # Offense
    "pass_rate_overall":      "Pass rate (overall)",
    "pass_rate_neutral":      "Pass rate (neutral script)",
    "early_down_pass_rate":   "Early-down pass rate",
    "shotgun_rate":           "Shotgun rate",
    "no_huddle_rate":         "No-huddle rate",
    "rz_pass_rate":           "Red-zone pass rate",
    "fourth_down_go_rate":    "4th-down go-for-it rate",
    "avg_air_yards":          "Avg air yards",
    "short_pass_rate":        "Short-pass rate (<10 air yds)",
    "deep_pass_rate":         "Deep-pass rate (≥20 air yds)",
    "pass_left_rate":         "Pass left rate",
    "pass_middle_rate":       "Pass middle rate",
    "pass_right_rate":        "Pass right rate",
    "epa_per_play_off":       "EPA / play (offense)",
    # Defense
    "blitz_rate":             "Blitz rate (≥5 rushers)",
    "pressure_rate":          "Pressure rate",
    "man_coverage_rate":      "Man-coverage rate",
    "zone_coverage_rate":     "Zone-coverage rate",
    "coverage_label_rate":    "Coverage label availability",
    "box_loaded_rush_rate":   "8+ in box on rush plays",
    "epa_per_play_def":       "EPA / play allowed",
}

OFFENSE_METRICS = [
    "pass_rate_overall", "pass_rate_neutral", "early_down_pass_rate",
    "shotgun_rate", "no_huddle_rate", "rz_pass_rate",
    "fourth_down_go_rate", "avg_air_yards", "short_pass_rate",
    "deep_pass_rate", "pass_left_rate", "pass_middle_rate",
    "pass_right_rate", "epa_per_play_off",
]
DEFENSE_METRICS = [
    "blitz_rate", "pressure_rate", "man_coverage_rate",
    "zone_coverage_rate", "coverage_label_rate",
    "box_loaded_rush_rate", "epa_per_play_def",
]


@st.cache_data(show_spinner=False)
def load_scheme_deltas() -> pd.DataFrame:
    if not SCHEME.exists():
        return pd.DataFrame()
    return pd.read_parquet(SCHEME)


def get_team_scheme(team: str, season: int,
                    side: str = "offense") -> pd.Series | None:
    df = load_scheme_deltas()
    if df.empty:
        return None
    sub = df[(df["team"] == team)
             & (df["season"] == int(season))
             & (df["side"] == side)]
    return None if sub.empty else sub.iloc[0]


def rank_teams(season: int, side: str, metric: str,
               top_n: int = 10, ascending: bool = False
               ) -> pd.DataFrame:
    """Return top_n teams ranked by `metric` for a given season+side."""
    df = load_scheme_deltas()
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    sub = df[(df["season"] == int(season)) & (df["side"] == side)]
    return (sub.sort_values(metric, ascending=ascending)
               .head(top_n)
               .reset_index(drop=True))


def season_year_over_year(team: str, side: str,
                          metric: str) -> pd.DataFrame:
    """Track one team-side metric across all available seasons."""
    df = load_scheme_deltas()
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    sub = df[(df["team"] == team) & (df["side"] == side)]
    return sub.sort_values("season")[
        ["season", metric, f"{metric}_delta"]
    ].reset_index(drop=True)

"""Weather Production Window engine — Feature 4.5.

Cohort-matches a player's historical games by weather conditions and
returns the empirical P10/P50/P90 of his primary production stat.

Public entry points
-------------------
    load_weather_table()              — cached parquet load
    primary_stat_for_position(pos)    — which stat to model
    weather_cohort(player_id, ...)    — empirical distribution
    confidence_for_n(n)               — HIGH/MEDIUM/LOW label
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
WEATHER_TABLE = DATA / "player_games_weather.parquet"


# Map position → primary stat we'll model under weather conditions.
PRIMARY_STAT = {
    "QB":  "passing_yards",
    "RB":  "rushing_yards",
    "FB":  "rushing_yards",
    "WR":  "receiving_yards",
    "TE":  "receiving_yards",
    "K":   "fg_pct",
    "P":   None,
}


def primary_stat_for_position(pos: str) -> str | None:
    return PRIMARY_STAT.get(str(pos).upper())


@st.cache_data(show_spinner=False)
def load_weather_table() -> pd.DataFrame:
    if not WEATHER_TABLE.exists():
        return pd.DataFrame()
    return pd.read_parquet(WEATHER_TABLE)


def confidence_for_n(n: int) -> str:
    if n >= 15:
        return "HIGH"
    if n >= 6:
        return "MEDIUM"
    return "LOW"


@dataclass
class WeatherResult:
    n_games: int
    p10: float
    p50: float
    p90: float
    mean: float
    confidence: str   # HIGH / MEDIUM / LOW
    stat: str         # passing_yards / receiving_yards / rushing_yards
    cohort_mode: str  # "player" / "tier_blend" / "league"


def weather_cohort(player_id: str, position: str,
                    target_temp: float | None = None,
                    target_wind: float | None = None,
                    target_roof: str | None = None,
                    target_surface: str | None = None,
                    temp_tol: float = 8.0,
                    wind_tol: float = 5.0,
                    min_games: int = 5) -> WeatherResult:
    """Find player's historical games matching the target weather and
    return empirical P10/P50/P90 of his primary production stat."""
    stat = primary_stat_for_position(position)
    if stat is None:
        return WeatherResult(0, 0, 0, 0, 0, "LOW", "—", "league")

    df = load_weather_table()
    if df.empty:
        return WeatherResult(0, 0, 0, 0, 0, "LOW", stat, "league")

    own = df[(df["player_id"] == player_id) & (df[stat].notna())].copy()

    # Apply weather filters with tolerance bounds. Roof and surface are
    # exact match; temp and wind use ±tolerance.
    cohort = own.copy()
    if target_temp is not None and "temp" in cohort.columns:
        cohort = cohort[cohort["temp"].notna()
                        & cohort["temp"].between(target_temp - temp_tol,
                                                  target_temp + temp_tol)]
    if target_wind is not None and "wind" in cohort.columns:
        cohort = cohort[cohort["wind"].notna()
                        & cohort["wind"].between(max(0, target_wind - wind_tol),
                                                  target_wind + wind_tol)]
    if target_roof and "roof" in cohort.columns:
        cohort = cohort[cohort["roof"] == target_roof]
    if target_surface and "surface" in cohort.columns:
        cohort = cohort[cohort["surface"].astype(str).str.contains(
            target_surface, case=False, na=False)]

    cohort_mode = "player"
    if len(cohort) < min_games:
        # Fall back: blend with player's overall games so we have at
        # least *something*. Mark as "tier_blend" so confidence reflects.
        cohort = own
        cohort_mode = "tier_blend" if len(own) > 0 else "league"

    if cohort.empty or cohort_mode == "league":
        # Truly nothing — fall back to all players at this position
        league = df[(df["position"] == position) & (df[stat].notna())]
        if league.empty:
            return WeatherResult(0, 0, 0, 0, 0, "LOW", stat, "league")
        vals = league[stat]
        return WeatherResult(
            n_games=len(vals),
            p10=float(np.percentile(vals, 10)),
            p50=float(np.percentile(vals, 50)),
            p90=float(np.percentile(vals, 90)),
            mean=float(vals.mean()),
            confidence="LOW",
            stat=stat,
            cohort_mode="league",
        )

    vals = cohort[stat]
    return WeatherResult(
        n_games=len(vals),
        p10=float(np.percentile(vals, 10)),
        p50=float(np.percentile(vals, 50)),
        p90=float(np.percentile(vals, 90)),
        mean=float(vals.mean()),
        confidence=confidence_for_n(len(vals)),
        stat=stat,
        cohort_mode=cohort_mode,
    )


def all_player_options(position: str | None = None,
                       min_games: int = 8) -> pd.DataFrame:
    """Return (player_id, player_display_name, position, team, n_games)
    for pickers in the lab UI. `team` is the player's MOST-RECENT team
    in the table (handles trades by picking the latest)."""
    df = load_weather_table()
    if df.empty:
        return df
    sub = df[df["player_id"].notna()].copy()
    if position:
        sub = sub[sub["position"] == position]
    counts = (sub.groupby(["player_id", "player_display_name", "position"])
              .size()
              .reset_index()
              .rename(columns={0: "n_games"}))
    # Most-recent team per player
    sub_sorted = sub.sort_values(["season", "week"],
                                  ascending=[False, False])
    teams = (sub_sorted.drop_duplicates(["player_id"])[
        ["player_id", "team"]
    ])
    out = counts.merge(teams, on="player_id", how="left")
    return (out[out["n_games"] >= min_games]
            .sort_values("n_games", ascending=False)
            .reset_index(drop=True))

"""
college_data.py — College football data module.
Place in repo root alongside lib_shared.py and team_selector.py.

Provides college z-score lookups for NFL players, enabling the
college/pro toggle on career arc charts.
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

COLLEGE_DATA_DIR = Path(__file__).resolve().parent / "data" / "college"

# Map NFL position pages to college parquet files
COLLEGE_PARQUET_MAP = {
    "qb": "college_qb_all_seasons.parquet",
    "wr": "college_wr_all_seasons.parquet",
    "te": "college_te_all_seasons.parquet",
    "rb": "college_rb_all_seasons.parquet",
    "de": "college_def_all_seasons.parquet",
    "dt": "college_def_all_seasons.parquet",
    "lb": "college_def_all_seasons.parquet",
    "cb": "college_def_all_seasons.parquet",
    "s": "college_def_all_seasons.parquet",
}

# Map NFL position groups to college defensive pos_groups
NFL_TO_COLLEGE_DEF_POS = {
    "de": ["EDGE"],
    "dt": ["DL"],
    "lb": ["LB"],
    "cb": ["CB"],
    "s": ["DB"],
}

# College z-score columns by position
COLLEGE_Z_COLS = {
    "qb": ["completion_pct_z", "td_rate_z", "int_rate_z", "yards_per_attempt_z",
            "pass_yards_z", "pass_tds_z", "rush_yards_total_z", "total_tds_z"],
    "wr": ["rec_yards_total_z", "rec_tds_total_z", "receptions_total_z",
            "yards_per_rec_z", "rec_long_z"],
    "te": ["rec_yards_total_z", "rec_tds_total_z", "receptions_total_z",
            "yards_per_rec_z"],
    "rb": ["rush_yards_total_z", "rush_tds_total_z", "carries_total_z",
            "yards_per_carry_z", "total_yards_z", "total_tds_z",
            "receptions_total_z", "rec_yards_total_z"],
    "de": ["sacks_per_game_z", "tfl_per_game_z", "qb_hurries_per_game_z",
           "tackles_per_game_z", "pressure_rate_z"],
    "dt": ["sacks_per_game_z", "tfl_per_game_z", "qb_hurries_per_game_z",
           "tackles_per_game_z", "pressure_rate_z"],
    "lb": ["tackles_per_game_z", "solo_tackles_per_game_z", "tfl_per_game_z",
           "sacks_per_game_z", "pd_per_game_z", "int_per_game_z"],
    "cb": ["pd_per_game_z", "int_per_game_z", "tackles_per_game_z",
           "solo_tackles_per_game_z", "tfl_per_game_z"],
    "s": ["pd_per_game_z", "int_per_game_z", "tackles_per_game_z",
          "solo_tackles_per_game_z", "tfl_per_game_z"],
}

# Friendly labels for college stats
COLLEGE_STAT_LABELS = {
    "completion_pct_z": "Comp %",
    "td_rate_z": "TD rate",
    "int_rate_z": "INT rate",
    "yards_per_attempt_z": "Yds/att",
    "pass_yards_z": "Pass yards",
    "pass_tds_z": "Pass TDs",
    "rush_yards_total_z": "Rush yards",
    "total_tds_z": "Total TDs",
    "rec_yards_total_z": "Rec yards",
    "rec_tds_total_z": "Rec TDs",
    "receptions_total_z": "Receptions",
    "yards_per_rec_z": "Yds/rec",
    "rec_long_z": "Long rec",
    "rush_tds_total_z": "Rush TDs",
    "carries_total_z": "Carries",
    "yards_per_carry_z": "Yds/carry",
    "total_yards_z": "Total yards",
    "sacks_per_game_z": "Sacks/gm",
    "tfl_per_game_z": "TFL/gm",
    "qb_hurries_per_game_z": "QB hurries/gm",
    "tackles_per_game_z": "Tackles/gm",
    "solo_tackles_per_game_z": "Solo tackles/gm",
    "pressure_rate_z": "Pressure rate",
    "pd_per_game_z": "PD/gm",
    "int_per_game_z": "INT/gm",
}


@st.cache_data
def load_college_parquet(position_group):
    """Load the college parquet for a given NFL position group."""
    parquet_name = COLLEGE_PARQUET_MAP.get(position_group)
    if not parquet_name:
        return pd.DataFrame()
    path = COLLEGE_DATA_DIR / parquet_name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Filter by defensive position group if needed
    if position_group in NFL_TO_COLLEGE_DEF_POS and "pos_group" in df.columns:
        allowed = NFL_TO_COLLEGE_DEF_POS[position_group]
        df = df[df["pos_group"].isin(allowed + ["UNKNOWN"])].copy()
    return df


@st.cache_data
def load_draft_linkage():
    """Load the college-to-NFL draft linkage table."""
    path = COLLEGE_DATA_DIR / "college_to_nfl_linked.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def find_college_history(nfl_player_name, nfl_player_id, position_group):
    """Find a player's college stats by matching through draft linkage.
    
    Returns: DataFrame of college seasons, or empty DataFrame.
    """
    linkage = load_draft_linkage()
    if len(linkage) == 0:
        return pd.DataFrame()
    
    college_data = load_college_parquet(position_group)
    if len(college_data) == 0:
        return pd.DataFrame()
    
    # Match priority: nfl_id (most reliable) → exact name → fuzzy.
    # The old code did fuzzy FIRST, which collided on common last names —
    # "Jack Campbell" matched "Roderick Campbell" (Northwestern), giving
    # the wrong school. We now exhaust precise matches before fuzzy AND
    # disable fuzzy entirely when an nfl_id was provided (in that case
    # a missing match means "not in linkage" — better to show no data
    # than guess and risk a wrong-school credibility hit).
    player_link = pd.DataFrame()

    if nfl_player_id and "nfl_id" in linkage.columns:
        player_link = linkage[linkage["nfl_id"] == nfl_player_id]

    if len(player_link) == 0:
        player_link = linkage[linkage["player"] == nfl_player_name]

    if len(player_link) == 0 and not nfl_player_id:
        # Fuzzy last-name fallback — only when caller had no nfl_id
        # to disambiguate. Risky (collides on common last names) so
        # we never run it when nfl_id was supplied and missed.
        player_link = linkage[
            linkage["player"].str.contains(
                nfl_player_name.split()[-1], na=False, case=False
            )
        ]

    if len(player_link) == 0:
        return pd.DataFrame()
    
    # Get the college player_id from the linkage
    college_ids = player_link["player_id"].unique()
    college_teams = player_link["team"].unique()
    
    # Find in college data by ID
    history = college_data[college_data["player_id"].isin(college_ids)]
    
    if len(history) == 0:
        # Try name + team match directly in college data
        last_name = nfl_player_name.split()[-1]
        for team in college_teams:
            history = college_data[
                (college_data["player"].str.contains(last_name, na=False, case=False)) &
                (college_data["team"] == team)
            ]
            if len(history) > 0:
                break
    
    return history.sort_values("season") if len(history) > 0 else pd.DataFrame()


def compute_college_composite(row, position_group):
    """Compute average z-score across available college stats."""
    z_cols = COLLEGE_Z_COLS.get(position_group, [])
    values = [row[c] for c in z_cols if c in row.index and pd.notna(row.get(c))]
    return np.mean(values) if values else np.nan

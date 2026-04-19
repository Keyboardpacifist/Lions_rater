"""
team_selector.py — Global team and season selector.
Place in repo root alongside lib_shared.py.

Renders "NFL Rater" title with team + season dropdowns in the main
content area (not the sidebar). Selection persists across pages.
"""
import streamlit as st

NFL_TEAMS = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

TEAM_ALIASES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "WSH": "WAS",
}

AVAILABLE_SEASONS = list(range(2024, 2015, -1))


def normalize_team(abbr):
    """Normalize team abbreviations to current names."""
    if abbr in TEAM_ALIASES:
        return TEAM_ALIASES[abbr]
    return abbr


def get_team_and_season():
    """Render title bar with team + season selectors in the main content area.
    Returns (team_abbr, season_year)."""

    if "selected_team" not in st.session_state:
        st.session_state.selected_team = "DET"
    if "selected_season" not in st.session_state:
        st.session_state.selected_season = 2024

    team_options = sorted(NFL_TEAMS.keys())
    team_labels = [f"{abbr} — {NFL_TEAMS[abbr]}" for abbr in team_options]
    current_idx = team_options.index(st.session_state.selected_team) if st.session_state.selected_team in team_options else 0

    # Title + dropdowns on one row
    col_title, col_team, col_season = st.columns([2, 2, 1])
    with col_title:
        st.markdown("<h1 style='margin:0; padding:4px 0;'>🏈 NFL Rater</h1>", unsafe_allow_html=True)
    with col_team:
        selected_label = st.selectbox(
            "Team",
            options=team_labels,
            index=current_idx,
            key="team_selector_widget",
            label_visibility="collapsed",
        )
    with col_season:
        selected_season = st.selectbox(
            "Season",
            options=AVAILABLE_SEASONS,
            index=AVAILABLE_SEASONS.index(st.session_state.selected_season) if st.session_state.selected_season in AVAILABLE_SEASONS else 0,
            key="season_selector_widget",
            label_visibility="collapsed",
        )

    selected_team = selected_label.split(" — ")[0]
    st.session_state.selected_team = selected_team
    st.session_state.selected_season = selected_season

    st.markdown("---")
    return selected_team, selected_season


def filter_by_team_and_season(df, team, season, team_col="recent_team", season_col="season_year"):
    """Filter a league-wide dataframe to a specific team and season.
    Handles historical team abbreviation changes and column name variations."""
    if team_col in df.columns:
        df[team_col] = df[team_col].apply(normalize_team)

    if season_col not in df.columns:
        if "season_year" in df.columns:
            season_col = "season_year"
        elif "season" in df.columns:
            season_col = "season"
        else:
            if team_col in df.columns:
                return df[df[team_col] == team].copy()
            return df.copy()

    mask = (df[season_col] == season)
    if team_col in df.columns:
        mask = mask & (df[team_col] == team)

    return df[mask].copy()

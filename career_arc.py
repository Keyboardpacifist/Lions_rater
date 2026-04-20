"""
career_arc.py — Career arc chart with college + NFL on one timeline.
Place in repo root alongside lib_shared.py and team_selector.py.

Shows a player's composite z-score plotted year by year. If college
data is available (via draft linkage), both college and NFL seasons
appear on the same chart with a visual divider at the draft.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from pathlib import Path
import polars as pl


def zscore_to_percentile(z):
    if pd.isna(z):
        return None
    return float(norm.cdf(z) * 100)


@st.cache_data
def load_league_data(parquet_path):
    """Load the full league-wide parquet for career arc lookups."""
    return pl.read_parquet(parquet_path).to_pandas()


def find_player_history(league_df, player_id, player_name, id_col="player_id", name_col="player_display_name"):
    """Find all seasons for a player by ID, falling back to name match."""
    if id_col in league_df.columns and pd.notna(player_id):
        history = league_df[league_df[id_col] == player_id].copy()
        if len(history) > 0:
            return history

    if name_col in league_df.columns and pd.notna(player_name):
        history = league_df[league_df[name_col] == player_name].copy()
        if len(history) > 0:
            return history

    for col in ["player_name", "player_display_name", "player", "full_name"]:
        if col in league_df.columns:
            history = league_df[league_df[col] == player_name].copy()
            if len(history) > 0:
                return history

    return pd.DataFrame()


def compute_composite_score(row, stat_cols, weights=None):
    """Compute a simple average of available z-score columns."""
    z_cols = [c for c in stat_cols if c.endswith("_z") and c in row.index]
    values = [row[c] for c in z_cols if pd.notna(row[c])]
    if not values:
        return np.nan
    return np.mean(values)


def _find_college_history(player_name, position_group):
    """Try to find college history for this player."""
    try:
        from college_data import find_college_history, COLLEGE_Z_COLS
        college_hist = find_college_history(player_name, None, position_group=position_group)
        if college_hist is not None and len(college_hist) > 0:
            college_z_cols = COLLEGE_Z_COLS.get(position_group, [])
            return college_hist, college_z_cols
    except (ImportError, FileNotFoundError):
        pass
    return pd.DataFrame(), []


def career_arc_section(player, league_parquet_path, z_score_cols, stat_labels=None,
                       id_col="player_id", name_col="player_display_name",
                       position_label="players", position_group=None):
    """Render the career arc chart below the radar chart.
    
    Args:
        player: Series — the currently selected player row
        league_parquet_path: Path to the league-wide parquet
        z_score_cols: list of z-score column names used for this position
        stat_labels: dict mapping z_col -> display name
        id_col: column name for player ID in the parquet
        name_col: column name for player display name
        position_label: e.g. "defensive ends" for captions
        position_group: e.g. "de", "qb" — used for college data lookup
    """
    if stat_labels is None:
        stat_labels = {}

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Career arc")

    # Load NFL data
    try:
        league_df = load_league_data(str(league_parquet_path))
    except FileNotFoundError:
        st.caption("Career arc data not available.")
        return

    player_id = player.get("player_id", player.get("gsis_id", None))
    player_name = player.get("player_display_name", player.get("player_name", player.get("full_name", None)))

    nfl_history = find_player_history(league_df, player_id, player_name, id_col=id_col, name_col=name_col)
    nfl_season_col = "season_year" if "season_year" in league_df.columns else "season"

    # Compute NFL composite scores
    if len(nfl_history) > 0:
        nfl_history = nfl_history.sort_values(nfl_season_col)
        nfl_history["composite_z"] = nfl_history.apply(
            lambda row: compute_composite_score(row, z_score_cols), axis=1
        )

    # Try to find college history
    pg = position_group
    if pg is None:
        # Infer from position_label
        pg_map = {"defensive ends": "de", "defensive tackles": "dt", "linebackers": "lb",
                  "cornerbacks": "cb", "safeties": "s", "quarterbacks": "qb",
                  "wide receivers": "wr", "tight ends": "te", "running backs": "rb",
                  "offensive linemen": "ol", "kickers": "k", "punters": "p"}
        pg = pg_map.get(position_label, None)

    college_history, college_z_cols = _find_college_history(player_name, pg)

    has_college = len(college_history) > 0
    has_nfl = len(nfl_history) > 0

    if not has_nfl and not has_college:
        st.caption(f"No career data found for {player_name}.")
        return

    # Compute college composite scores
    if has_college:
        college_season_col = "season" if "season" in college_history.columns else "season_year"
        college_history = college_history.sort_values(college_season_col)
        college_history["composite_z"] = college_history.apply(
            lambda row: compute_composite_score(row, college_z_cols), axis=1
        )

    # Build combined timeline
    seasons = []
    values = []
    teams = []
    is_college = []

    if has_college:
        for _, row in college_history.iterrows():
            s = row.get(college_season_col)
            seasons.append(int(s))
            values.append(row["composite_z"])
            teams.append(row.get("team", ""))
            is_college.append(True)

    if has_nfl:
        for _, row in nfl_history.iterrows():
            s = row.get(nfl_season_col)
            seasons.append(int(s))
            values.append(row["composite_z"])
            team_col = "recent_team" if "recent_team" in row.index else "team"
            teams.append(row.get(team_col, ""))
            is_college.append(False)

    if len(seasons) < 2:
        if len(seasons) == 1:
            level = "college" if is_college[0] else "NFL"
            st.caption(f"Only one season of data for {player_name} ({level}). Career arc requires 2+ seasons.")
        else:
            st.caption(f"No career data found for {player_name}.")
        return

    # Sort by season
    combined = sorted(zip(seasons, values, teams, is_college), key=lambda x: x[0])
    seasons, values, teams, is_college = zip(*combined)
    seasons, values, teams, is_college = list(seasons), list(values), list(teams), list(is_college)

    # Find draft year for divider
    draft_year = None
    if has_college and has_nfl:
        # Draft year = first NFL season
        nfl_seasons = [s for s, c in zip(seasons, is_college) if not c]
        college_seasons = [s for s, c in zip(seasons, is_college) if c]
        if nfl_seasons and college_seasons:
            draft_year = min(nfl_seasons)

    percentiles = [zscore_to_percentile(v) if pd.notna(v) else None for v in values]

    # Build chart
    fig = go.Figure()

    # League average line
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1,
                  annotation_text="League avg", annotation_position="bottom left",
                  annotation_font_size=10, annotation_font_color="#888")

    # Draft divider
    if draft_year:
        fig.add_vline(x=draft_year - 0.5, line_dash="dot", line_color="#666", line_width=2,
                      annotation_text="DRAFTED", annotation_position="top",
                      annotation_font_size=10, annotation_font_color="#666")

    # Color points
    colors = []
    for v, college in zip(values, is_college):
        if pd.isna(v):
            colors.append("#ccc")
        elif college:
            # College colors — gold tones
            if v >= 1.0:
                colors.append("#B8860B")  # dark gold
            elif v >= 0.0:
                colors.append("#DAA520")  # goldenrod
            elif v >= -1.0:
                colors.append("#CD853F")  # peru
            else:
                colors.append("#A0522D")  # sienna
        else:
            # NFL colors — blue tones
            if v >= 1.0:
                colors.append("#0076B6")  # elite
            elif v >= 0.0:
                colors.append("#4CAF50")  # above average
            elif v >= -1.0:
                colors.append("#FF9800")  # below average
            else:
                colors.append("#F44336")  # poor

    # Marker symbols — square for college, circle for NFL
    symbols = ["square" if c else "circle" for c in is_college]

    hover_text = []
    for s, v, p, t, c in zip(seasons, values, percentiles, teams, is_college):
        level = "College" if c else "NFL"
        pop = "FBS" if c else "NFL"
        if pd.notna(v):
            pct_str = f"top {100 - int(p)}%" if p and p >= 50 else f"bottom {int(p)}%" if p else "—"
            hover_text.append(f"<b>{s}</b> ({t}) — {level}<br>Composite: {v:+.2f}<br>{pct_str} of {pop}")
        else:
            hover_text.append(f"<b>{s}</b> ({t}) — {level}<br>No data")

    # Plot college trace
    college_idx = [i for i, c in enumerate(is_college) if c]
    nfl_idx = [i for i, c in enumerate(is_college) if not c]

    if college_idx:
        fig.add_trace(go.Scatter(
            x=[seasons[i] for i in college_idx],
            y=[values[i] for i in college_idx],
            mode="lines+markers",
            name="College",
            line=dict(color="#DAA520", width=3),
            marker=dict(size=12, color=[colors[i] for i in college_idx],
                        symbol="square",
                        line=dict(width=2, color="white")),
            hovertext=[hover_text[i] for i in college_idx],
            hoverinfo="text",
        ))

    if nfl_idx:
        fig.add_trace(go.Scatter(
            x=[seasons[i] for i in nfl_idx],
            y=[values[i] for i in nfl_idx],
            mode="lines+markers",
            name="NFL",
            line=dict(color="#0076B6", width=3),
            marker=dict(size=12, color=[colors[i] for i in nfl_idx],
                        symbol="circle",
                        line=dict(width=2, color="white")),
            hovertext=[hover_text[i] for i in nfl_idx],
            hoverinfo="text",
        ))

    # Connect college to NFL with dotted line if both exist
    if college_idx and nfl_idx:
        last_college = max(college_idx)
        first_nfl = min(nfl_idx)
        fig.add_trace(go.Scatter(
            x=[seasons[last_college], seasons[first_nfl]],
            y=[values[last_college], values[first_nfl]],
            mode="lines",
            line=dict(color="#888", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Shade regions
    all_valid = [v for v in values if pd.notna(v)]
    if all_valid:
        y_max = max(max(all_valid) + 0.5, 2.0)
        y_min = min(min(all_valid) - 0.5, -2.0)
        fig.add_hrect(y0=1.0, y1=y_max, fillcolor="rgba(0,118,182,0.05)", line_width=0)
        fig.add_hrect(y0=-1.0, y1=y_min, fillcolor="rgba(244,67,54,0.05)", line_width=0)

    fig.update_layout(
        xaxis=dict(
            title="Season",
            tickmode="array",
            tickvals=seasons,
            ticktext=[str(s) for s in seasons],
            gridcolor="#eee",
        ),
        yaxis=dict(
            title="Composite z-score",
            gridcolor="#eee",
            zeroline=True,
            zerolinecolor="#888",
            zerolinewidth=1,
        ),
        height=380,
        margin=dict(l=50, r=20, t=20, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=(has_college and has_nfl),
    )

    st.plotly_chart(fig, use_container_width=True)

    if has_college and has_nfl:
        st.caption(f"■ College (vs FBS {position_label}) · ● NFL (vs NFL {position_label}) · 0.00 = league average · Dotted line = draft transition")
    elif has_college:
        st.caption(f"Each point is one college season vs. all FBS {position_label}. 0.00 = league average.")
    else:
        st.caption(f"Each point is one NFL season vs. all NFL {position_label}. 0.00 = league average.")

"""
career_arc.py — Career arc chart for player detail section.
Place in repo root alongside lib_shared.py and team_selector.py.

Shows a player's composite z-score plotted year by year, with a toggle
to switch between our composite score and individual stat metrics.

Requires: league-wide parquet with season_year and player_id columns,
plus all z-score columns.
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
    # Try ID first (most reliable)
    if id_col in league_df.columns and pd.notna(player_id):
        history = league_df[league_df[id_col] == player_id].copy()
        if len(history) > 0:
            return history

    # Fall back to name match
    if name_col in league_df.columns and pd.notna(player_name):
        history = league_df[league_df[name_col] == player_name].copy()
        if len(history) > 0:
            return history

    # Try alternate name columns
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
    if weights:
        weighted = []
        for c in z_cols:
            if pd.notna(row[c]) and c in weights:
                weighted.append(row[c] * weights[c])
        return np.mean(weighted) if weighted else np.mean(values)
    return np.mean(values)


def career_arc_section(player, league_parquet_path, z_score_cols, stat_labels=None,
                       id_col="player_id", name_col="player_display_name",
                       position_label="players"):
    """Render the career arc chart below the radar chart.
    
    Args:
        player: Series — the currently selected player row
        league_parquet_path: Path to the league-wide parquet
        z_score_cols: list of z-score column names used for this position
        stat_labels: dict mapping z_col -> display name
        id_col: column name for player ID in the parquet
        name_col: column name for player display name
        position_label: e.g. "defensive ends" for captions
    """
    if stat_labels is None:
        stat_labels = {}

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Career arc")

    # Load league data and find player history
    try:
        league_df = load_league_data(str(league_parquet_path))
    except FileNotFoundError:
        st.caption("Career arc data not available.")
        return

    player_id = player.get("player_id", player.get("gsis_id", None))
    player_name = player.get("player_display_name", player.get("player_name", player.get("full_name", None)))

    history = find_player_history(league_df, player_id, player_name, id_col=id_col, name_col=name_col)

    if len(history) == 0:
        st.caption(f"No multi-season data found for {player_name}.")
        return

    # Determine season column
    season_col = "season_year" if "season_year" in history.columns else "season"
    history = history.sort_values(season_col)

    # Compute composite score for each season
    history["composite_z"] = history.apply(
        lambda row: compute_composite_score(row, z_score_cols), axis=1
    )

    seasons = history[season_col].tolist()
    if len(seasons) < 2:
        st.caption(f"Only one season of data for {player_name}. Career arc requires 2+ seasons.")
        return

    # Build the metric options for the toggle
    available_metrics = ["Composite score"]
    metric_data = {"Composite score": history["composite_z"].tolist()}

    for z_col in z_score_cols:
        if z_col in history.columns and history[z_col].notna().any():
            label = stat_labels.get(z_col, z_col.replace("_z", "").replace("_", " ").title())
            available_metrics.append(label)
            metric_data[label] = history[z_col].tolist()

    # Toggle
    selected_metric = st.selectbox(
        "Metric",
        options=available_metrics,
        index=0,
        key=f"career_arc_metric_{player_name}",
        label_visibility="collapsed",
    )

    values = metric_data[selected_metric]
    percentiles = [zscore_to_percentile(v) if pd.notna(v) else None for v in values]

    # Determine team for each season
    team_col = None
    for tc in ["recent_team", "team"]:
        if tc in history.columns:
            team_col = tc
            break
    teams = history[team_col].tolist() if team_col else [""] * len(seasons)

    # Build chart
    fig = go.Figure()

    # Add zero line (league average)
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1,
                  annotation_text="League avg", annotation_position="bottom left",
                  annotation_font_size=10, annotation_font_color="#888")

    # Color points by value
    colors = []
    for v in values:
        if pd.isna(v):
            colors.append("#ccc")
        elif v >= 1.0:
            colors.append("#0076B6")  # elite
        elif v >= 0.0:
            colors.append("#4CAF50")  # above average
        elif v >= -1.0:
            colors.append("#FF9800")  # below average
        else:
            colors.append("#F44336")  # poor

    hover_text = []
    for i, (s, v, p, t) in enumerate(zip(seasons, values, percentiles, teams)):
        if pd.notna(v):
            pct_str = f"top {100 - int(p)}%" if p and p >= 50 else f"bottom {int(p)}%" if p else "—"
            hover_text.append(f"<b>{int(s)}</b> ({t})<br>{selected_metric}: {v:+.2f}<br>{pct_str}")
        else:
            hover_text.append(f"<b>{int(s)}</b> ({t})<br>No data")

    fig.add_trace(go.Scatter(
        x=[int(s) for s in seasons],
        y=values,
        mode="lines+markers",
        line=dict(color="#0076B6", width=3),
        marker=dict(size=12, color=colors, line=dict(width=2, color="white")),
        hovertext=hover_text,
        hoverinfo="text",
    ))

    # Shade regions
    fig.add_hrect(y0=1.0, y1=max(max(v for v in values if pd.notna(v)) + 0.5, 2.0),
                  fillcolor="rgba(0,118,182,0.05)", line_width=0)
    fig.add_hrect(y0=-1.0, y1=min(min(v for v in values if pd.notna(v)) - 0.5, -2.0),
                  fillcolor="rgba(244,67,54,0.05)", line_width=0)

    fig.update_layout(
        xaxis=dict(
            title="Season",
            tickmode="array",
            tickvals=[int(s) for s in seasons],
            ticktext=[str(int(s)) for s in seasons],
            gridcolor="#eee",
        ),
        yaxis=dict(
            title=selected_metric,
            gridcolor="#eee",
            zeroline=True,
            zerolinecolor="#888",
            zerolinewidth=1,
        ),
        height=350,
        margin=dict(l=50, r=20, t=20, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Each point is one season's {selected_metric.lower()} vs. all {position_label} league-wide that year. 0.00 = league average. Toggle the dropdown to see individual stats over time.")

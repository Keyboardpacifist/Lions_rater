"""
career_arc.py — NFL career arc + college profile section.
Two separate charts: NFL line chart, then college profile with radar + college line chart.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from pathlib import Path
import polars as pl


def zscore_to_percentile(z):
    if pd.isna(z): return None
    return float(norm.cdf(z) * 100)


@st.cache_data
def load_league_data(parquet_path):
    return pl.read_parquet(parquet_path).to_pandas()


def find_player_history(league_df, player_id, player_name, id_col="player_id", name_col="player_display_name"):
    if id_col in league_df.columns and pd.notna(player_id):
        history = league_df[league_df[id_col] == player_id].copy()
        if len(history) > 0: return history
    if name_col in league_df.columns and pd.notna(player_name):
        history = league_df[league_df[name_col] == player_name].copy()
        if len(history) > 0: return history
    for col in ["player_name", "player_display_name", "player", "full_name"]:
        if col in league_df.columns:
            history = league_df[league_df[col] == player_name].copy()
            if len(history) > 0: return history
    return pd.DataFrame()


def compute_composite_score(row, stat_cols):
    z_cols = [c for c in stat_cols if c.endswith("_z") and c in row.index]
    values = [row[c] for c in z_cols if pd.notna(row[c])]
    return np.mean(values) if values else np.nan


def _find_college_history(player_name, position_group):
    try:
        from college_data import find_college_history, COLLEGE_Z_COLS
        college_hist = find_college_history(player_name, None, position_group=position_group)
        if college_hist is not None and len(college_hist) > 0:
            college_z_cols = COLLEGE_Z_COLS.get(position_group, [])
            return college_hist, college_z_cols
    except (ImportError, FileNotFoundError):
        pass
    return pd.DataFrame(), []


def _build_line_chart(seasons, values, teams, color, label, population_label):
    """Build a single line chart for either NFL or college data."""
    percentiles = [zscore_to_percentile(v) if pd.notna(v) else None for v in values]

    colors = []
    for v in values:
        if pd.isna(v): colors.append("#ccc")
        elif v >= 1.0: colors.append(color)
        elif v >= 0.0: colors.append("#4CAF50")
        elif v >= -1.0: colors.append("#FF9800")
        else: colors.append("#F44336")

    hover_text = []
    for s, v, p, t in zip(seasons, values, percentiles, teams):
        if pd.notna(v):
            pct_str = f"top {100 - int(p)}%" if p and p >= 50 else f"bottom {int(p)}%" if p else "—"
            hover_text.append(f"<b>{int(s)}</b> ({t})<br>Composite: {v:+.2f}<br>{pct_str} of {population_label}")
        else:
            hover_text.append(f"<b>{int(s)}</b> ({t})<br>No data")

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1,
                  annotation_text=f"{population_label} avg", annotation_position="bottom left",
                  annotation_font_size=10, annotation_font_color="#888")

    fig.add_trace(go.Scatter(
        x=[int(s) for s in seasons], y=values,
        mode="lines+markers", name=label,
        line=dict(color=color, width=3),
        marker=dict(size=12, color=colors, line=dict(width=2, color="white")),
        hovertext=hover_text, hoverinfo="text",
    ))

    all_valid = [v for v in values if pd.notna(v)]
    if all_valid:
        y_max = max(max(all_valid) + 0.5, 2.0)
        y_min = min(min(all_valid) - 0.5, -2.0)
        fig.add_hrect(y0=1.0, y1=y_max, fillcolor="rgba(0,118,182,0.05)", line_width=0)
        fig.add_hrect(y0=-1.0, y1=y_min, fillcolor="rgba(244,67,54,0.05)", line_width=0)

    fig.update_layout(
        xaxis=dict(title="Season", tickmode="array",
                   tickvals=[int(s) for s in seasons],
                   ticktext=[str(int(s)) for s in seasons], gridcolor="#eee"),
        yaxis=dict(title="Composite z-score", gridcolor="#eee",
                   zeroline=True, zerolinecolor="#888", zerolinewidth=1),
        height=320, margin=dict(l=50, r=20, t=20, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def career_arc_section(player, league_parquet_path, z_score_cols, stat_labels=None,
                       id_col="player_id", name_col="player_display_name",
                       position_label="players", position_group=None):
    if stat_labels is None: stat_labels = {}

    # ── Load NFL data ─────────────────────────────────────
    try:
        league_df = load_league_data(str(league_parquet_path))
    except FileNotFoundError:
        st.caption("Career arc data not available.")
        return

    player_id = player.get("player_id", player.get("gsis_id", None))
    player_name = player.get("player_display_name", player.get("player_name", player.get("full_name", None)))

    nfl_history = find_player_history(league_df, player_id, player_name, id_col=id_col, name_col=name_col)
    nfl_season_col = "season_year" if "season_year" in league_df.columns else "season"

    if len(nfl_history) > 0:
        nfl_history = nfl_history.sort_values(nfl_season_col)
        nfl_history["composite_z"] = nfl_history.apply(
            lambda row: compute_composite_score(row, z_score_cols), axis=1)

    # ── Infer position group ──────────────────────────────
    pg = position_group
    if pg is None:
        pg_map = {"defensive ends": "de", "defensive tackles": "dt", "linebackers": "lb",
                  "cornerbacks": "cb", "safeties": "s", "quarterbacks": "qb",
                  "wide receivers": "wr", "tight ends": "te", "running backs": "rb",
                  "offensive linemen": "ol", "kickers": "k", "punters": "p"}
        pg = pg_map.get(position_label, None)

    # ── Find college data ─────────────────────────────────
    college_history, college_z_cols = _find_college_history(player_name, pg)
    has_college = len(college_history) > 0
    has_nfl = len(nfl_history) > 0

    if not has_nfl and not has_college:
        st.caption(f"No career data found for {player_name}.")
        return

    if has_college:
        college_season_col = "season" if "season" in college_history.columns else "season_year"
        college_history = college_history.sort_values(college_season_col)
        college_history["composite_z"] = college_history.apply(
            lambda row: compute_composite_score(row, college_z_cols), axis=1)

    # ══════════════════════════════════════════════════════
    # NFL CAREER ARC
    # ══════════════════════════════════════════════════════
    if has_nfl:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### NFL career arc")

        nfl_seasons = nfl_history[nfl_season_col].tolist()
        nfl_values = nfl_history["composite_z"].tolist()
        team_col = "recent_team" if "recent_team" in nfl_history.columns else "team"
        nfl_teams = nfl_history[team_col].tolist()

        if len(nfl_seasons) >= 2:
            fig = _build_line_chart(nfl_seasons, nfl_values, nfl_teams,
                                    "#0076B6", "NFL", f"NFL {position_label}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Each point is one NFL season vs. all NFL {position_label}. 0.00 = league average.")
        elif len(nfl_seasons) == 1:
            score = nfl_values[0]
            if pd.notna(score):
                pct = zscore_to_percentile(score)
                sign = "+" if score >= 0 else ""
                pct_label = f"top {100 - int(pct)}%" if pct >= 50 else f"bottom {int(pct)}%"
                st.caption(f"One NFL season: {sign}{score:.2f} ({pct_label} of NFL {position_label})")
            else:
                st.caption(f"One NFL season — insufficient data for scoring.")

    # ══════════════════════════════════════════════════════
    # COLLEGE PROFILE
    # ══════════════════════════════════════════════════════
    if has_college:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### College profile")
        st.caption(f"Z-scored against all FBS {position_label} each season")

        college_seasons_list = sorted(college_history[college_season_col].unique())

        # Season selector
        if len(college_seasons_list) > 1:
            selected_college_season = st.selectbox(
                "College season",
                options=college_seasons_list,
                index=len(college_seasons_list) - 1,
                format_func=lambda s: f"{int(s)} — {college_history[college_history[college_season_col] == s]['team'].iloc[0]}",
                key=f"college_season_{player_name}",
            )
        else:
            selected_college_season = college_seasons_list[0]

        college_row = college_history[college_history[college_season_col] == selected_college_season].iloc[0]
        college_team = college_row.get("team", "")
        college_conf = college_row.get("conference", "")

        # ── Stat table + radar ────────────────────────────
        cc1, cc2 = st.columns([1, 1])

        with cc1:
            st.markdown(f"**{player_name}** — {int(selected_college_season)} {college_team}")
            if college_conf:
                st.caption(f"{college_conf}")

            try:
                from college_data import COLLEGE_Z_COLS, COLLEGE_STAT_LABELS
                cz_cols = COLLEGE_Z_COLS.get(pg, [])
            except ImportError:
                cz_cols = college_z_cols
                COLLEGE_STAT_LABELS = {}

            college_stat_rows = []
            for z_col in cz_cols:
                if z_col not in college_row.index: continue
                z = college_row.get(z_col)
                if pd.isna(z): continue
                raw_col = z_col.replace("_z", "")
                raw = college_row.get(raw_col)
                label = COLLEGE_STAT_LABELS.get(z_col, z_col.replace("_z", "").replace("_", " ").title())
                pct = zscore_to_percentile(z)
                pct_str = f"{int(pct)}th" if pct is not None else "—"
                college_stat_rows.append({
                    "Stat": label,
                    "Value": f"{raw:.2f}" if pd.notna(raw) else "—",
                    "Z-score": f"{z:+.2f}",
                    "Percentile": pct_str,
                })

            if college_stat_rows:
                st.dataframe(pd.DataFrame(college_stat_rows), use_container_width=True, hide_index=True)

            comp = compute_composite_score(college_row, college_z_cols)
            if pd.notna(comp):
                comp_pct = zscore_to_percentile(comp)
                sign = "+" if comp >= 0 else ""
                pct_label = f"top {100 - int(comp_pct)}%" if comp_pct >= 50 else f"bottom {int(comp_pct)}%"
                st.markdown(f"**College composite: {sign}{comp:.2f}** ({pct_label} of FBS {position_label})")

        with cc2:
            st.markdown(f"**College percentile profile** vs. FBS {position_label}")
            st.caption("50th = FBS average. Higher = better.")

            radar_axes, radar_values = [], []
            for z_col in cz_cols:
                if z_col not in college_row.index: continue
                z = college_row.get(z_col)
                if pd.isna(z): continue
                pct = zscore_to_percentile(z)
                label = COLLEGE_STAT_LABELS.get(z_col, z_col.replace("_z", "").replace("_", " ").title())
                radar_axes.append(label)
                radar_values.append(pct)

            if len(radar_axes) >= 3:
                radar_fig = go.Figure()
                radar_fig.add_trace(go.Scatterpolar(
                    r=radar_values + [radar_values[0]],
                    theta=radar_axes + [radar_axes[0]],
                    fill="toself",
                    fillcolor="rgba(218, 165, 32, 0.25)",
                    line=dict(color="rgba(184, 134, 11, 0.9)", width=2),
                    marker=dict(size=6, color="rgba(184, 134, 11, 1)"),
                    hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<extra></extra>",
                ))
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100],
                                        tickvals=[25, 50, 75, 100],
                                        ticktext=["25th", "50th", "75th", "100th"],
                                        tickfont=dict(size=9, color="#888"), gridcolor="#ddd"),
                        angularaxis=dict(tickfont=dict(size=11), gridcolor="#ddd"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    showlegend=False, margin=dict(l=60, r=60, t=20, b=20),
                    height=350, paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.caption("Not enough stats for radar chart.")

        # ── College career arc line chart ─────────────────
        college_seasons = [int(s) for s in college_history[college_season_col].tolist()]
        college_values = college_history["composite_z"].tolist()
        college_teams = college_history["team"].tolist()

        if len(college_seasons) >= 2:
            st.markdown("**College career arc**")
            fig = _build_line_chart(college_seasons, college_values, college_teams,
                                    "#B8860B", "College", f"FBS {position_label}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Each point is one college season vs. all FBS {position_label}. 0.00 = FBS average.")

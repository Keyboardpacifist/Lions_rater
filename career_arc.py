"""
career_arc.py — NFL career arc + college profile section.
Two separate charts: NFL line chart, then college profile with radar + college line chart.
"""
import json
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


@st.cache_data
def _load_team_colors():
    """Load team primary/secondary colors generated from nflverse.

    Returns a dict keyed by nflverse team abbr, with fields:
      {abbr: {"name": str, "primary": "#RRGGBB", "secondary": "#RRGGBB"}}
    Falls back to an empty dict if the file is missing.
    """
    path = Path(__file__).resolve().parent / "data" / "team_colors.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _team_color(team_abbr, fallback="#0076B6"):
    """Look up a team's primary color, falling back if unknown."""
    colors = _load_team_colors()
    return colors.get(team_abbr, {}).get("primary", fallback)


def _team_secondary(team_abbr, fallback="#888"):
    colors = _load_team_colors()
    return colors.get(team_abbr, {}).get("secondary", fallback)


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


def _compute_starter_benchmark(league_df, season_col, value_col,
                                snap_col="off_snaps", top_n=32):
    """For each season in `league_df`, mean of `value_col` across the
    top-N rows by snap count — a "starter-tier" reference for NFL.

    Returns a dict {season: benchmark_value}.
    """
    if (season_col not in league_df.columns
            or value_col not in league_df.columns
            or snap_col not in league_df.columns):
        return {}

    out = {}
    for season, group in league_df.groupby(season_col):
        ranked = group.sort_values(snap_col, ascending=False).head(top_n)
        vals = ranked[value_col].dropna()
        if len(vals) == 0:
            continue
        out[int(season)] = float(vals.mean())
    return out


def _compute_median_benchmark(league_df, season_col, value_col,
                                sort_col=None, top_n=None):
    """For each season, the MEDIAN of `value_col`.

    When `sort_col` + `top_n` are provided, the median is computed across
    only the top-N players by `sort_col` (e.g., top 130 by carries =
    "the starting pool"). Otherwise, median of all rows in the group.

    Median is more robust than mean for college populations with
    extreme variance between programs (Alabama → small school).
    """
    if season_col not in league_df.columns or value_col not in league_df.columns:
        return {}
    out = {}
    for season, group in league_df.groupby(season_col):
        if sort_col is not None and sort_col in group.columns and top_n is not None:
            group = group.sort_values(sort_col, ascending=False).head(top_n)
        vals = group[value_col].dropna()
        if len(vals) == 0:
            continue
        out[int(season)] = float(vals.median())
    return out


# Per-position volume column for picking the "starting pool" of college
# players. Top-N by this column ≈ "the player who starts at this position
# for an FBS program."
COLLEGE_VOLUME_COL = {
    "wr": "receptions_total",
    "te": "receptions_total",
    "rb": "carries_total",
    "qb": "pass_att",
    # Defensive positions all share the college_def_all_seasons.parquet,
    # which is filtered by pos_group on load (see college_data.load_college_parquet).
    # tackles_total is a universal starter proxy across all defensive groups.
    "de": "tackles_total",
    "dt": "tackles_total",
    "lb": "tackles_total",
    "cb": "tackles_total",
    "s":  "tackles_total",
}

# How many "starters" to take per season. ~130 FBS teams → roughly one
# starter per team per position. WRs have 2-3 starters per team; if we
# decide to widen the WR pool later, bump this.
COLLEGE_STARTER_TOP_N = 130


def _coverage_note(player_seasons, league_df, season_col, label):
    """Return a soft note when the player's earliest season equals the
    earliest season in the source data — i.e., the player's career
    likely extends before our coverage window.

    Returns None when the player's min season is comfortably inside the
    coverage range, so no note is needed.
    """
    if not player_seasons or season_col not in league_df.columns:
        return None
    try:
        player_min = int(min(s for s in player_seasons if s is not None))
        data_min = int(league_df[season_col].dropna().min())
    except (ValueError, TypeError):
        return None
    if player_min == data_min:
        return (
            f"ℹ️ {label} data in this app starts at {data_min}. "
            f"This player's career may extend earlier — those seasons "
            f"aren't shown."
        )
    return None


def _add_benchmark_trace(fig, benchmark_dict, label, season_anchor=0.5,
                          show_in_legend=True, color="#666"):
    """Overlay a dashed-diamond benchmark line on an existing figure.

    `benchmark_dict` is {season: y_value}. Markers are placed at
    x = season + season_anchor (0.5 centers them in the season slot
    when ticks are integer; 0 if x-axis is integer-aligned).

    Adds the trace in place; figure is returned for chaining.
    """
    if not benchmark_dict:
        return fig
    seasons = sorted(benchmark_dict.keys())
    xs = [s + season_anchor for s in seasons]
    ys = [benchmark_dict[s] for s in seasons]
    hover = [
        f"<b>{s} typical {label.lower()}</b><br>Composite: {benchmark_dict[s]:+.2f}"
        for s in seasons
    ]
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines+markers",
        line=dict(color=color, width=2, dash="dot"),
        marker=dict(size=8, color=color, symbol="diamond",
                    line=dict(width=1, color="white")),
        hovertext=hover, hoverinfo="text",
        name=label,
        showlegend=show_in_legend,
    ))
    return fig


def _build_per_stint_chart(history_df, value_col, season_col, team_col,
                            population_label, fallback_color="#0076B6",
                            league_df=None, benchmark_top_n=32):
    """Per-stint NFL career arc with team primary colors.

    Each row in history_df is a (player, team, season) stint with:
      - value_col: the y-value (composite z, or a single stat z)
      - season_col: the season year (int)
      - team_col: nflverse team abbreviation
      - first_week, last_week: optional, used to position markers within
        a season slot proportional to when the stint happened. Falls back
        to weeks 1..17 if absent (legacy data).

    Markers are positioned at x = season + (mid_week / 18). A trade
    transition between two stints in the same season is drawn as two
    line segments (each colored by its team) meeting at a small node
    placed at the trade boundary.
    """
    try:
        from team_selector import display_abbr as _disp
    except ImportError:
        _disp = lambda x: x

    df = history_df.copy()
    if season_col not in df.columns or team_col not in df.columns:
        return None
    if "first_week" not in df.columns:
        df["first_week"] = 1
    if "last_week" not in df.columns:
        df["last_week"] = 17
    df["first_week"] = df["first_week"].fillna(1)
    df["last_week"] = df["last_week"].fillna(17)
    df = df.sort_values([season_col, "first_week"]).reset_index(drop=True)

    # Build the list of points
    points = []
    for _, row in df.iterrows():
        season = int(row[season_col])
        first_w = float(row["first_week"])
        last_w = float(row["last_week"])
        x_mid = season + (first_w + last_w) / 2.0 / 18.0
        team = row[team_col]
        primary = _team_color(team, fallback=fallback_color)
        secondary = _team_secondary(team, fallback="#888")
        y = row[value_col]
        games = row.get("games") if "games" in row.index else None
        points.append(dict(
            x=x_mid, y=y, team=team, season=season,
            first_week=first_w, last_week=last_w,
            primary=primary, secondary=secondary, games=games,
        ))

    if not points:
        return None

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1,
                  annotation_text=f"Avg starter ({population_label})",
                  annotation_position="bottom left",
                  annotation_font_size=10, annotation_font_color="#888")

    # Connect consecutive points as line segments. Within-season
    # transitions (trades) get a color split at the trade boundary;
    # cross-season transitions get a single source-team segment.
    for i in range(1, len(points)):
        prev, curr = points[i - 1], points[i]
        if pd.isna(prev["y"]) or pd.isna(curr["y"]):
            continue
        same_season = prev["season"] == curr["season"]
        if same_season:
            # Trade boundary at the midpoint between prev's last week and curr's first week
            boundary_week = (prev["last_week"] + curr["first_week"]) / 2.0
            trade_x = prev["season"] + boundary_week / 18.0
            # Linear interpolation of y at the boundary
            span = curr["x"] - prev["x"]
            t = (trade_x - prev["x"]) / span if span else 0.5
            trade_y = prev["y"] + t * (curr["y"] - prev["y"])
            # Source-team segment up to boundary
            fig.add_trace(go.Scatter(
                x=[prev["x"], trade_x], y=[prev["y"], trade_y],
                mode="lines",
                line=dict(color=prev["primary"], width=3),
                showlegend=False, hoverinfo="skip",
            ))
            # Destination-team segment from boundary to next marker
            fig.add_trace(go.Scatter(
                x=[trade_x, curr["x"]], y=[trade_y, curr["y"]],
                mode="lines",
                line=dict(color=curr["primary"], width=3),
                showlegend=False, hoverinfo="skip",
            ))
            # Small transition node at the trade boundary
            fig.add_trace(go.Scatter(
                x=[trade_x], y=[trade_y],
                mode="markers",
                marker=dict(size=7, color="white",
                            line=dict(width=2, color="#666")),
                showlegend=False,
                hovertext=(
                    f"<b>Trade {prev['season']}</b><br>"
                    f"{_disp(prev['team'])} → {_disp(curr['team'])}"
                ),
                hoverinfo="text",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[prev["x"], curr["x"]], y=[prev["y"], curr["y"]],
                mode="lines",
                line=dict(color=prev["primary"], width=3),
                showlegend=False, hoverinfo="skip",
            ))

    # Stint markers
    hover_text = []
    for p in points:
        team_d = _disp(p["team"])
        games_str = f"{int(p['games'])}g" if p["games"] is not None and pd.notna(p["games"]) else "?g"
        wk_str = f"weeks {int(p['first_week'])}–{int(p['last_week'])}"
        if pd.notna(p["y"]):
            pct = zscore_to_percentile(p["y"])
            pct_str = (f"top {100 - int(pct)}%" if pct and pct >= 50
                       else f"bottom {int(pct)}%" if pct else "—")
            hover_text.append(
                f"<b>{p['season']} ({team_d})</b><br>"
                f"{games_str} · {wk_str}<br>"
                f"Composite: {p['y']:+.2f}<br>"
                f"{pct_str} of {population_label}"
            )
        else:
            hover_text.append(f"<b>{p['season']} ({team_d})</b><br>{games_str} · {wk_str}<br>No data")

    fig.add_trace(go.Scatter(
        x=[p["x"] for p in points],
        y=[p["y"] for p in points],
        mode="markers",
        marker=dict(
            size=14,
            color=[p["primary"] for p in points],
            line=dict(width=3, color=[p["secondary"] for p in points]),
        ),
        hovertext=hover_text, hoverinfo="text",
        showlegend=False,
    ))

    # Starter-tier benchmark line — mean of top-N players by snaps per season.
    # Population label is taken from the chart's `population_label` (e.g.,
    # "NFL running backs"), so the legend and hover read correctly per position.
    if league_df is not None:
        bench = _compute_starter_benchmark(
            league_df, season_col=season_col, value_col=value_col,
            top_n=benchmark_top_n,
        )
        # Strip the leading "NFL " from "NFL running backs" -> "running backs"
        pos_text = population_label.replace("NFL ", "", 1) if population_label.startswith("NFL ") else population_label
        seasons_in_career = sorted(set(int(p["season"]) for p in points))
        bench_xs, bench_ys, bench_hover = [], [], []
        for s in seasons_in_career:
            if s not in bench:
                continue
            bench_xs.append(s + 0.5)  # center of the season slot
            bench_ys.append(bench[s])
            bench_hover.append(
                f"<b>{s} starter benchmark</b><br>"
                f"Mean of top {benchmark_top_n} {pos_text} by snaps<br>"
                f"Composite: {bench[s]:+.2f}"
            )
        if bench_xs:
            fig.add_trace(go.Scatter(
                x=bench_xs, y=bench_ys,
                mode="lines+markers",
                line=dict(color="#666", width=2, dash="dot"),
                marker=dict(size=8, color="#666",
                            line=dict(width=1, color="white"),
                            symbol="diamond"),
                hovertext=bench_hover, hoverinfo="text",
                name=f"Top {benchmark_top_n} starter avg",
                showlegend=True,
            ))

    valid_y = [p["y"] for p in points if pd.notna(p["y"])]
    if valid_y:
        y_max = max(max(valid_y) + 0.5, 2.0)
        y_min = min(min(valid_y) - 0.5, -2.0)
        fig.add_hrect(y0=1.0, y1=y_max, fillcolor="rgba(0,118,182,0.05)", line_width=0)
        fig.add_hrect(y0=-1.0, y1=y_min, fillcolor="rgba(244,67,54,0.05)", line_width=0)

    seasons_unique = sorted(set(p["season"] for p in points))
    fig.update_layout(
        xaxis=dict(
            title="Season",
            tickmode="array",
            tickvals=[s + 0.5 for s in seasons_unique],
            ticktext=[str(s) for s in seasons_unique],
            gridcolor="#eee",
        ),
        yaxis=dict(title="Composite z-score", gridcolor="#eee",
                   zeroline=True, zerolinecolor="#888", zerolinewidth=1),
        height=320, margin=dict(l=50, r=20, t=20, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.7)", bordercolor="#ccc", borderwidth=1,
            font=dict(size=10),
        ),
    )
    return fig


_MULTI_METRIC_COLORS_NFL = [
    "#0076B6", "#1f77b4", "#2ca02c", "#d62728", "#9467bd",
    "#17becf", "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f",
]


def _build_multi_metric_nfl_chart(history_df, season_col, metric_columns,
                                    selected_metrics, population_label):
    """Multi-metric overlay of an NFL player's career arc.

    Each selected metric is drawn as its own colored line. Per-stint
    team coloring (used by the single-metric chart) is dropped here —
    line colors are reserved for distinguishing metrics, not teams.
    The typical-starter benchmark is also dropped because each metric
    has its own scale.
    """
    if history_df.empty or not selected_metrics:
        return None
    seasons = [int(s) for s in history_df[season_col].tolist()]
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1,
                  annotation_text=f"Avg starter ({population_label})",
                  annotation_position="bottom left",
                  annotation_font_size=10, annotation_font_color="#888")
    plotted = 0
    for i, metric in enumerate(selected_metrics):
        col = metric_columns.get(metric)
        if col is None or col not in history_df.columns:
            continue
        values = history_df[col].tolist()
        color = _MULTI_METRIC_COLORS_NFL[i % len(_MULTI_METRIC_COLORS_NFL)]
        hover = []
        for s, v in zip(seasons, values):
            if pd.notna(v):
                hover.append(f"<b>{s}</b><br>{metric}: {v:+.2f}")
            else:
                hover.append(f"<b>{s}</b><br>No data")
        fig.add_trace(go.Scatter(
            x=seasons, y=values, mode="lines+markers", name=metric,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color, line=dict(width=1.5, color="white")),
            hovertext=hover, hoverinfo="text",
        ))
        plotted += 1
    if plotted == 0:
        return None
    fig.update_layout(
        xaxis=dict(title="Season", tickmode="array", tickvals=seasons,
                   ticktext=[str(s) for s in seasons], gridcolor="#eee"),
        yaxis=dict(title="Z-score (vs. NFL avg)", gridcolor="#eee",
                   zeroline=True, zerolinecolor="#888", zerolinewidth=1),
        height=380, margin=dict(l=50, r=20, t=20, b=80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.18,
                    xanchor="center", x=0.5,
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="#ccc", borderwidth=1,
                    font=dict(size=10)),
    )
    return fig


def _render_nfl_career_chart(*, history_df, season_col, team_col, league_df,
                              z_score_cols, stat_labels, population_label,
                              selected_metrics, multi_metric, key_suffix):
    """Render the NFL career arc chart for one player.

    Single-metric mode → per-stint team-colored chart with starter
    benchmark. Multi-metric mode → overlay each metric as its own line.
    Returns the rendered Plotly figure (or None if no data).
    """
    nfl_metric_columns = {"Composite score": "composite_z"}
    for z_col in z_score_cols:
        if z_col in history_df.columns and history_df[z_col].notna().any():
            label = stat_labels.get(z_col, z_col.replace("_z", "").replace("_", " ").title())
            nfl_metric_columns[label] = z_col

    if multi_metric:
        return _build_multi_metric_nfl_chart(
            history_df, season_col=season_col,
            metric_columns=nfl_metric_columns,
            selected_metrics=selected_metrics,
            population_label=population_label,
        )

    value_col = nfl_metric_columns.get(selected_metrics[0], "composite_z")
    league_for_bench = league_df.copy()
    if value_col == "composite_z":
        league_for_bench["composite_z"] = league_for_bench.apply(
            lambda row: compute_composite_score(row, z_score_cols), axis=1)
    return _build_per_stint_chart(
        history_df, value_col=value_col,
        season_col=season_col, team_col=team_col,
        population_label=population_label,
        fallback_color="#0076B6",
        league_df=league_for_bench,
        benchmark_top_n=32,
    )


def _build_line_chart(seasons, values, teams, color, label, population_label):
    """Build a single line chart for either NFL or college data."""
    try:
        from team_selector import display_abbr as _disp
    except ImportError:
        _disp = lambda x: x  # college path doesn't need the NFL display map

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
        team_disp = _disp(t) if t else t
        if pd.notna(v):
            pct_str = f"top {100 - int(p)}%" if p and p >= 50 else f"bottom {int(p)}%" if p else "—"
            hover_text.append(f"<b>{int(s)}</b> ({team_disp})<br>Composite: {v:+.2f}<br>{pct_str} of {population_label}")
        else:
            hover_text.append(f"<b>{int(s)}</b> ({team_disp})<br>No data")

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="#888", line_width=1,
                  annotation_text=f"Avg starter ({population_label})", annotation_position="bottom left",
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
    # COMBINE / PRO DAY DATA
    # ══════════════════════════════════════════════════════
    COLLEGE_DATA_DIR = Path(__file__).resolve().parent / "data" / "college"

    @st.cache_data
    def _load_workout_file(path_str):
        return pd.read_parquet(path_str)

    def _find_workout(df, name, name_col="player_name"):
        if len(df) == 0 or not name: return None
        exact = df[df[name_col].str.lower() == name.lower()]
        if len(exact) >= 1: return exact.iloc[0]
        parts = name.split()
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            matches = df[
                (df[name_col].str.contains(last, na=False, case=False)) &
                (df[name_col].str.contains(first, na=False, case=False))
            ]
            if len(matches) > 0: return matches.iloc[0]
            # Removed the "single last-name match" fallback — it silently
            # attributed Brian O'Neill's combine to Eric O'Neill. Require
            # both first AND last name to match.
        return None

    comb = None
    source_label = "NFL Combine"
    combine_path = COLLEGE_DATA_DIR / "nfl_combine.parquet"
    workouts_path = COLLEGE_DATA_DIR / "nfl_all_workouts.parquet"

    if combine_path.exists():
        try:
            comb = _find_workout(_load_workout_file(str(combine_path)), player_name)
        except: pass

    if comb is None and workouts_path.exists():
        try:
            wk = _find_workout(_load_workout_file(str(workouts_path)), player_name)
            if wk is not None:
                comb = wk
                if pd.notna(wk.get("source")) and wk["source"] == "pro_day":
                    source_label = "Pro Day"
        except: pass

    if comb is not None:
        parts = []
        if pd.notna(comb.get("ht")): parts.append(f"Ht: {comb['ht']}")
        elif pd.notna(comb.get("height_in")):
            inches = int(comb["height_in"])
            parts.append(f"Ht: {inches // 12}-{inches % 12}")
        if pd.notna(comb.get("wt")): parts.append(f"Wt: {int(comb['wt'])}")
        elif pd.notna(comb.get("weight")): parts.append(f"Wt: {int(comb['weight'])}")
        if pd.notna(comb.get("forty")): parts.append(f"40: {comb['forty']:.2f}s")
        if pd.notna(comb.get("bench")): parts.append(f"Bench: {int(comb['bench'])}")
        if pd.notna(comb.get("vertical")): parts.append(f"Vert: {comb['vertical']}\"")
        if pd.notna(comb.get("broad_jump")): parts.append(f"Broad: {int(comb['broad_jump'])}\"")
        if pd.notna(comb.get("cone")): parts.append(f"3-cone: {comb['cone']:.2f}s")
        if pd.notna(comb.get("shuttle")): parts.append(f"Shuttle: {comb['shuttle']:.2f}s")
        if parts:
            source_icon = "🏋️" if source_label == "NFL Combine" else "🏟️"
            draft_parts = []
            if pd.notna(comb.get("draft_round")): draft_parts.append(f"Rd {int(comb['draft_round'])}")
            if pd.notna(comb.get("draft_ovr")): draft_parts.append(f"Pick #{int(comb['draft_ovr'])}")
            if pd.notna(comb.get("draft_team")): draft_parts.append(f"→ {comb['draft_team']}")
            draft_str = f" | Draft: {' '.join(draft_parts)}" if draft_parts else ""
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.caption(f"{source_icon} **{source_label}:** {' · '.join(parts)}{draft_str}")

    # ══════════════════════════════════════════════════════
    # NFL CAREER ARC
    # ══════════════════════════════════════════════════════
    if has_nfl:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### NFL career arc")

        # Soft note if player's earliest season touches our data floor
        nfl_note = _coverage_note(
            nfl_history[nfl_season_col].tolist(), league_df,
            season_col=nfl_season_col, label="NFL",
        )
        if nfl_note:
            st.caption(nfl_note)

        team_col = "recent_team" if "recent_team" in nfl_history.columns else "team"
        # Sort chronologically: by season, then by first_week within season
        sort_cols = [nfl_season_col]
        if "first_week" in nfl_history.columns:
            sort_cols.append("first_week")
        nfl_history = nfl_history.sort_values(sort_cols).reset_index(drop=True)
        n_stints = len(nfl_history)

        if n_stints >= 2:
            # Build metric options — each stint is one row
            nfl_metric_options = ["Composite score"]
            for z_col in z_score_cols:
                if z_col in nfl_history.columns and nfl_history[z_col].notna().any():
                    label = stat_labels.get(z_col, z_col.replace("_z", "").replace("_", " ").title())
                    nfl_metric_options.append(label)

            # ── Multi-metric toggle ──────────────────────
            nfl_multi_metric = st.checkbox(
                "📊 Show multiple metrics on this chart",
                key=f"nfl_career_multi_{player_name}",
                help=("Plot multiple metrics on the same chart, each in "
                      "a different color. Drops the per-stint team "
                      "coloring and the typical-starter benchmark "
                      "(each metric has its own scale)."),
            )

            if nfl_multi_metric:
                selected_nfl_metrics = st.pills(
                    "Metrics to overlay — click to add or remove lines",
                    options=nfl_metric_options,
                    default=["Composite score"],
                    selection_mode="multi",
                    key=f"nfl_career_pills_{player_name}",
                )
                if not selected_nfl_metrics:
                    st.caption("Click at least one metric pill above.")
                    selected_nfl_metrics = ["Composite score"]
                metric_caption_label = "selected metrics"
            else:
                selected_nfl_metric = st.selectbox(
                    "Metric",
                    options=nfl_metric_options,
                    index=0,
                    key=f"nfl_career_metric_{player_name}",
                    label_visibility="collapsed",
                )
                selected_nfl_metrics = [selected_nfl_metric]
                metric_caption_label = selected_nfl_metric.lower()

            # ── Compare to another player — render BEFORE the chart
            # so we can overlay the comparison player on the SAME
            # figure rather than building two stacked charts.
            nfl_compare = st.checkbox(
                "🔍 Compare to another NFL player",
                key=f"nfl_career_compare_{player_name}",
                help=("Overlay a second player's career arc on this "
                      "same chart for direct visual comparison."),
            )
            cmp_history = None
            compare_name_nfl = None
            if nfl_compare and name_col in league_df.columns:
                all_pos_players = sorted(set(
                    str(n) for n in league_df[name_col].dropna().unique()
                    if str(n).strip()
                ))
                default_cmp = next(
                    (p for p in all_pos_players if p != player_name),
                    all_pos_players[0] if all_pos_players else None)
                if default_cmp:
                    compare_name_nfl = st.selectbox(
                        f"Comparison {position_label[:-1].lower() if position_label.endswith('s') else position_label.lower()}",
                        options=all_pos_players,
                        index=all_pos_players.index(default_cmp),
                        key=f"nfl_career_compare_select_{player_name}",
                    )
                    if compare_name_nfl:
                        cmp_history = find_player_history(
                            league_df, None, compare_name_nfl,
                            id_col=id_col, name_col=name_col,
                        )
                        if len(cmp_history) > 0:
                            cmp_history = cmp_history.sort_values(
                                nfl_season_col).reset_index(drop=True)
                            cmp_history["composite_z"] = cmp_history.apply(
                                lambda row: compute_composite_score(
                                    row, z_score_cols), axis=1)
                        else:
                            cmp_history = None

            fig = _render_nfl_career_chart(
                history_df=nfl_history, season_col=nfl_season_col,
                team_col=team_col, league_df=league_df,
                z_score_cols=z_score_cols, stat_labels=stat_labels,
                population_label=f"NFL {position_label}",
                selected_metrics=selected_nfl_metrics,
                multi_metric=nfl_multi_metric,
                key_suffix=player_name,
            )

            # ── Overlay the comparison player as a second trace ──
            if (fig is not None and cmp_history is not None
                    and not cmp_history.empty and compare_name_nfl):
                # Resolve which y-column to plot for the second
                # player. Single-metric path mirrors what the primary
                # chart did at line ~504; multi-metric path adds one
                # trace per selected metric.
                COMPARE_COLOR = "#ff7f0e"
                metric_columns = {"Composite score": "composite_z"}
                for z_col in z_score_cols:
                    if (z_col in nfl_history.columns
                            and nfl_history[z_col].notna().any()):
                        label = stat_labels.get(
                            z_col,
                            z_col.replace("_z", "").replace("_", " ").title())
                        metric_columns[label] = z_col

                _seasons_cmp = cmp_history[nfl_season_col].astype(int).tolist()

                def _add_cmp_trace(value_col: str, name: str,
                                       dash: str = "dash") -> None:
                    if value_col not in cmp_history.columns:
                        return
                    yvals = cmp_history[value_col].tolist()
                    if not any(pd.notna(v) for v in yvals):
                        return
                    fig.add_trace(go.Scatter(
                        x=_seasons_cmp,
                        y=yvals,
                        mode="lines+markers",
                        name=name,
                        line=dict(color=COMPARE_COLOR, width=3,
                                  dash=dash),
                        marker=dict(size=11, color=COMPARE_COLOR,
                                    line=dict(width=2, color="white")),
                        hovertemplate=(
                            f"<b>{compare_name_nfl}</b><br>"
                            "Season: %{x}<br>%{y:+.2f}"
                            "<extra></extra>"
                        ),
                    ))

                if nfl_multi_metric:
                    for m in selected_nfl_metrics:
                        col = metric_columns.get(m)
                        if col:
                            _add_cmp_trace(col, f"{compare_name_nfl} · {m}")
                else:
                    primary_metric = (selected_nfl_metrics[0]
                                       if selected_nfl_metrics
                                       else "Composite score")
                    col = metric_columns.get(primary_metric, "composite_z")
                    _add_cmp_trace(col, compare_name_nfl)

                fig.update_layout(showlegend=True,
                                    legend=dict(orientation="h",
                                                yanchor="bottom",
                                                y=1.02, xanchor="left",
                                                x=0))

            if fig is not None:
                st.plotly_chart(fig, use_container_width=True,
                                  key=f"nfl_chart_main_{player_name}")
                n_traded_seasons = (
                    nfl_history.groupby(nfl_season_col).size().gt(1).sum()
                )
                if cmp_history is not None and compare_name_nfl:
                    st.caption(
                        f"**{player_name}** in team colors · "
                        f"**{compare_name_nfl}** overlaid in orange "
                        "(dashed). 0.00 = avg starter."
                    )
                elif nfl_multi_metric:
                    st.caption(
                        f"Each line is one metric across this player's NFL "
                        f"career. 0.00 = avg starter for that metric."
                    )
                elif n_traded_seasons:
                    st.caption(
                        f"Each marker is a (season, team) stint. "
                        f"Marker color = team primary; outline = team secondary. "
                        f"Trade transitions show as line color changes with a small node at the trade boundary. "
                        f"{n_traded_seasons} season(s) included a mid-season trade."
                    )
                else:
                    st.caption(
                        f"Each marker is one NFL season's {metric_caption_label} vs. all NFL {position_label}. "
                        f"Marker color = team primary; outline = team secondary. 0.00 = avg starter."
                    )
        elif n_stints == 1:
            nfl_values = nfl_history["composite_z"].tolist()
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

        # Soft note if the player's earliest college season equals our
        # data floor (e.g., Goff at Cal — our college data starts 2014,
        # his freshman 2013 season isn't captured).
        try:
            from college_data import COLLEGE_PARQUET_MAP, COLLEGE_DATA_DIR as _CDD
            _full_path = Path(_CDD) / COLLEGE_PARQUET_MAP.get(pg, "")
            if _full_path.exists():
                _full_college = pl.read_parquet(str(_full_path)).to_pandas()
                _college_note = _coverage_note(
                    college_history[college_season_col].tolist(),
                    _full_college,
                    season_col=college_season_col,
                    label="College",
                )
                if _college_note:
                    st.caption(_college_note)
        except (ImportError, FileNotFoundError, KeyError):
            pass

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
            st.caption("50th = avg starter. Higher = better. (Z-scores baselined on starters at this position.)")

            radar_axes, radar_values, radar_z_cols_used = [], [], []
            for z_col in cz_cols:
                if z_col not in college_row.index: continue
                z = college_row.get(z_col)
                if pd.isna(z): continue
                pct = zscore_to_percentile(z)
                label = COLLEGE_STAT_LABELS.get(z_col, z_col.replace("_z", "").replace("_", " ").title())
                radar_axes.append(label)
                radar_values.append(pct)
                radar_z_cols_used.append(z_col)

            # Build the "typical starting <position>" benchmark for THIS season
            # using the same college parquet as the line chart.
            radar_bench_pcts = []
            radar_bench_raws = []  # raw stat values for hover
            try:
                from college_data import COLLEGE_PARQUET_MAP, COLLEGE_DATA_DIR as _CDD
                full_path = Path(_CDD) / COLLEGE_PARQUET_MAP.get(pg, "")
                if full_path.exists() and len(radar_z_cols_used) >= 3:
                    full_college = pl.read_parquet(str(full_path)).to_pandas()
                    season_pool = full_college[full_college[college_season_col] == selected_college_season]
                    sort_col = COLLEGE_VOLUME_COL.get(pg)
                    if sort_col and sort_col in season_pool.columns:
                        starters = season_pool.sort_values(sort_col, ascending=False).head(COLLEGE_STARTER_TOP_N)
                    else:
                        starters = season_pool
                    for z_col in radar_z_cols_used:
                        if z_col in starters.columns:
                            mean_z = starters[z_col].dropna().median()
                            radar_bench_pcts.append(zscore_to_percentile(mean_z) if pd.notna(mean_z) else None)
                            raw_col = z_col.replace("_z", "")
                            if raw_col in starters.columns and starters[raw_col].notna().any():
                                radar_bench_raws.append(float(starters[raw_col].median()))
                            else:
                                radar_bench_raws.append(None)
                        else:
                            radar_bench_pcts.append(None)
                            radar_bench_raws.append(None)
            except (ImportError, FileNotFoundError, KeyError):
                pass

            if len(radar_axes) >= 3:
                radar_fig = go.Figure()
                # Player polygon FIRST so the benchmark layers on top
                radar_fig.add_trace(go.Scatterpolar(
                    r=radar_values + [radar_values[0]],
                    theta=radar_axes + [radar_axes[0]],
                    fill="toself",
                    fillcolor="rgba(218, 165, 32, 0.25)",
                    line=dict(color="rgba(184, 134, 11, 0.9)", width=2),
                    marker=dict(size=6, color="rgba(184, 134, 11, 1)"),
                    name="This player",
                    hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<extra></extra>",
                ))

                bench_label = (
                    f"Typical starting {position_label[:-1]}"
                    if position_label.endswith("s") else f"Typical starting {position_label}"
                )
                if any(p is not None for p in radar_bench_pcts):
                    bv_clean = [p if p is not None else 50 for p in radar_bench_pcts]
                    bench_hover = []
                    for ax, pct, raw in zip(radar_axes, bv_clean, radar_bench_raws):
                        raw_str = f"median: {raw:.2f} · " if raw is not None else ""
                        bench_hover.append(
                            f"<b>{ax}</b><br>{bench_label}<br>{raw_str}{pct:.0f}th percentile"
                        )
                    bench_hover.append(bench_hover[0])
                    radar_fig.add_trace(go.Scatterpolar(
                        r=bv_clean + [bv_clean[0]],
                        theta=radar_axes + [radar_axes[0]],
                        mode="lines+markers",
                        line=dict(color="rgba(102, 102, 102, 0.9)", width=2, dash="dot"),
                        marker=dict(size=10, color="rgba(102, 102, 102, 0.95)",
                                    symbol="diamond", line=dict(width=2, color="white")),
                        name=bench_label,
                        hovertext=bench_hover,
                        hoverinfo="text",
                    ))

                # Legend below the chart (horizontal) so it doesn't cover
                # the angular axis labels — was unreadable on phones.
                radar_fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100],
                                        tickvals=[25, 50, 75, 100],
                                        ticktext=["25th", "50th", "75th", "100th"],
                                        tickfont=dict(size=9, color="#888"), gridcolor="#ddd"),
                        angularaxis=dict(tickfont=dict(size=11), gridcolor="#ddd"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    showlegend=any(p is not None for p in radar_bench_pcts),
                    legend=dict(orientation="h", yanchor="top", y=-0.05,
                                xanchor="center", x=0.5,
                                bgcolor="rgba(255,255,255,0)", bordercolor="rgba(0,0,0,0)",
                                font=dict(size=10)),
                    margin=dict(l=60, r=60, t=20, b=70),
                    height=400, paper_bgcolor="rgba(0,0,0,0)",
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

            # Build metric options
            col_metric_options = ["Composite score"]
            col_metric_data = {"Composite score": college_values}

            for z_col in college_z_cols:
                if z_col in college_history.columns and college_history[z_col].notna().any():
                    label = COLLEGE_STAT_LABELS.get(z_col, z_col.replace("_z", "").replace("_", " ").title())
                    col_metric_options.append(label)
                    col_metric_data[label] = college_history[z_col].tolist()

            selected_col_metric = st.selectbox(
                "College metric",
                options=col_metric_options,
                index=0,
                key=f"college_career_metric_{player_name}",
                label_visibility="collapsed",
            )

            chart_values = col_metric_data[selected_col_metric]
            fig = _build_line_chart(college_seasons, chart_values, college_teams,
                                    "#B8860B", "College", f"FBS {position_label}")

            # "Typical starter" benchmark — median of the top-N players by
            # the position's volume column (carries for RB, receptions for
            # WR/TE, pass_att for QB). Filtering to top-N first weeds out
            # walk-ons and third-stringers; median resists outliers within
            # that pool.
            value_col_for_bench = "composite_z"
            if selected_col_metric != "Composite score":
                for z_col in college_z_cols:
                    label = COLLEGE_STAT_LABELS.get(
                        z_col, z_col.replace("_z", "").replace("_", " ").title())
                    if label == selected_col_metric:
                        value_col_for_bench = z_col
                        break

            try:
                from college_data import COLLEGE_PARQUET_MAP, COLLEGE_DATA_DIR as _CDD
                full_path = Path(_CDD) / COLLEGE_PARQUET_MAP.get(pg, "")
                if full_path.exists():
                    full_college = pl.read_parquet(str(full_path)).to_pandas()
                    if value_col_for_bench == "composite_z":
                        full_college["composite_z"] = full_college.apply(
                            lambda row: compute_composite_score(row, college_z_cols), axis=1)
                    bench = _compute_median_benchmark(
                        full_college,
                        season_col=college_season_col,
                        value_col=value_col_for_bench,
                        sort_col=COLLEGE_VOLUME_COL.get(pg),
                        top_n=COLLEGE_STARTER_TOP_N,
                    )
                else:
                    bench = {}
            except (ImportError, FileNotFoundError, KeyError):
                bench = {}

            seasons_in_career = set(college_seasons)
            bench = {s: v for s, v in bench.items() if s in seasons_in_career}
            starter_label = f"Typical starting {position_label.rstrip('s').rstrip(' ').lower()}"
            if position_label.endswith("s"):
                # "running backs" -> "running back"
                starter_label = f"Typical starting {position_label[:-1]}"
            if bench:
                _add_benchmark_trace(fig, bench, label=starter_label, season_anchor=0)
                fig.update_layout(showlegend=True,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                                bgcolor="rgba(255,255,255,0.7)",
                                bordercolor="#ccc", borderwidth=1, font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)
            cap_extra = (
                f" · Dashed gray = {starter_label.lower()} (median of top {COLLEGE_STARTER_TOP_N} by volume per season)."
                if bench else ""
            )
            st.caption(f"Each point is one college season's {selected_col_metric.lower()} vs. all FBS {position_label}. 0.00 = avg starter.{cap_extra}")

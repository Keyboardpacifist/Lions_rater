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
    """For each season in `league_df`, compute the mean of `value_col`
    across the top-N rows (by snap count) — a "starter-tier" reference.

    Returns a dict {season: benchmark_value}. Skips seasons that don't
    have enough rows or have NaN values throughout.
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
                  annotation_text=f"{population_label} avg",
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

    # Starter-tier benchmark line — mean of top-N WRs (by snaps) per season.
    # Spans the same season range as the player's career so the comparison is
    # always against the relevant years.
    if league_df is not None:
        bench = _compute_starter_benchmark(
            league_df, season_col=season_col, value_col=value_col,
            top_n=benchmark_top_n,
        )
        seasons_in_career = sorted(set(int(p["season"]) for p in points))
        bench_xs, bench_ys, bench_hover = [], [], []
        for s in seasons_in_career:
            if s not in bench:
                continue
            bench_xs.append(s + 0.5)  # center of the season slot
            bench_ys.append(bench[s])
            bench_hover.append(
                f"<b>{s} starter benchmark</b><br>"
                f"Mean of top {benchmark_top_n} WRs by snaps<br>"
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
            last_only = df[df[name_col].str.contains(last, na=False, case=False)]
            if len(last_only) == 1: return last_only.iloc[0]
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
            nfl_metric_columns = {"Composite score": "composite_z"}

            for z_col in z_score_cols:
                if z_col in nfl_history.columns and nfl_history[z_col].notna().any():
                    label = stat_labels.get(z_col, z_col.replace("_z", "").replace("_", " ").title())
                    nfl_metric_options.append(label)
                    nfl_metric_columns[label] = z_col

            selected_nfl_metric = st.selectbox(
                "Metric",
                options=nfl_metric_options,
                index=0,
                key=f"nfl_career_metric_{player_name}",
                label_visibility="collapsed",
            )

            value_col = nfl_metric_columns[selected_nfl_metric]
            # For the benchmark line, the league_df needs a column matching the
            # selected metric. composite_z is computed per-row above on
            # nfl_history but not on the full league_df — recompute it on the fly.
            league_for_bench = league_df.copy()
            if value_col == "composite_z":
                league_for_bench["composite_z"] = league_for_bench.apply(
                    lambda row: compute_composite_score(row, z_score_cols), axis=1)
            fig = _build_per_stint_chart(
                nfl_history, value_col=value_col,
                season_col=nfl_season_col, team_col=team_col,
                population_label=f"NFL {position_label}",
                fallback_color="#0076B6",
                league_df=league_for_bench,
                benchmark_top_n=32,
            )
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                n_traded_seasons = (
                    nfl_history.groupby(nfl_season_col).size().gt(1).sum()
                )
                if n_traded_seasons:
                    st.caption(
                        f"Each marker is a (season, team) stint. "
                        f"Marker color = team primary; outline = team secondary. "
                        f"Trade transitions show as line color changes with a small node at the trade boundary. "
                        f"{n_traded_seasons} season(s) included a mid-season trade."
                    )
                else:
                    st.caption(
                        f"Each marker is one NFL season's {selected_nfl_metric.lower()} vs. all NFL {position_label}. "
                        f"Marker color = team primary; outline = team secondary. 0.00 = league average."
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
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Each point is one college season's {selected_col_metric.lower()} vs. all FBS {position_label}. 0.00 = FBS average.")

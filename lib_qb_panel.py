"""QB panel splits — pressure, situational, competition, throw map.

All built from the per-dropback feed (`data/qb_dropbacks.parquet`).
Used by `pages/QB.py` to surface what the season-aggregate parquet
can't: contextual performance.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_qb_dropbacks() -> pd.DataFrame:
    return pd.read_parquet(_DATA / "qb_dropbacks.parquet")


@st.cache_data
def load_team_pass_def_quality() -> pd.DataFrame:
    return pd.read_parquet(_DATA / "team_pass_def_quality.parquet")


@st.cache_data
def load_qb_ngs() -> pd.DataFrame:
    path = _DATA / "qb_ngs_seasons.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data
def get_qb_peers(season: int | None = None,
                   min_attempts: int = 100,
                   exclude_player_id: str | None = None) -> list[dict]:
    """Build the comparison picker's option list — every QB with
    a meaningful sample. For career view (season=None) we offer
    season-specific entries so users can compare e.g. "2024 Mahomes"
    vs. "2018 Mahomes" or another QB in any year.
    """
    db = load_qb_dropbacks()
    db = db[db["pass_attempt"] == 1]
    if season is not None:
        db = db[db["season"] == season]
    counts = (
        db.groupby(["passer_player_id", "passer_player_name", "season"])
        .size()
        .reset_index(name="att")
    )
    counts = counts[counts["att"] >= min_attempts]
    if exclude_player_id and season is not None:
        counts = counts[counts["passer_player_id"] != exclude_player_id]
    counts = counts.sort_values(["season", "att"], ascending=[False, False])
    return [
        {
            "label": f"{r.passer_player_name} — {int(r.season)} ({int(r.att)} att)",
            "player_id": r.passer_player_id,
            "season": int(r.season),
        }
        for r in counts.itertuples()
    ]


@st.cache_data
def _league_pressure_avg(season: int | None) -> dict:
    db = load_qb_dropbacks()
    if season is not None:
        db = db[db["season"] == season]
    pressured = db[db["is_pressured"]]
    clean = db[~db["is_pressured"]]
    return {
        "clean_epa": float(clean["epa"].mean()) if len(clean) else 0.0,
        "pressured_epa": float(pressured["epa"].mean()) if len(pressured) else 0.0,
        "pressure_rate": float(db["is_pressured"].mean()) if len(db) else 0.0,
    }


def _filter_player(player_id: str, season: int | None) -> pd.DataFrame:
    db = load_qb_dropbacks()
    plays = db[db["passer_player_id"] == player_id]
    if season is not None:
        plays = plays[plays["season"] == season]
    return plays


def render_pressure_split(player_id: str, player_name: str, *,
                            season: int | None = None,
                            theme: dict | None = None) -> None:
    """Bucket 7 — Under pressure.

    Two-bar EPA comparison (clean vs. pressured) with league-avg
    benchmarks, plus a detail row of completion/sack rates.
    """
    plays = _filter_player(player_id, season)
    if plays.empty or len(plays) < 30:
        st.caption(f"_Not enough dropbacks ({len(plays)}) to compute pressure splits._")
        return

    pressured = plays[plays["is_pressured"]]
    clean = plays[~plays["is_pressured"]]

    qb_clean = float(clean["epa"].mean()) if len(clean) else 0.0
    qb_pressed = float(pressured["epa"].mean()) if len(pressured) else 0.0
    league = _league_pressure_avg(season)

    primary = (theme or {}).get("primary", "#1F77B4")
    pressed_color = (theme or {}).get("secondary", "#D62728")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=player_name,
        x=["Clean pocket", "Pressured"],
        y=[qb_clean, qb_pressed],
        marker_color=[primary, pressed_color],
        text=[f"{qb_clean:+.2f}", f"{qb_pressed:+.2f}"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:+.3f} EPA/dropback<br>"
                      "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=["Clean pocket", "Pressured"],
        y=[league["clean_epa"], league["pressured_epa"]],
        mode="markers",
        marker=dict(symbol="line-ew", size=60, color="#999",
                    line=dict(width=3, color="#999")),
        name="League avg",
        hovertemplate="League avg: %{y:+.3f} EPA<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="EPA per dropback",
        height=360,
        margin=dict(l=40, r=20, t=30, b=40),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(255,255,255,0.7)"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_hline(y=0, line_color="#ccc", line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pressure rate",
               f"{plays['is_pressured'].mean()*100:.1f}%",
               help="% of his dropbacks where he got hit. League avg: "
                    f"{league['pressure_rate']*100:.1f}%")
    c2.metric("Clean comp %",
               f"{clean['complete_pass'].mean()*100:.1f}%" if len(clean) else "—")
    c3.metric("Pressured comp %",
               f"{pressured['complete_pass'].mean()*100:.1f}%" if len(pressured) else "—")
    sack_rate = float(pressured["sack"].mean()) if len(pressured) else 0.0
    c4.metric("Sack-when-pressured",
               f"{sack_rate*100:.1f}%",
               help="When he got hit, did he get sacked? Lower = better")

    # Narrative line: how does his clean→pressured drop compare to league?
    qb_drop = qb_clean - qb_pressed
    lg_drop = league["clean_epa"] - league["pressured_epa"]
    delta = qb_drop - lg_drop
    if delta < -0.05:
        st.success(
            f"**Holds up under pressure.** EPA drops only "
            f"{qb_drop:.2f}/dropback when hit — league avg drop is "
            f"{lg_drop:.2f}, so he loses {abs(delta):.2f} less than typical."
        )
    elif delta > 0.05:
        st.warning(
            f"**Struggles under pressure.** EPA drops "
            f"{qb_drop:.2f}/dropback when hit — league avg drop is only "
            f"{lg_drop:.2f}, so he loses {delta:.2f} more than typical."
        )
    else:
        st.info(
            f"**Average under pressure.** EPA drop of {qb_drop:.2f}/dropback "
            f"is roughly league-typical ({lg_drop:.2f})."
        )


def render_competition_split(player_id: str, player_name: str, *,
                                season: int | None = None,
                                theme: dict | None = None) -> None:
    """Bucket 8 — Elite vs. weak competition.

    Splits the QB's plays by which quartile of pass D they faced
    (1=elite, 4=bad), shows aggregate EPA & completion % per quartile,
    and renders a per-game scatter (opp D quality vs. QB EPA) that
    reveals whether he rises or fades against good defenses.
    """
    plays = _filter_player(player_id, season)
    if plays.empty or len(plays) < 50:
        st.caption(f"_Not enough dropbacks ({len(plays)}) to compute "
                    f"competition splits._")
        return

    def_quality = load_team_pass_def_quality()
    # Merge: opponent's pass D quality for each play
    plays = plays.merge(
        def_quality[["team", "season", "pass_epa_allowed_per_play",
                     "epa_rank", "quality_quartile"]],
        left_on=["defteam", "season"],
        right_on=["team", "season"],
        how="left",
    )
    plays = plays.dropna(subset=["quality_quartile"])
    if plays.empty:
        st.caption("_No matched defenses for these plays._")
        return

    # Aggregate by quartile
    by_q = (
        plays.groupby("quality_quartile")
        .agg(
            n_plays=("epa", "size"),
            epa=("epa", "mean"),
            comp_pct=("complete_pass", "mean"),
            int_rate=("interception", "mean"),
            sack_rate=("sack", "mean"),
        )
        .reset_index()
    )
    quartile_labels = {1: "Elite (top 25%)", 2: "Above avg", 3: "Below avg",
                        4: "Bad (bottom 25%)"}
    by_q["label"] = by_q["quality_quartile"].map(quartile_labels)

    # ── Bar chart: EPA per dropback by quartile ──
    primary = (theme or {}).get("primary", "#1F77B4")
    secondary = (theme or {}).get("secondary", "#D62728")
    bar_colors = [secondary, "#aa6e7e", "#7e9eaa", primary]  # 1→elite (red), 4→bad (primary)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_q["label"],
        y=by_q["epa"],
        marker_color=[bar_colors[int(q) - 1] for q in by_q["quality_quartile"]],
        text=[f"{e:+.2f}" for e in by_q["epa"]],
        textposition="outside",
        customdata=by_q[["n_plays", "comp_pct"]].values,
        hovertemplate="<b>%{x}</b><br>EPA/play: %{y:+.3f}<br>"
                      "Plays: %{customdata[0]}<br>"
                      "Comp %: %{customdata[1]:.1%}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0, line_color="#ccc", line_width=1)
    fig.update_layout(
        yaxis_title="EPA per dropback",
        xaxis_title="Opposing pass defense quality",
        height=340,
        margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-game scatter: opp D quality on x, QB EPA on y ──
    per_game = (
        plays.groupby(["game_id", "defteam", "season",
                        "pass_epa_allowed_per_play"])
        .agg(qb_epa=("epa", "mean"), n_plays=("epa", "size"))
        .reset_index()
    )
    per_game = per_game[per_game["n_plays"] >= 10]

    if len(per_game) >= 5:
        # Trendline coefficients
        import numpy as np
        x = per_game["pass_epa_allowed_per_play"].astype(float)
        y = per_game["qb_epa"].astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        # If slope > 0: as opp D gets worse (higher EPA allowed), QB EPA
        # rises → he beats up bad defenses. If slope < 0: he rises against
        # good D (the rare elite-clutch profile).
        x_range = np.linspace(x.min(), x.max(), 20)
        y_fit = slope * x_range + intercept

        scat = go.Figure()
        scat.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(size=8, color=primary, opacity=0.7,
                        line=dict(width=1, color="white")),
            text=per_game["defteam"] + " " + per_game["season"].astype(str),
            hovertemplate="<b>vs. %{text}</b><br>"
                          "Opp pass D allowed: %{x:+.3f} EPA/play<br>"
                          "QB EPA: %{y:+.3f}<extra></extra>",
            name="Game",
        ))
        scat.add_trace(go.Scatter(
            x=x_range, y=y_fit,
            mode="lines",
            line=dict(color="#888", width=2, dash="dot"),
            name="Trendline",
            hoverinfo="skip",
        ))
        scat.add_hline(y=0, line_color="#ccc", line_width=1)
        scat.add_vline(x=0, line_color="#ccc", line_width=1)
        scat.update_layout(
            xaxis_title="Opponent pass EPA allowed/play  (← elite D    bad D →)",
            yaxis_title="QB EPA in this game",
            height=380,
            margin=dict(l=40, r=20, t=30, b=50),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(scat, use_container_width=True)

        # Narrative line about slope
        if slope > 1.5:
            st.warning(
                f"**Stat-padder profile.** Slope of {slope:.2f} — "
                f"clearly feasts on bad defenses, struggles vs. elite. "
                f"The kind of QB whose box-score average outruns his "
                f"playoff readiness."
            )
        elif slope > 0.5:
            st.info(
                f"**Plays to opponent quality.** Slope of {slope:.2f} — "
                f"performs better vs. weaker defenses, as most QBs do, "
                f"but not extreme."
            )
        elif slope > -0.5:
            st.success(
                f"**Steady against any defense.** Slope of {slope:.2f} "
                f"(near zero) — plays at his level regardless of opponent. "
                f"The mark of a real dog."
            )
        else:
            st.success(
                f"**Rises against elite D.** Slope of {slope:.2f} — "
                f"counterintuitively *better* vs. top defenses. Rare "
                f"clutch profile."
            )

    # ── Detail row ──
    by_q_display = by_q.copy()
    by_q_display["EPA/play"] = by_q_display["epa"].apply(lambda v: f"{v:+.3f}")
    by_q_display["Comp %"] = by_q_display["comp_pct"].apply(lambda v: f"{v*100:.1f}%")
    by_q_display["INT rate"] = by_q_display["int_rate"].apply(lambda v: f"{v*100:.2f}%")
    by_q_display["Sack rate"] = by_q_display["sack_rate"].apply(lambda v: f"{v*100:.1f}%")
    by_q_display = by_q_display.rename(columns={"label": "Defense", "n_plays": "Plays"})
    st.dataframe(
        by_q_display[["Defense", "Plays", "EPA/play", "Comp %", "INT rate", "Sack rate"]],
        use_container_width=True, hide_index=True,
    )


# ── Throw-map zone definitions ────────────────────────────────────
_DEPTH_BUCKETS = [
    ("Deep", 20, 99),         # 20+ air yards
    ("Intermediate", 10, 20), # 10-19
    ("Short", -99, 10),       # <10 (negative for screens / behind LOS)
]
_LOCATIONS = ["left", "middle", "right"]
_LOC_LABELS = {"left": "Left", "middle": "Middle", "right": "Right"}


@st.cache_data
def _league_throw_map(season: int | None) -> pd.DataFrame:
    """Per-zone league averages — used for the Δ vs. league cell text."""
    db = load_qb_dropbacks()
    db = db[(db["pass_attempt"] == 1) & db["air_yards"].notna()
            & db["pass_location"].notna()]
    if season is not None:
        db = db[db["season"] == season]
    rows = []
    for depth_label, lo, hi in _DEPTH_BUCKETS:
        for loc in _LOCATIONS:
            zone = db[(db["air_yards"] >= lo) & (db["air_yards"] < hi)
                      & (db["pass_location"] == loc)]
            if len(zone) >= 50:
                rows.append({
                    "depth": depth_label, "location": loc,
                    "league_epa": float(zone["epa"].mean()),
                    "league_comp_pct": float(zone["complete_pass"].mean()),
                })
    return pd.DataFrame(rows)


_COVERAGE_OPTIONS = [
    ("COVER_0", "Cover 0 (all-out blitz)"),
    ("COVER_1", "Cover 1 (man w/ deep safety)"),
    ("COVER_2", "Cover 2 (2 deep zone)"),
    ("2_MAN", "Cover 2 Man"),
    ("COVER_3", "Cover 3 (3 deep zone)"),
    ("COVER_4", "Cover 4 (quarters)"),
    ("COVER_6", "Cover 6 (split field)"),
    ("COVER_9", "Cover 9 / palms"),
    ("COMBO", "Combo coverage"),
]
_COVERAGE_LABELS = dict(_COVERAGE_OPTIONS)
_PERSONNEL_OPTIONS = ["11", "12", "21", "Heavy", "Empty"]


def _player_attempts(player_id: str, season: int | None) -> pd.DataFrame:
    plays = _filter_player(player_id, season)
    return plays[(plays["pass_attempt"] == 1)
                 & plays["air_yards"].notna()
                 & plays["pass_location"].notna()]


def _apply_throw_filters(plays: pd.DataFrame, *,
                            coverages: list[str],
                            manzone: str,
                            pressure: str,
                            rushers: str,
                            personnel: list[str]) -> pd.DataFrame:
    if coverages:
        plays = plays[plays["defense_coverage_type"].isin(coverages)]
    if manzone == "Man":
        plays = plays[plays["defense_man_zone_type"] == "MAN_COVERAGE"]
    elif manzone == "Zone":
        plays = plays[plays["defense_man_zone_type"] == "ZONE_COVERAGE"]
    if pressure == "Pressured":
        plays = plays[plays["is_pressured"]]
    elif pressure == "Clean":
        plays = plays[~plays["is_pressured"]]
    if rushers == "3":
        plays = plays[plays["number_of_pass_rushers"] == 3]
    elif rushers == "4":
        plays = plays[plays["number_of_pass_rushers"] == 4]
    elif rushers == "5+":
        plays = plays[plays["number_of_pass_rushers"] >= 5]
    if personnel:
        plays = plays[plays["personnel_group"].isin(personnel)]
    return plays


def _build_throw_grid(plays: pd.DataFrame,
                        league: pd.DataFrame) -> tuple[list[list], list[list]]:
    """Returns (epa_matrix, annotation_matrix) for a 3×3 grid."""
    epa_matrix, ann_matrix = [], []
    for depth_label, lo, hi in _DEPTH_BUCKETS:
        epa_row, ann_row = [], []
        for loc in _LOCATIONS:
            zone = plays[(plays["air_yards"] >= lo)
                         & (plays["air_yards"] < hi)
                         & (plays["pass_location"] == loc)]
            n = len(zone)
            if n < 5:
                epa_row.append(None)
                ann_row.append(f"<b>—</b><br>{n} att")
                continue
            epa = float(zone["epa"].mean())
            comp_pct = float(zone["complete_pass"].mean())
            lg = league[(league["depth"] == depth_label)
                        & (league["location"] == loc)]
            if not lg.empty:
                delta = epa - float(lg["league_epa"].iloc[0])
                delta_str = f"{'▲' if delta > 0 else '▼'} {abs(delta):.2f} vs lg"
            else:
                delta_str = ""
            epa_row.append(epa)
            ann_row.append(
                f"<b>{epa:+.2f} EPA</b><br>"
                f"{comp_pct*100:.0f}% on {n} att<br>"
                f"<span style='font-size:11px;color:#666'>{delta_str}</span>"
            )
        epa_matrix.append(epa_row)
        ann_matrix.append(ann_row)
    return epa_matrix, ann_matrix


def _render_throw_grid(epa_matrix: list[list], ann_matrix: list[list],
                         *, title: str, z_scale: float, height: int = 420,
                         show_colorbar: bool = True) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=epa_matrix,
        x=[_LOC_LABELS[l] for l in _LOCATIONS],
        y=[d[0] for d in _DEPTH_BUCKETS],
        zmin=-z_scale, zmax=z_scale,
        colorscale=[[0, "#c0392b"], [0.5, "#f7f7f7"], [1, "#27ae60"]],
        colorbar=(dict(title="EPA<br>per att", thickness=14)
                  if show_colorbar else None),
        showscale=show_colorbar,
        hoverinfo="skip",
    ))
    for i, depth in enumerate([d[0] for d in _DEPTH_BUCKETS]):
        for j, loc in enumerate(_LOCATIONS):
            fig.add_annotation(
                x=_LOC_LABELS[loc], y=depth,
                text=ann_matrix[i][j],
                showarrow=False,
                font=dict(size=12, color="#222"),
                align="center",
            )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                    font=dict(size=14)),
        height=height,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(side="top", title="Pass location",
                    title_font=dict(size=11, color="#666")),
        yaxis=dict(title="Air-yard depth",
                    title_font=dict(size=11, color="#666")),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_throw_map(player_id: str, player_name: str, *,
                       season: int | None = None,
                       theme: dict | None = None,
                       key_prefix: str = "qb",
                       comparison_player_id: str | None = None,
                       comparison_player_name: str | None = None,
                       comparison_season: int | None = None) -> None:
    """Bucket 3 — Contextual throw map (page hero).

    3×3 grid (Short/Intermediate/Deep × Left/Middle/Right). Filters
    above re-render for any scenario combination. When
    `comparison_player_id` is supplied, two grids render side-by-side
    on a shared color scale.
    """
    plays_all = _player_attempts(player_id, season)
    if plays_all.empty or len(plays_all) < 30:
        st.caption(f"_Not enough throws ({len(plays_all)}) to build a throw map._")
        return

    # ── Filter chips ──────────────────────────────────────────────
    st.markdown(
        "**Throws by scenario** — _toggle filters; chart re-renders._"
    )
    f1, f2 = st.columns([3, 2])
    with f1:
        # Only show coverages this QB has actually faced (≥ 5 plays)
        avail = (
            plays_all["defense_coverage_type"]
            .dropna()
            .value_counts()
        )
        cov_avail = [c for c in avail.index
                     if avail[c] >= 5 and c in _COVERAGE_LABELS]
        cov_options = [c for c in [k for k, _ in _COVERAGE_OPTIONS]
                        if c in cov_avail]
        cov_pretty = [_COVERAGE_LABELS[c] for c in cov_options]
        cov_selected_pretty = st.multiselect(
            "Coverage type",
            options=cov_pretty,
            default=[],
            placeholder="All coverages",
            key=f"{key_prefix}_throw_cov",
        )
        cov_pretty_to_code = {v: k for k, v in _COVERAGE_LABELS.items()}
        cov_selected = [cov_pretty_to_code[p] for p in cov_selected_pretty]
    with f2:
        manzone = st.radio(
            "Coverage style",
            options=["All", "Man", "Zone"],
            horizontal=True,
            key=f"{key_prefix}_throw_manzone",
        )

    f3, f4, f5 = st.columns(3)
    with f3:
        pressure = st.radio(
            "Pressure",
            options=["All", "Pressured", "Clean"],
            horizontal=True,
            key=f"{key_prefix}_throw_press",
        )
    with f4:
        rushers = st.radio(
            "Pass rushers",
            options=["All", "3", "4", "5+"],
            horizontal=True,
            key=f"{key_prefix}_throw_rush",
        )
    with f5:
        # Personnel — only show ones with sample
        pers_avail_counts = (
            plays_all["personnel_group"].dropna().value_counts()
        )
        pers_avail = [p for p in _PERSONNEL_OPTIONS
                      if pers_avail_counts.get(p, 0) >= 10]
        personnel = st.multiselect(
            "Offensive personnel",
            options=pers_avail,
            default=[],
            placeholder="All",
            key=f"{key_prefix}_throw_pers",
        )

    # ── Apply filters ──────────────────────────────────────────────
    filter_kwargs = dict(coverages=cov_selected, manzone=manzone,
                          pressure=pressure, rushers=rushers,
                          personnel=personnel)
    plays = _apply_throw_filters(plays_all, **filter_kwargs)
    if len(plays) < 10:
        st.warning(
            f"Only {len(plays)} throws match this scenario — too few "
            f"to render. Loosen the filters."
        )
        return

    # Headline metrics for the filtered slice — A vs. B if comparing
    is_comp = comparison_player_id is not None
    if is_comp:
        comp_plays_all = _player_attempts(comparison_player_id, comparison_season)
        comp_plays = _apply_throw_filters(comp_plays_all, **filter_kwargs)
    else:
        comp_plays = pd.DataFrame()

    def _safe_metric(s_plays: pd.DataFrame) -> dict:
        if s_plays.empty:
            return {"n": 0, "epa": None, "comp": None, "int": None}
        return {
            "n": len(s_plays),
            "epa": float(s_plays["epa"].mean()),
            "comp": float(s_plays["complete_pass"].mean()),
            "int": float(s_plays["interception"].mean()) * 100,
        }
    m_a = _safe_metric(plays)
    m_b = _safe_metric(comp_plays) if is_comp else None

    def _fmt_metric(label, primary_val, comp_val=None):
        if comp_val is None or comp_val.get("epa") is None:
            return label, primary_val, None
        return label, primary_val, comp_val
    m1, m2, m3, m4 = st.columns(4)
    if is_comp:
        m1.metric("Throws in scenario",
                   f"{m_a['n']:,} · {m_b['n']:,}",
                   help=f"{player_name} · {comparison_player_name}")
        delta_epa = (m_a['epa'] - m_b['epa']) if (m_a['epa'] is not None and m_b['epa'] is not None) else None
        m2.metric("EPA / attempt",
                   f"{m_a['epa']:+.3f}" if m_a['epa'] is not None else "—",
                   delta=(f"{delta_epa:+.3f} vs comp" if delta_epa is not None else None))
        m3.metric("Comp %",
                   f"{m_a['comp']*100:.1f}% · {m_b['comp']*100:.1f}%"
                       if (m_a['comp'] is not None and m_b['comp'] is not None)
                       else "—")
        m4.metric("INT rate",
                   f"{m_a['int']:.2f}% · {m_b['int']:.2f}%"
                       if (m_a['int'] is not None and m_b['int'] is not None)
                       else "—")
    else:
        m1.metric("Throws in scenario", f"{m_a['n']:,}")
        m2.metric("EPA / attempt", f"{m_a['epa']:+.3f}")
        m3.metric("Completion %", f"{m_a['comp']*100:.1f}%")
        m4.metric("INT rate", f"{m_a['int']:.2f}%")

    league = _league_throw_map(season)

    # Build the primary grid + (optional) comparison grid; share a
    # color scale so cells are directly comparable between the two.
    epa_a, ann_a = _build_throw_grid(plays, league)
    epa_b, ann_b = (_build_throw_grid(comp_plays,
                                        _league_throw_map(comparison_season))
                    if is_comp else (None, None))

    flat = [v for row in epa_a for v in row if v is not None]
    if is_comp and epa_b is not None:
        flat += [v for row in epa_b for v in row if v is not None]
    if not flat:
        st.caption("_Not enough throws per zone to render._")
        return
    abs_max = max(abs(min(flat)), abs(max(flat)))
    z_scale = abs_max if abs_max > 0 else 0.5

    if is_comp:
        cA, cB = st.columns(2)
        with cA:
            fig_a = _render_throw_grid(epa_a, ann_a,
                                         title=f"{player_name} ({season or 'career'})",
                                         z_scale=z_scale,
                                         show_colorbar=False)
            st.plotly_chart(fig_a, use_container_width=True)
        with cB:
            fig_b = _render_throw_grid(epa_b, ann_b,
                                         title=f"{comparison_player_name} "
                                               f"({comparison_season or 'career'})",
                                         z_scale=z_scale,
                                         show_colorbar=True)
            st.plotly_chart(fig_b, use_container_width=True)
    else:
        fig = _render_throw_grid(epa_a, ann_a,
                                   title="",
                                   z_scale=z_scale,
                                   show_colorbar=True)
        st.plotly_chart(fig, use_container_width=True)

    # Best / worst zone caption (primary QB only — comparison is visual)
    flat_zones = []
    for i, depth in enumerate([d[0] for d in _DEPTH_BUCKETS]):
        for j, loc in enumerate(_LOCATIONS):
            v = epa_a[i][j]
            if v is not None:
                flat_zones.append((v, f"{depth.lower()} {loc}"))
    if len(flat_zones) >= 2:
        flat_zones.sort()
        worst_v, worst_z = flat_zones[0]
        best_v, best_z = flat_zones[-1]
        st.caption(
            f"**Best zone:** {best_z} ({best_v:+.2f} EPA/att). "
            f"**Worst zone:** {worst_z} ({worst_v:+.2f} EPA/att). "
            f"Green = better than league, red = worse. Δ shown bottom of each cell."
        )


# ── Situational splits ────────────────────────────────────────────
_SITUATIONS = [
    ("3rd down", "is_third_down", "Money downs.",
     "Plays where the down is 3rd."),
    ("Red zone", "is_red_zone", "Inside the 20.",
     "Plays starting from the opponent's 20-yard line or closer."),
    ("4th quarter", "is_fourth_quarter", "Clutch time.",
     "Plays in the 4th quarter."),
    ("2-min drill", "is_two_minute", "End-of-half pressure.",
     "Plays inside 2 minutes of either half."),
]


@st.cache_data
def _league_situational_avg(season: int | None) -> dict:
    db = load_qb_dropbacks()
    if season is not None:
        db = db[db["season"] == season]
    out = {}
    for label, flag, _, _ in _SITUATIONS:
        sub = db[db[flag]]
        out[label] = {
            "epa": float(sub["epa"].mean()) if len(sub) else 0.0,
            "success": float(sub["success"].mean()) if len(sub) else 0.0,
            "comp_pct": float(sub["complete_pass"].mean()) if len(sub) else 0.0,
        }
    return out


def render_situational_split(player_id: str, player_name: str, *,
                                season: int | None = None,
                                theme: dict | None = None) -> None:
    """Bucket 6 — Situational performance.

    Four side-by-side panels: 3rd down · red zone · 4th quarter ·
    2-minute drill. Each shows player's EPA per dropback in that
    situation vs. league average, with sample size and delta.
    """
    plays = _filter_player(player_id, season)
    if plays.empty or len(plays) < 50:
        st.caption(f"_Not enough dropbacks ({len(plays)}) to compute "
                    f"situational splits._")
        return

    league = _league_situational_avg(season)

    cols = st.columns(4)
    for i, (label, flag, tagline, helptext) in enumerate(_SITUATIONS):
        sub = plays[plays[flag]]
        n = len(sub)
        with cols[i]:
            st.markdown(f"**{label}**  \n_{tagline}_")
            if n < 10:
                st.metric("EPA / dropback", "—",
                           help=f"Only {n} plays — not enough.")
                st.caption(f"_{n} plays_")
                continue
            qb_epa = float(sub["epa"].mean())
            lg_epa = league[label]["epa"]
            delta = qb_epa - lg_epa
            st.metric(
                "EPA / dropback",
                f"{qb_epa:+.2f}",
                delta=f"{delta:+.2f} vs league",
                help=helptext,
            )
            success_pct = float(sub["success"].mean()) * 100
            st.caption(
                f"{n} plays · {success_pct:.0f}% successful · "
                f"league avg: {lg_epa:+.2f}"
            )

    # Auto-narrative: which situation does he own / fold in?
    deltas = []
    for label, flag, _, _ in _SITUATIONS:
        sub = plays[plays[flag]]
        if len(sub) >= 30:
            d = float(sub["epa"].mean()) - league[label]["epa"]
            deltas.append((d, label))
    if len(deltas) >= 2:
        deltas.sort()
        worst_d, worst_l = deltas[0]
        best_d, best_l = deltas[-1]
        if best_d > 0.05 and worst_d < -0.05:
            st.info(
                f"**Situational signature:** Best in **{best_l}** "
                f"({best_d:+.2f} EPA above league). Worst in **{worst_l}** "
                f"({worst_d:+.2f} below)."
            )
        elif best_d > 0.05:
            st.success(
                f"**Best situation:** {best_l} ({best_d:+.2f} EPA above "
                f"league average)."
            )
        elif worst_d < -0.05:
            st.warning(
                f"**Worst situation:** {worst_l} ({worst_d:+.2f} EPA "
                f"below league average)."
            )


def render_presnap_split(player_id: str, player_name: str, *,
                            season: int | None = None,
                            theme: dict | None = None) -> None:
    """Bucket 1 — Pre-snap.

    From the dropback feed: shotgun usage, no-huddle rate, average
    play clock at snap (proxy for tempo / hard count behavior),
    play distribution by down. Compared to league.
    """
    plays = _filter_player(player_id, season)
    if plays.empty or len(plays) < 50:
        st.caption(f"_Not enough dropbacks ({len(plays)}) for pre-snap stats._")
        return

    # League comparators
    db = load_qb_dropbacks()
    if season is not None:
        db = db[db["season"] == season]

    qb_shotgun = float(plays["shotgun"].mean())
    qb_no_huddle = float(plays["no_huddle"].mean())
    qb_play_clock = float(plays["play_clock"].dropna().astype(float).mean())
    lg_shotgun = float(db["shotgun"].mean())
    lg_no_huddle = float(db["no_huddle"].mean())
    lg_play_clock = float(db["play_clock"].dropna().astype(float).mean())

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Shotgun rate",
        f"{qb_shotgun*100:.1f}%",
        delta=f"{(qb_shotgun-lg_shotgun)*100:+.1f}% vs league",
        help="% of dropbacks from shotgun. Higher = pass-first / spread offense.",
    )
    c2.metric(
        "No-huddle rate",
        f"{qb_no_huddle*100:.1f}%",
        delta=f"{(qb_no_huddle-lg_no_huddle)*100:+.1f}% vs league",
        help="% of plays run without a huddle — tempo signal.",
    )
    c3.metric(
        "Avg play clock at snap",
        f"{qb_play_clock:.1f}s",
        delta=f"{qb_play_clock-lg_play_clock:+.1f}s vs league",
        delta_color="off",
        help="Time on the play clock when ball is snapped. Lower = "
             "snapping fast (uptempo). Higher = holding to last second "
             "(might indicate audibles / hard counts).",
    )

    # Down-distribution table
    dd = (
        plays.groupby("down")
        .agg(
            n=("epa", "size"),
            epa=("epa", "mean"),
            comp=("complete_pass", "mean"),
        )
        .reset_index()
    )
    dd = dd[dd["down"].isin([1, 2, 3, 4])]
    dd["Down"] = dd["down"].astype(int).astype(str) + (
        dd["down"].map({1: "st", 2: "nd", 3: "rd", 4: "th"})
    )
    dd["EPA/play"] = dd["epa"].apply(lambda v: f"{v:+.2f}")
    dd["Comp %"] = dd["comp"].apply(lambda v: f"{v*100:.0f}%")
    dd = dd.rename(columns={"n": "Plays"})
    st.dataframe(dd[["Down", "Plays", "EPA/play", "Comp %"]],
                  use_container_width=True, hide_index=True)


def render_processing_split(player_id: str, player_name: str, *,
                                season: int | None = None,
                                theme: dict | None = None) -> None:
    """Bucket 2 — Processing.

    From NGS: time-to-throw, aggressiveness (% into tight windows),
    intended vs. completed air yards (the gap = receiver YAC + drops),
    and air yards to sticks (passing the marker on average).
    """
    ngs = load_qb_ngs()
    if ngs.empty:
        st.caption("_NGS data not loaded — run `tools/build_qb_ngs.py`._")
        return

    rows = ngs[ngs["player_gsis_id"] == player_id]
    if season is not None:
        rows = rows[rows["season"] == season]
    if rows.empty:
        st.caption("_No NGS data for this player/season._")
        return

    # Avg across selected seasons (if career view, weight by attempts)
    if "attempts" in rows.columns and rows["attempts"].sum() > 0:
        w = rows["attempts"]
    else:
        w = pd.Series(1, index=rows.index)

    def _wmean(col):
        if col not in rows.columns or rows[col].dropna().empty:
            return None
        valid = rows[col].notna()
        if valid.sum() == 0:
            return None
        return float((rows.loc[valid, col] * w[valid]).sum() / w[valid].sum())

    qb_ttt = _wmean("avg_time_to_throw")
    qb_aggr = _wmean("aggressiveness")
    qb_intended = _wmean("avg_intended_air_yards")
    qb_completed = _wmean("avg_completed_air_yards")
    qb_to_sticks = _wmean("avg_air_yards_to_sticks")

    # League comparators (same season filter)
    lg = ngs.copy()
    if season is not None:
        lg = lg[lg["season"] == season]
    if "attempts" in lg.columns:
        lg = lg[lg["attempts"] >= 100]  # qualifier

    def _lg_mean(col):
        if col not in lg.columns or lg[col].dropna().empty:
            return None
        return float(lg[col].mean())

    lg_ttt = _lg_mean("avg_time_to_throw")
    lg_aggr = _lg_mean("aggressiveness")
    lg_intended = _lg_mean("avg_intended_air_yards")
    lg_to_sticks = _lg_mean("avg_air_yards_to_sticks")

    def _fmt(v, fmt):
        if v is None or pd.isna(v):
            return "—"
        return fmt.format(v)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Time to throw",
        _fmt(qb_ttt, "{:.2f}s"),
        delta=(_fmt(qb_ttt - lg_ttt, "{:+.2f}s vs league")
                if qb_ttt is not None and lg_ttt is not None else None),
        delta_color="off",
        help="Avg seconds from snap to throw. Lower = quicker processor; "
             "higher = holds the ball longer (more chances + more sacks).",
    )
    c2.metric(
        "Aggressiveness",
        _fmt(qb_aggr, "{:.1f}%"),
        delta=(_fmt(qb_aggr - lg_aggr, "{:+.1f}% vs league")
                if qb_aggr is not None and lg_aggr is not None else None),
        help="% of attempts into tight coverage (defender within 1 yard "
             "of receiver at catch). Higher = more risk-taking.",
    )
    c3.metric(
        "Intended air yards",
        _fmt(qb_intended, "{:.1f} yd"),
        delta=(_fmt(qb_intended - lg_intended, "{:+.1f} vs league")
                if qb_intended is not None and lg_intended is not None else None),
        help="Avg depth of his targets (how far the ball travels in the "
             "air, on average). Higher = downfield thrower.",
    )
    c4.metric(
        "Air yards to sticks",
        _fmt(qb_to_sticks, "{:+.2f}"),
        delta=(_fmt(qb_to_sticks - lg_to_sticks, "{:+.2f} vs league")
                if qb_to_sticks is not None and lg_to_sticks is not None else None),
        delta_color="off",
        help="On average, how far past (or short of) the first-down "
             "marker his targets are. Positive = throws past the sticks.",
    )

    # Narrative: time-to-throw + aggressiveness combo tells a profile
    if qb_ttt is not None and qb_aggr is not None and lg_ttt is not None and lg_aggr is not None:
        slow = qb_ttt - lg_ttt > 0.10
        fast = qb_ttt - lg_ttt < -0.10
        aggressive = qb_aggr - lg_aggr > 1.5
        conservative = qb_aggr - lg_aggr < -1.5
        if fast and conservative:
            st.info("**Quick, conservative processor** — gets the ball out fast "
                     "and avoids tight windows. Game-manager profile.")
        elif slow and aggressive:
            st.warning("**Holds the ball, takes risks** — high time-to-throw "
                        "and high aggressiveness. Boom-bust hero-ball.")
        elif fast and aggressive:
            st.success("**Quick AND aggressive** — rare combo. Reads fast and "
                        "still attacks tight windows.")
        elif slow and conservative:
            st.info("**Patient and conservative** — holds the ball but doesn't "
                     "attack risky throws. Often paired with strong scrambling.")

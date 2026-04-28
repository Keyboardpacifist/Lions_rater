"""
Team page — the destination from the league-wide NFL grid.

Hero: team header (logo, colors, name, season).
Body: team rater snapshot, comp engine (top-3 historical comparables
with generated narrative), and click-through into individual position
pages for the roster.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lib_shared import inject_css, team_theme
from lib_team_comps import find_team_comps, load_team_seasons
from lib_team_contention import (
    classify_team,
    render_contention_badge,
    compute_gap_analysis,
    compute_trajectory,
    compute_team_timeline,
    render_team_timeline_html,
    _GAP_TITLES,
)
from lib_team_drilldown import get_drilldown_narrative

st.set_page_config(
    page_title="Team Profile",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Pick team / season ──────────────────────────────────────────
qp = st.query_params
qp_team = qp.get("abbr")
qp_season = qp.get("season")

team_df = load_team_seasons()
if team_df.empty:
    st.error("Team data not loaded. Run `python tools/build_team_seasons.py`.")
    st.stop()

teams_avail = sorted(team_df["team"].unique().tolist())
seasons_avail = sorted(team_df["season"].unique().tolist(), reverse=True)

# Streamlit selectboxes prefer session_state over index=, so when the
# user clicks a different team on the landing grid, we have to PUSH the
# new value into session_state BEFORE the selectbox renders. Otherwise
# the previously-picked team sticks even though the URL changed.
if qp_team and qp_team in teams_avail:
    st.session_state["team_pick"] = qp_team
elif "team_pick" not in st.session_state:
    st.session_state["team_pick"] = "DET"

if qp_season:
    try:
        s_int = int(qp_season)
        if s_int in seasons_avail:
            st.session_state["season_pick"] = s_int
    except (ValueError, TypeError):
        pass
if "season_pick" not in st.session_state:
    st.session_state["season_pick"] = seasons_avail[0]

c1, c2 = st.columns([2, 1])
with c1:
    team = st.selectbox(
        "Team",
        options=teams_avail,
        key="team_pick",
    )
with c2:
    season = st.selectbox(
        "Season",
        options=seasons_avail,
        key="season_pick",
    )

# Keep the URL in sync with the active selection so it's shareable
st.query_params.update({"abbr": team, "season": str(season)})

theme = team_theme(team)
primary = theme.get("primary", "#1F2A44")
secondary = theme.get("secondary", "#0B1730")
logo = theme.get("logo", "")
team_name = theme.get("name", team)

row = team_df[(team_df["team"] == team) & (team_df["season"] == season)]
if row.empty:
    st.warning(f"No data for {team} in {season}.")
    st.stop()
row = row.iloc[0]

# ── Hero header — logo, name, season, contention badge, gap analysis
contention = classify_team(team, int(season))
contention_html = render_contention_badge(
    contention["state"], contention["rationale"]
)
gaps = compute_gap_analysis(team_df, team, int(season), n_gaps=3)
trajectory = compute_trajectory(team_df, team, int(season))

_logo_html = (
    f'<img src="{logo}" style="height:110px;width:110px;object-fit:contain;'
    'filter:drop-shadow(0 4px 10px rgba(0,0,0,0.35));"/>'
    if logo else ''
)
hero_html = (
    f'<div style="background:linear-gradient(135deg,{primary} 0%,{secondary} 100%);'
    'border-radius:18px;padding:28px 32px;margin-bottom:16px;'
    'box-shadow:0 6px 18px rgba(0,0,0,0.18);color:white;">'
    '<div style="display:flex;align-items:center;gap:24px;">'
    f'{_logo_html}'
    '<div style="flex:1;">'
    f'<div style="font-size:38px;font-weight:900;letter-spacing:-0.5px;line-height:1;">{team_name}</div>'
    f'<div style="font-size:14px;opacity:0.8;margin-top:6px;font-weight:500;letter-spacing:1px;">{season} SEASON</div>'
    f'<div style="margin-top:14px;">{contention_html}</div>'
    '</div>'
    '</div>'
    '</div>'
)
st.markdown(hero_html, unsafe_allow_html=True)

# ── Contention timeline — visual arc across all available seasons ──
timeline = compute_team_timeline(team)
timeline_html = render_team_timeline_html(timeline,
                                              highlight_season=int(season))
st.markdown(timeline_html, unsafe_allow_html=True)


# ── Gap analysis — expandable rows with drill-down narratives ──
def _ord(n):
    suf = "th"
    if n % 100 not in (11, 12, 13):
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


if gaps:
    gap_title = _GAP_TITLES.get(contention["state"], "Biggest gaps")
    st.markdown(f"#### 🎯 {gap_title}")
    st.caption(
        "Click any item for a player-level breakdown of where the issue lives."
    )
    for g in gaps:
        header = (
            f"**{g['label'].title()}** — {g['rank']} of {g['total']} · "
            f"_{g['phrase']}_"
        )
        with st.expander(header, expanded=False):
            narrative = get_drilldown_narrative(
                team, int(season), g["label"], direction="gap"
            )
            st.markdown(narrative)


# ── Year-over-year trajectory ──
if trajectory["improved"] or trajectory["slipped"]:
    prior = trajectory["prior_season"]
    st.markdown(f"#### 📈 Year over year — vs {prior}")
    st.caption(
        "Rank shifts in each phase. Click any item for a player-level "
        "explanation of what drove the change."
    )
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**▲ Improved**")
        if not trajectory["improved"]:
            st.caption("_— no meaningful gains —_")
        for d in trajectory["improved"]:
            header = (
                f"**{d['label']}** — {_ord(d['prior_rank'])} → "
                f"{_ord(d['current_rank'])} (▲ {d['spots_gained']} spots)"
            )
            with st.expander(header, expanded=False):
                st.markdown(get_drilldown_narrative(
                    team, int(season), d["label"],
                    direction="improvement",
                ))
    with tc2:
        st.markdown("**▼ Slipped**")
        if not trajectory["slipped"]:
            st.caption("_— no meaningful drops —_")
        for d in trajectory["slipped"]:
            header = (
                f"**{d['label']}** — {_ord(d['prior_rank'])} → "
                f"{_ord(d['current_rank'])} (▼ {abs(d['spots_gained'])} spots)"
            )
            with st.expander(header, expanded=False):
                st.markdown(get_drilldown_narrative(
                    team, int(season), d["label"],
                    direction="slipped",
                ))

# ── Phase-by-phase panels with league ranks ──────────────────
def _rank_in_season(team_df, season, stat, ascending=False):
    """Returns (rank, total) for the given (team, season, stat).
    `ascending=True` for stats where lower is better."""
    season_pool = team_df[team_df["season"] == season].copy()
    season_pool = season_pool.dropna(subset=[stat])
    if season_pool.empty:
        return None, 0
    season_pool = season_pool.sort_values(stat, ascending=ascending)
    season_pool = season_pool.reset_index(drop=True)
    total = len(season_pool)
    match = season_pool.index[season_pool["team"] == team].tolist()
    if not match:
        return None, total
    return match[0] + 1, total


def _ord(n):
    if n is None:
        return "—"
    suf = "th"
    if n % 100 not in (11, 12, 13):
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _render_stat_row(label, value, fmt, rank_value, total, help_text=""):
    """One stat row with raw value + league rank pill."""
    rank_str = f"{_ord(rank_value)} of {total}" if rank_value else "—"
    st.markdown(
        f"""
<div style="display: flex; justify-content: space-between; align-items: center;
             padding: 8px 12px; border-bottom: 1px solid rgba(0,0,0,0.06);
             font-size: 14px;">
    <div title="{help_text}">{label}</div>
    <div style="display: flex; gap: 12px; align-items: center;">
        <div style="font-weight: 600; min-width: 70px; text-align: right;">
            {fmt.format(value) if pd.notna(value) else '—'}
        </div>
        <div style="font-size: 11px; opacity: 0.6; min-width: 55px;
                     text-align: right;">
            {rank_str}
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# (label, raw_col, format, ascending) — ascending=True for "lower better"
_OFFENSE_STATS = [
    ("Points / game",          "points_per_game",          "{:.1f}",   False),
    ("Off EPA / play",         "off_epa_per_play",         "{:+.3f}",  False),
    ("Pass EPA / play",        "off_pass_epa_per_play",    "{:+.3f}",  False),
    ("Rush EPA / play",        "off_rush_epa_per_play",    "{:+.3f}",  False),
    ("Success rate",           "off_success_rate",         "{:.1%}",   False),
    ("Explosive play rate",    "off_explosive_rate",       "{:.1%}",   False),
    ("Red zone TD rate",       "off_red_zone_td_rate",     "{:.1%}",   False),
    ("3rd down conversion",    "off_third_down_conv_rate", "{:.1%}",   False),
    ("Giveaway rate",          "off_giveaway_rate",        "{:.2%}",   True),
]
_DEFENSE_STATS = [
    ("Points allowed / game",  "points_allowed_per_game",  "{:.1f}",   True),
    ("Def EPA allowed / play", "def_epa_per_play",         "{:+.3f}",  True),
    ("Pass EPA allowed",       "def_pass_epa_allowed",     "{:+.3f}",  True),
    ("Rush EPA allowed",       "def_rush_epa_allowed",     "{:+.3f}",  True),
    ("Success rate allowed",   "def_success_rate_allowed", "{:.1%}",   True),
    ("Takeaway rate",          "def_takeaway_rate",        "{:.2%}",   False),
    ("Pressure rate",          "def_pressure_rate",        "{:.1%}",   False),
    ("Sack rate",              "def_sack_rate",            "{:.1%}",   False),
]
_SITUATIONAL_STATS = [
    ("Point differential / G", "point_differential_per_game", "{:+.1f}", False),
    ("4Q offense EPA",         "fourth_q_off_epa",            "{:+.3f}", False),
    ("4Q defense EPA",         "fourth_q_def_epa",            "{:+.3f}", True),
    ("Penalty yards / game",   "penalty_yards_per_game",      "{:.1f}",  True),
]


def _render_phase_panel(title, stats_cfg, team_df, team, season, row):
    st.markdown(f"#### {title}")
    for label, col, fmt, ascending in stats_cfg:
        if col not in row.index:
            continue
        val = row.get(col)
        rank, total = _rank_in_season(team_df, season, col, ascending=ascending)
        _render_stat_row(label, val, fmt, rank, total)


cc1, cc2 = st.columns(2)
with cc1:
    _render_phase_panel("⚔️ Offense", _OFFENSE_STATS,
                          team_df, team, int(season), row)
with cc2:
    _render_phase_panel("🛡️ Defense", _DEFENSE_STATS,
                          team_df, team, int(season), row)

st.markdown("")
_render_phase_panel("⏱️ Situational & Discipline", _SITUATIONAL_STATS,
                      team_df, team, int(season), row)

# ── Comp engine — the headline feature ────────────────────────
st.markdown("---")
st.markdown(
    f"### 🔮  Most comparable {season - 1 if season else 'past'}–era teams"
)
st.caption(
    "Cosine similarity across 320 (team × season) profiles since 2016. "
    "The engine finds team-seasons with the most similar statistical "
    "DNA and writes a one-sentence reason citing the shared traits."
)

scope = st.radio(
    "Compare against:",
    options=["offense", "defense", "full"],
    horizontal=True,
    format_func=lambda s: {"offense": "Offensive twins",
                             "defense": "Defensive twins",
                             "full":    "Full-team twins"}[s],
    key="comp_scope",
)

comps = find_team_comps(
    team=team, season=int(season),
    scope=scope, n=3,
    exclude_same_team=True,
)
if not comps:
    st.info("Not enough data to compute comps for this team-season yet.")
else:
    cols = st.columns(3)
    for col, c in zip(cols, comps):
        comp_theme = team_theme(c["team"])
        comp_primary = comp_theme.get("primary", "#1F2A44")
        comp_secondary = comp_theme.get("secondary", "#0B1730")
        comp_logo = comp_theme.get("logo", "")
        comp_name = comp_theme.get("name", c["team"])
        with col:
            st.markdown(
                f"""
<div style="
    background: linear-gradient(135deg, {comp_primary} 0%, {comp_secondary} 100%);
    border-radius: 14px;
    padding: 20px;
    height: 100%;
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
">
    <div style="display: flex; align-items: center; gap: 12px;">
        {f'<img src="{comp_logo}" style="height: 60px; width: 60px; object-fit: contain;"/>' if comp_logo else ''}
        <div>
            <div style="font-size: 11px; opacity: 0.7; letter-spacing: 1.5px;">
                SIMILARITY {c["similarity"]*100:.0f}%
            </div>
            <div style="font-size: 22px; font-weight: 800; line-height: 1.1;">
                {c["season"]} {comp_name}
            </div>
        </div>
    </div>
    <div style="margin-top: 16px; font-size: 14px; line-height: 1.5;
                 opacity: 0.95;">
        {c["reason"]}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown("")  # spacer
            if st.button(
                f"Open {c['season']} {c['team']} →",
                key=f"go_{c['team']}_{c['season']}",
                use_container_width=True,
            ):
                # Push to session_state and query_params — same as the
                # landing-grid button handler, for the same reason.
                st.session_state["team_pick"] = c["team"]
                st.session_state["season_pick"] = int(c["season"])
                st.query_params.update({
                    "abbr": c["team"],
                    "season": str(c["season"]),
                })
                st.rerun()

# ── Roster — top players per position ──────────────────────────
st.markdown("---")
st.markdown("### 🦌  Roster — top performers")
st.caption(
    "Top 3 players per position by all-stats z-score, this team-season. "
    "Click any player to drill into their full rater page."
)

import polars as pl

# Each entry: position label, parquet, optional row filter, page to open
_ROSTER_SOURCES = [
    ("QB",  "league_qb_all_seasons.parquet",  None,                "pages/QB.py"),
    ("WR",  "league_wr_all_seasons.parquet",  ("position", "WR"),  "pages/WR.py"),
    ("TE",  "league_te_all_seasons.parquet",  ("position", "TE"),  "pages/TE.py"),
    ("RB",  "league_rb_all_seasons.parquet",  None,                "pages/2_Running_backs.py"),
    ("OL",  "league_ol_all_seasons.parquet",  None,                "pages/3_Offensive_Line.py"),
    ("EDGE","league_de_all_seasons.parquet",  None,                "pages/DE.py"),
    ("DT",  "league_dt_all_seasons.parquet",  None,                "pages/DT.py"),
    ("LB",  "league_lb_all_seasons.parquet",  None,                "pages/LB.py"),
    ("CB",  "league_cb_all_seasons.parquet",  None,                "pages/CB.py"),
    ("S",   "league_s_all_seasons.parquet",   None,                "pages/Safety..py"),
    ("K",   "league_k_all_seasons.parquet",   None,                "pages/Kicker.py"),
    ("P",   "league_p_all_seasons.parquet",   None,                "pages/Punter.py"),
]


@st.cache_data(show_spinner=False)
def _team_roster_top(team: str, season: int) -> dict:
    """Returns {position: [(name, score, page_slug, player_id), …top 3]}.
    Score = average of all available z-stat columns for that player-row."""
    out: dict[str, list] = {}
    base = REPO_ROOT / "data"
    for pos_label, fname, row_filter, page_slug in _ROSTER_SOURCES:
        path = base / fname
        if not path.exists():
            continue
        try:
            df = pl.read_parquet(path).to_pandas()
        except Exception:
            continue
        team_col = "recent_team" if "recent_team" in df.columns else (
            "team" if "team" in df.columns else None)
        season_col = "season_year" if "season_year" in df.columns else (
            "season" if "season" in df.columns else None)
        if team_col is None or season_col is None:
            continue
        sub = df[(df[team_col] == team) & (df[season_col] == season)]
        if row_filter:
            col, val = row_filter
            if col in sub.columns:
                sub = sub[sub[col] == val]
        if sub.empty:
            continue
        z_cols = [c for c in sub.columns if c.endswith("_z")]
        if not z_cols:
            continue
        # All-stats average — same idea as the league_score calculation,
        # just using equal weight across z-cols. Good enough for "top
        # performers" because we're comparing within the same parquet.
        sub = sub.copy()
        sub["_avg_z"] = sub[z_cols].mean(axis=1, skipna=True)
        # Player display name column varies — fall back to first match
        for name_col in ("player_display_name", "player_name", "full_name"):
            if name_col in sub.columns:
                break
        else:
            name_col = None
        pid_col = "player_id" if "player_id" in sub.columns else None
        if name_col is None:
            continue
        sub = sub.dropna(subset=["_avg_z", name_col])
        if sub.empty:
            continue
        top = sub.sort_values("_avg_z", ascending=False).head(3)
        rows = []
        for _, r in top.iterrows():
            rows.append({
                "name": str(r[name_col]),
                "score": float(r["_avg_z"]),
                "page": page_slug,
                "pid": str(r[pid_col]) if pid_col and pd.notna(r[pid_col]) else "",
            })
        out[pos_label] = rows
    return out


_roster = _team_roster_top(team, int(season))
if not _roster:
    st.info("No roster data for this team-season yet.")
else:
    # Render in 4-column rows of position cards
    pos_keys = list(_roster.keys())
    for i in range(0, len(pos_keys), 4):
        row_keys = pos_keys[i:i + 4]
        cols = st.columns(4)
        for col, pk in zip(cols, row_keys):
            with col:
                st.markdown(
                    f"""
<div style="font-size: 11px; font-weight: 800; letter-spacing: 1.5px;
             opacity: 0.55; margin: 4px 0 4px 4px;">
    {pk.upper()}
</div>
""",
                    unsafe_allow_html=True,
                )
                for r in _roster[pk]:
                    sign = "+" if r["score"] >= 0 else ""
                    label = f"{r['name']}  ·  {sign}{r['score']:.2f}"
                    if st.button(
                        label,
                        key=f"roster_{pk}_{r['pid'] or r['name']}",
                        use_container_width=True,
                        help=f"Open {r['name']}'s {pk} page",
                    ):
                        st.switch_page(r["page"])

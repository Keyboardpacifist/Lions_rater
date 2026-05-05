"""
Team page — the destination from the league-wide NFL grid.

Hero (always visible): team header + contention badge + gap analysis +
trajectory + timeline ribbon.

Body (in tabs): Stats & Identity / Schedule & Games / Tendencies /
Roster.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from lib_shared import inject_css, team_theme
from lib_top_nav import render_home_button
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
from lib_team_game_log import render_game_log, get_team_game_log
from lib_team_tendencies import (
    get_team_tendencies,
    get_filter_options,
    render_filtered_route_tree,
    render_filtered_run_profile,
    render_filtered_throw_map,
)

st.set_page_config(
    page_title="Team Profile",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

render_home_button()  # ← back to landing
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

# Apply pending nav intent (set by the "Open …" comp buttons) before
# the selectbox widgets render. We can't write to widget keys after
# they're instantiated, so the click handler stashes the target in
# `team_nav_intent` and we resolve it here, on the rerun.
_nav = st.session_state.pop("team_nav_intent", None)
if _nav:
    nav_team, nav_season = _nav
    if nav_team in teams_avail:
        st.session_state["team_pick"] = nav_team
    if nav_season in seasons_avail:
        st.session_state["season_pick"] = int(nav_season)

# Read query params into session_state ONLY on first render. After
# that, the selectbox is the source of truth — pushing stale qp
# values back in on every rerun would overwrite the user's pick
# (st.query_params.update() doesn't sync the URL until *after* the
# rerun, so on the rerun triggered by the selectbox change, qp
# still holds the old value).
if "team_pick" not in st.session_state:
    st.session_state["team_pick"] = (
        qp_team if (qp_team and qp_team in teams_avail) else "DET"
    )
if "season_pick" not in st.session_state:
    _s_int = None
    if qp_season:
        try:
            _s_int = int(qp_season)
        except (ValueError, TypeError):
            pass
    st.session_state["season_pick"] = (
        _s_int if _s_int in seasons_avail else seasons_avail[0]
    )

c1, c2 = st.columns([2, 1])
with c1:
    team = st.selectbox("Team", options=teams_avail, key="team_pick")
with c2:
    season = st.selectbox("Season", options=seasons_avail, key="season_pick")

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


# ════════════════════════════════════════════════════════════════
# STICKY IDENTITY HERO — always visible above the tabs
# ════════════════════════════════════════════════════════════════

# ── Hero header — logo, name, season, contention badge ────────
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
    '</div></div></div>'
)
st.markdown(hero_html, unsafe_allow_html=True)

# ── Top summary strip — record · SOS · EPA composites ────────
from lib_team_header_strip import render_team_header_strip
render_team_header_strip(team, int(season), row)

# ── Contention timeline ribbon ───────────────────────────────
timeline = compute_team_timeline(team)
timeline_html = render_team_timeline_html(timeline,
                                              highlight_season=int(season))
st.markdown(timeline_html, unsafe_allow_html=True)


def _ord(n):
    if n is None:
        return "—"
    suf = "th"
    if n % 100 not in (11, 12, 13):
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


# ── Gap analysis (expandable rows) ──────────────────────────
if gaps:
    gap_title = _GAP_TITLES.get(contention["state"], "Biggest gaps")
    st.markdown(f"#### 🎯 {gap_title}")
    st.caption("Click any item for a player-level breakdown of where the issue lives.")
    for g in gaps:
        header = (
            f"**{g['label'].title()}** — {g['rank']} of {g['total']} · "
            f"_{g['phrase']}_"
        )
        with st.expander(header, expanded=False):
            st.markdown(get_drilldown_narrative(
                team, int(season), g["label"], direction="gap"
            ))


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
                    team, int(season), d["label"], direction="improvement",
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
                    team, int(season), d["label"], direction="slipped",
                ))


# ════════════════════════════════════════════════════════════════
# TAB HELPERS — defined once, used inside the tab blocks below
# ════════════════════════════════════════════════════════════════

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
                     text-align: right;">{rank_str}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


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
        rank, total = _rank_in_season(team_df, season, col,
                                          ascending=ascending)
        _render_stat_row(label, val, fmt, rank, total)


_POS_KEY_PREFIX = {
    "QB": "qb", "WR": "wr", "TE": "te", "RB": "rb", "OL": "ol",
    "EDGE": "de", "DT": "dt", "LB": "lb", "CB": "cb", "S": "s",
    "K": "k", "P": "p",
}


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
    """Returns {position: [{name, score, page, pid} ×3]} —
    top 3 players per position by all-stats avg z-score."""
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
        season_col = "season_year" if "season_year" in df.columns else (
            "season" if "season" in df.columns else None)
        if season_col is None:
            continue
        # Prefer `recent_team`, but some parquets (CB/S 2025) leave
        # it NaN and only populate `team`. Try recent_team first; if
        # that returns nothing, fall back to `team`.
        sub = pd.DataFrame()
        for team_col in ("recent_team", "team"):
            if team_col not in df.columns:
                continue
            sub = df[(df[team_col] == team)
                      & (df[season_col] == season)]
            if not sub.empty:
                break
        if row_filter:
            col, val = row_filter
            if col in sub.columns:
                sub = sub[sub[col] == val]
        if sub.empty:
            continue
        z_cols = [c for c in sub.columns if c.endswith("_z")]
        if not z_cols:
            continue
        sub = sub.copy()
        sub["_avg_z"] = sub[z_cols].mean(axis=1, skipna=True)
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


# ════════════════════════════════════════════════════════════════
# TABS — page body split into four logical groups
# ════════════════════════════════════════════════════════════════

st.markdown("---")
(tab_stats, tab_schedule, tab_tendencies, tab_evolution,
 tab_roster) = st.tabs([
    "📊 Stats & Identity",
    "📅 Schedule & Games",
    "🎯 Tendencies",
    "🧪 Evolution",
    "🦌 Roster",
])


# ─── 📊 STATS & IDENTITY ────────────────────────────────────
with tab_stats:
    st.markdown("### 📊  Phase-by-phase stats")
    st.caption("Raw value + league rank within this season.")
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

    # Comp engine
    st.markdown("---")
    st.markdown(f"### 🔮  Most comparable team-seasons to {team} {season}")
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
        scope=scope, n=3, exclude_same_team=True,
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
                _comp_logo_html = (
                    f'<img src="{comp_logo}" style="height:60px;width:60px;'
                    'object-fit:contain;"/>' if comp_logo else ''
                )
                _diverge_block = (
                    '<div style="font-size:10px;font-weight:800;'
                    'letter-spacing:1.5px;opacity:0.7;'
                    'margin:10px 0 4px 0;">WHERE THEY DIVERGE</div>'
                    f'<div>{c["divergence"].replace("Where they diverge: ", "")}</div>'
                    if c.get("divergence") else ''
                )
                _card_html = (
                    f'<div style="background:linear-gradient(135deg,'
                    f'{comp_primary} 0%,{comp_secondary} 100%);'
                    'border-radius:14px;padding:20px;height:100%;'
                    'color:white;box-shadow:0 4px 12px rgba(0,0,0,0.15);">'
                    '<div style="display:flex;align-items:center;gap:12px;">'
                    f'{_comp_logo_html}'
                    '<div>'
                    '<div style="font-size:11px;opacity:0.7;'
                    f'letter-spacing:1.5px;">SIMILARITY {c["similarity"]*100:.0f}%</div>'
                    '<div style="font-size:22px;font-weight:800;'
                    f'line-height:1.1;">{c["season"]} {comp_name}</div>'
                    '</div></div>'
                    '<div style="margin-top:16px;font-size:13px;'
                    'line-height:1.5;opacity:0.95;">'
                    '<div style="font-size:10px;font-weight:800;'
                    'letter-spacing:1.5px;opacity:0.7;'
                    f'margin-bottom:4px;">WHY SIMILAR</div>'
                    f'<div>{c["reason"]}</div>'
                    f'{_diverge_block}'
                    '</div>'
                    '</div>'
                )
                st.markdown(_card_html, unsafe_allow_html=True)
                st.markdown("")
                if st.button(
                    f"Open {c['season']} {c['team']} →",
                    key=f"go_{c['team']}_{c['season']}",
                    use_container_width=True,
                ):
                    # Stash the target on a NAV-INTENT key (not the
                    # widget keys) — those would raise
                    # StreamlitAPIException because the selectboxes
                    # already rendered this run. First-render logic
                    # at the top of the page applies the intent next
                    # rerun.
                    st.session_state["team_nav_intent"] = (
                        c["team"], int(c["season"])
                    )
                    st.query_params.update({
                        "abbr": c["team"], "season": str(c["season"]),
                    })
                    st.rerun()


# ─── 📅 SCHEDULE & GAMES ───────────────────────────────────
with tab_schedule:
    st.markdown(f"### 📅 Game log — {team} {season}")
    st.caption(
        "Every game with score, result, opponent's pass-defense rank, "
        "and how this team performed against the spread."
    )
    render_game_log(team, int(season))

    _glog = get_team_game_log(team, int(season))
    if not _glog.empty:
        week_options = ["—"] + [
            f"Wk {int(r['week'])} {r['home_away']} {r['opponent']} "
            f"({int(r['team_score'])}–{int(r['opp_score'])}, {r['result']})"
            for _, r in _glog.iterrows()
        ]
        week_choice = st.selectbox(
            "🔬 Pick a week to deep-dive",
            options=week_options,
            index=0,
            key="game_summary_week_pick",
            help="Shows WP arc, top 5 critical plays, and counterfactual "
                 "coverage analysis for each pass — what other coverages "
                 "would have produced against this matchup.",
        )
        if week_choice != "—":
            chosen_idx = week_options.index(week_choice) - 1
            chosen_row = _glog.iloc[chosen_idx]
            from lib_team_game_summary import render_game_summary
            render_game_summary(
                team, int(season), int(chosen_row["week"]),
                game_type=chosen_row.get("game_type", "REG"),
            )


# ─── 🎯 TENDENCIES ─────────────────────────────────────────
with tab_tendencies:
    st.markdown(f"### 🎯 Tendencies — {team} {season}")
    st.caption(
        "What this team does in specific situations. Filter by down, "
        "distance, formation, personnel, coverage faced — see run/pass "
        "split, run direction, and target distribution. Toggle to defense "
        "view to see what opponents do against this team."
    )

    t_side = st.radio(
        "View",
        options=["offense", "defense"],
        horizontal=True,
        format_func=lambda s: "🏈 Offense (when this team has the ball)" if s == "offense"
        else "🛡️ Defense (when opponents have the ball vs this team)",
        key="tendency_side",
    )
    t_opts = get_filter_options(team, int(season), side=t_side)

    f1, f2, f3 = st.columns(3)
    with f1:
        t_downs = st.multiselect(
            "Down", options=[1, 2, 3, 4], default=[],
            placeholder="All downs", key="tendency_downs",
        )
    with f2:
        t_dist = st.multiselect(
            "Distance", options=["Short", "Medium", "Long"],
            default=[], placeholder="All distances",
            key="tendency_dist",
            help="Short ≤3 · Medium 4-7 · Long 8+",
        )
    with f3:
        t_form = st.selectbox(
            "Formation",
            options=["All"] + (t_opts.get("formations") or []),
            index=0, key="tendency_form",
        )

    f4, f5, f6 = st.columns(3)
    with f4:
        t_pers = st.multiselect(
            "Personnel", options=t_opts.get("personnel") or [],
            default=[], placeholder="All personnel",
            key="tendency_pers",
        )
    with f5:
        t_manzone = st.radio(
            "Coverage style", options=["All", "Man", "Zone"],
            horizontal=True, key="tendency_manzone",
        )
    with f6:
        t_rushers = st.radio(
            "Pass rushers", options=["All", "3", "4", "5+"],
            horizontal=True, key="tendency_rushers",
            help="Only meaningful for pass plays",
        )

    t_cov = st.multiselect(
        "Coverage type (defense)",
        options=t_opts.get("coverages") or [],
        default=[], placeholder="All coverages",
        key="tendency_cov",
        help="The coverage shell the defense was in on this play",
    )

    tend = get_team_tendencies(
        team, int(season), side=t_side,
        downs=t_downs or None,
        distance_buckets=t_dist or None,
        formation=t_form if t_form != "All" else None,
        personnel=t_pers or None,
        coverage=t_cov or None,
        manzone=t_manzone if t_manzone != "All" else None,
        rushers=t_rushers if t_rushers != "All" else None,
    )

    if not tend or tend.get("n_plays", 0) < 5:
        st.info(
            f"Only {tend.get('n_plays', 0)} plays match this filter combo "
            "— loosen the filters."
        )
    else:
        n = tend["n_plays"]
        st.caption(f"**{n:,} plays** match this scenario.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Run / Pass split",
                   f"{tend['run_pct']*100:.0f}% run · {tend['pass_pct']*100:.0f}% pass")
        if tend.get("run_epa") is not None:
            m2.metric("Run EPA / play", f"{tend['run_epa']:+.3f}")
        if tend.get("pass_epa") is not None:
            m3.metric("Pass EPA / play", f"{tend['pass_epa']:+.3f}")
        if tend.get("sack_pct") is not None and (tend.get("n_passes") or 0) > 0:
            m4.metric("Sack rate", f"{tend['sack_pct']*100:.1f}%")

        rd = tend.get("run_direction") or {}
        if rd:
            rd_cols = st.columns(3)
            for col, key in zip(rd_cols, ("left", "middle", "right")):
                pct = rd.get(key, 0) * 100
                col.metric(f"Run {key.title()}", f"{pct:.0f}%")

        # Scenario kwargs for player chart drill-down
        _scenario_kwargs = dict(
            downs=t_downs or None,
            distance_buckets=t_dist or None,
            formation=t_form if t_form != "All" else None,
            personnel=t_pers or None,
            coverage=t_cov or None,
            manzone=t_manzone if t_manzone != "All" else None,
            rushers=t_rushers if t_rushers != "All" else None,
        )
        _scenario_label_parts = []
        if t_downs: _scenario_label_parts.append(
            f"down {','.join(str(d) for d in t_downs)}")
        if t_dist: _scenario_label_parts.append(", ".join(t_dist).lower())
        if t_form != "All": _scenario_label_parts.append(t_form.lower())
        if t_pers: _scenario_label_parts.append(", ".join(t_pers))
        if t_cov: _scenario_label_parts.append(", ".join(t_cov))
        if t_manzone != "All": _scenario_label_parts.append(f"{t_manzone} coverage")
        if t_rushers != "All": _scenario_label_parts.append(f"{t_rushers} rushers")
        _scenario_label = " · ".join(_scenario_label_parts) or "all situations"

        # Top players in scenario — clickable, opens filtered chart
        _selected_key = "tendency_selected_player"
        if t_side == "offense" and (
            tend.get("top_targets") or tend.get("top_runners")
            or tend.get("top_passers")
        ):
            cols_count = sum(1 for k in ("top_passers", "top_targets", "top_runners")
                              if tend.get(k))
            cols = st.columns(max(cols_count, 1))
            col_idx = 0
            if tend.get("top_passers"):
                with cols[col_idx]:
                    st.markdown("**Top passers** (dropbacks)")
                    for p in tend["top_passers"]:
                        label = (f"{p.get('name', p['passer_player_id'])} · "
                                 f"{int(p['n'])} dbk · "
                                 f"{p['comp_pct']*100:.0f}% comp · "
                                 f"{p['epa']:+.2f} EPA")
                        if st.button(label,
                                      key=f"qb_btn_{p['passer_player_id']}",
                                      use_container_width=True):
                            st.session_state[_selected_key] = (
                                "QB", p["passer_player_id"], p.get("name", "")
                            )
                col_idx += 1
            if tend.get("top_targets"):
                with cols[col_idx]:
                    st.markdown("**Top targets**")
                    for t in tend["top_targets"]:
                        label = (f"{t.get('name', t['receiver_player_id'])} · "
                                 f"{int(t['n'])} tgt · "
                                 f"{t['catch_pct']*100:.0f}% · "
                                 f"{t['epa']:+.2f}")
                        if st.button(label,
                                      key=f"wr_btn_{t['receiver_player_id']}",
                                      use_container_width=True):
                            st.session_state[_selected_key] = (
                                "WR", t["receiver_player_id"],
                                t.get("name", "")
                            )
                col_idx += 1
            if tend.get("top_runners"):
                with cols[col_idx]:
                    st.markdown("**Top runners**")
                    for r in tend["top_runners"]:
                        label = (f"{r['rusher_player_name']} · "
                                 f"{int(r['n'])} car · "
                                 f"{r['ypc']:.1f} YPC · "
                                 f"{r['epa']:+.2f}")
                        if st.button(label,
                                      key=f"rb_btn_{r['rusher_player_name']}",
                                      use_container_width=True):
                            st.session_state[_selected_key] = (
                                "RB", r['rusher_player_name'],
                                r['rusher_player_name']
                            )
        elif t_side == "defense" and tend.get("top_targets"):
            st.markdown("**Top targets opponents went to** (vs this defense)")
            for t in tend["top_targets"]:
                st.markdown(
                    f"- **{t.get('name', t['receiver_player_id'])}** · "
                    f"{int(t['n'])} targets · "
                    f"{t['catch_pct']*100:.0f}% caught · "
                    f"{t['epa']:+.3f} EPA/tgt"
                )

        # Drill-down chart for selected player
        selected = st.session_state.get(_selected_key)
        if selected and t_side == "offense":
            kind, ident, display_name = selected
            with st.expander(f"📊  {display_name} — drill-down chart "
                              f"(filtered to: {_scenario_label})", expanded=True):
                if st.button("✕ Close", key="tendency_close_drilldown"):
                    st.session_state.pop(_selected_key, None)
                    st.rerun()
                if kind == "QB":
                    render_filtered_throw_map(
                        ident, team, int(season),
                        scenario=_scenario_kwargs,
                        scenario_label=_scenario_label,
                    )
                elif kind == "WR":
                    render_filtered_route_tree(
                        ident, team, int(season),
                        scenario=_scenario_kwargs,
                        scenario_label=_scenario_label,
                    )
                elif kind == "RB":
                    render_filtered_run_profile(
                        ident, team, int(season),
                        scenario=_scenario_kwargs,
                        scenario_label=_scenario_label,
                    )


# ─── 🧪 EVOLUTION ──────────────────────────────────────────
with tab_evolution:
    import plotly.graph_objects as _evo_go
    from pathlib import Path as _EvoPath

    _evo_repo = _EvoPath(__file__).resolve().parent.parent
    _DRIFT_PATH = _evo_repo / "data" / "scheme" / "scheme_shift_team_drift.parquet"
    _CHANGE_PATH = _evo_repo / "data" / "scheme" / "scheme_shift_route_change.parquet"
    _FIT_PATH = _evo_repo / "data" / "scheme" / "scheme_shift_receiver_fit.parquet"
    _FP_PATH = _evo_repo / "data" / "scheme" / "team_passing_fingerprint.parquet"

    _evo_files_present = all(p.exists() for p in [
        _DRIFT_PATH, _CHANGE_PATH, _FIT_PATH, _FP_PATH])

    if not _evo_files_present:
        st.info(
            "Scheme Evolution data not built yet. Run:\n"
            "```\npython tools/build_scheme_shift.py\n```"
        )
    else:
        _drift = pd.read_parquet(_DRIFT_PATH)
        _change = pd.read_parquet(_CHANGE_PATH)
        _fit = pd.read_parquet(_FIT_PATH)
        _fp = pd.read_parquet(_FP_PATH)

        # ── HERO BANNER — JSD verdict + headline narrative ────
        team_drift = _drift[
            (_drift["team"] == team)
            & (_drift["to_season"] == 2025)
        ]
        if team_drift.empty:
            jsd = 0.0
        else:
            jsd = float(team_drift.iloc[0]["jsd"])

        # League-relative: rank this team's drift among all 32
        league_2025 = _drift[_drift["to_season"] == 2025]
        if not league_2025.empty:
            jsd_rank = int((league_2025["jsd"] > jsd).sum()) + 1
            jsd_pctl = 1.0 - (jsd_rank - 1) / max(len(league_2025), 1)
        else:
            jsd_rank = None
            jsd_pctl = 0.5

        if jsd >= 0.030:
            verdict_label = "🌪️ EVOLVING RAPIDLY"
            verdict_color = "#dc2626"
        elif jsd >= 0.020:
            verdict_label = "📈 SHIFTING NOTICEABLY"
            verdict_color = "#ea580c"
        elif jsd >= 0.012:
            verdict_label = "↔️ MODEST SHIFT"
            verdict_color = "#ca8a04"
        else:
            verdict_label = "🪨 MOSTLY STABLE"
            verdict_color = "#16a34a"

        # Hero banner using team theme
        _evo_hero = (
            f'<div style="background:linear-gradient(135deg,{primary} 0%,'
            f'{secondary} 100%);border-radius:14px;padding:22px 26px;'
            f'color:white;box-shadow:0 4px 14px rgba(0,0,0,0.16);'
            f'margin-bottom:18px;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:flex-end;flex-wrap:wrap;gap:18px;">'
            f'<div>'
            f'<div style="font-size:11px;font-weight:700;opacity:0.85;'
            f'letter-spacing:0.6px;">SCHEME EVOLUTION · 2024→2025</div>'
            f'<div style="font-size:30px;font-weight:900;line-height:1;'
            f'letter-spacing:-1px;margin-top:8px;">{verdict_label}</div>'
            f'<div style="font-size:13px;opacity:0.95;margin-top:6px;'
            f'font-weight:500;">'
            f'JSD = {jsd:.3f}'
            f'{f" — rank {jsd_rank}/32" if jsd_rank else ""}'
            f'</div>'
            f'</div>'
            f'<div style="font-size:11px;opacity:0.85;text-align:right;'
            f'max-width:340px;line-height:1.5;">'
            f'JSD measures how much this team\'s route distribution '
            f'shifted year-over-year. Higher = more change. ¹/³² of teams '
            f'are above 0.030; that\'s the line for "rapidly evolving."'
            f'</div>'
            f'</div></div>'
        )
        st.markdown(_evo_hero, unsafe_allow_html=True)

        # ── TWO COLUMNS — chart + top gainers/losers ──────────
        _ec1, _ec2 = st.columns([3, 2])

        # Multi-year route-share chart (2016-2025 with harmonized
        # taxonomy + landmark annotations)
        with _ec1:
            st.markdown("##### 📈  Scheme evolution roadmap "
                        "— 2016 to 2025")
            _HAR_PATH = (_evo_repo / "data" / "scheme"
                         / "team_route_harmonized.parquet")
            _LMK_PATH = (_evo_repo / "data" / "scheme"
                         / "scheme_landmarks.json")

            if not _HAR_PATH.exists():
                st.info(
                    "Multi-year harmonized data not built. Run:\n"
                    "```\npython tools/build_scheme_shift.py\n```"
                )
            else:
                team_fp = pd.read_parquet(_HAR_PATH)
                team_fp = team_fp[team_fp["team"] == team].copy()
                if team_fp.empty:
                    st.info("No route distribution history for this team.")
                else:
                    # Top routes by 2025 share for readability
                    latest = (team_fp[team_fp["season"] == 2025]
                              .sort_values("share", ascending=False))
                    if latest.empty:
                        # Fall back to most recent available season
                        max_yr = int(team_fp["season"].max())
                        latest = (team_fp[team_fp["season"] == max_yr]
                                  .sort_values("share", ascending=False))
                    top_routes = latest["route"].head(8).tolist()

                    # Load landmarks (may be empty for teams not yet
                    # curated)
                    import json as _json
                    landmarks = []
                    try:
                        if _LMK_PATH.exists():
                            with open(_LMK_PATH) as _lf:
                                _lmk = _json.load(_lf)
                            landmarks = _lmk.get(team, [])
                    except Exception:
                        landmarks = []

                    fig_e = _evo_go.Figure()
                    _evo_palette = [
                        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                        "#9467bd", "#8c564b", "#e377c2", "#17becf",
                    ]
                    for color, route in zip(_evo_palette, top_routes):
                        sub = (team_fp[team_fp["route"] == route]
                               .sort_values("season"))
                        if sub.empty:
                            continue
                        fig_e.add_trace(_evo_go.Scatter(
                            x=sub["season"].astype(int),
                            y=sub["share"] * 100,
                            mode="lines+markers",
                            name=route,
                            line=dict(width=2.5, color=color),
                            marker=dict(size=7),
                            hovertemplate=(
                                f"<b>{route}</b><br>"
                                "%{x}: %{y:.1f}%<extra></extra>"),
                        ))

                    # Taxonomy-change marker — vertical band 2022→2023
                    fig_e.add_vrect(
                        x0=2022.5, x1=2023.5,
                        fillcolor="rgba(120,120,120,0.08)",
                        layer="below", line_width=0,
                        annotation_text="route taxonomy "
                                        "refined (harmonized)",
                        annotation_position="top right",
                        annotation_font_size=9,
                        annotation_font_color="#888",
                    )

                    # Landmark vertical lines + annotations
                    LM_TYPE_COLOR = {
                        "hc":                 "#dc2626",
                        "oc":                 "#ea580c",
                        "dc":                 "#9333ea",
                        "qb":                 "#0891b2",
                        "acquisition":        "#16a34a",
                        "scheme_inflection":  "#facc15",
                    }
                    for i, lm in enumerate(landmarks):
                        s = int(lm.get("season", 0))
                        if s < 2016 or s > 2025:
                            continue
                        color = LM_TYPE_COLOR.get(
                            lm.get("type", ""), "#666")
                        # Stagger annotation y to avoid overlap
                        y_anchor = 92 - (i % 3) * 8
                        fig_e.add_vline(
                            x=s, line_color=color, line_width=1.5,
                            line_dash="dot", opacity=0.55)
                        fig_e.add_annotation(
                            x=s, y=y_anchor,
                            text=f"<b>{lm.get('label','')}</b>",
                            showarrow=False,
                            font=dict(size=10, color=color),
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor=color, borderwidth=1,
                            borderpad=2,
                            xanchor="left", xshift=4,
                        )

                    fig_e.update_layout(
                        xaxis=dict(
                            title="Season",
                            dtick=1, gridcolor="#eee",
                            range=[2015.6, 2025.4]),
                        yaxis=dict(title="Share %",
                                    gridcolor="#eee",
                                    range=[0, 100]),
                        height=440,
                        margin=dict(l=50, r=20, t=10, b=40),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(250,250,252,0.6)",
                        legend=dict(
                            orientation="h", yanchor="bottom",
                            y=-0.28, xanchor="left", x=0,
                            font=dict(size=10)),
                    )
                    st.plotly_chart(fig_e,
                                      use_container_width=True,
                                      key=f"team_evo_chart_{team}")

                    if landmarks:
                        st.caption(
                            f"_Annotated with **{len(landmarks)} "
                            "scheme landmarks** for this team._ "
                            "Hover any line for the season's % "
                            "share. Routes harmonized 2016-2025; "
                            "pre/post-2023 differences in route "
                            "vocabulary collapsed for "
                            "comparability."
                        )

                        # Landmark detail table
                        with st.expander(
                                f"📖 The {team_name} scheme story "
                                f"({len(landmarks)} landmarks)",
                                expanded=False):
                            for lm in sorted(landmarks,
                                              key=lambda x: x.get("season", 0)):
                                ic = LM_TYPE_COLOR.get(
                                    lm.get("type", ""), "#666")
                                st.markdown(
                                    f"<div style='border-left:4px "
                                    f"solid {ic};padding:6px 14px;"
                                    f"margin:8px 0;'>"
                                    f"<b style='color:{ic};'>"
                                    f"{lm.get('season')}</b> · "
                                    f"<b>{lm.get('label','')}</b> "
                                    f"<span style='font-size:10px;"
                                    f"color:#888;text-transform:"
                                    f"uppercase;letter-spacing:"
                                    f"0.5px;'>{lm.get('type','')}"
                                    f"</span><br>"
                                    f"<span style='color:#444;"
                                    f"font-size:13px;'>"
                                    f"{lm.get('story','')}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.caption(
                            f"_No landmarks curated for **{team}** "
                            "yet. Edit `data/scheme/"
                            "scheme_landmarks.json` to add HC/OC "
                            "changes, key acquisitions, or scheme "
                            "inflections — they'll annotate the "
                            "chart automatically._"
                        )

        # Top gainers / losers — what's actually changing
        with _ec2:
            st.markdown("##### 🎯  Routes UP / DOWN  (2024→2025)")
            tc = _change[_change["team"] == team].copy()
            if tc.empty:
                st.caption("No change data.")
            else:
                tc["delta_pp"] = (tc["delta"] * 100).round(2)
                up = tc.sort_values("delta_pp", ascending=False).head(4)
                down = tc.sort_values("delta_pp").head(4)
                lines = ["**Running MORE of:**"]
                for _, row in up.iterrows():
                    if row["delta_pp"] <= 0.05:
                        break
                    lines.append(
                        f"• **{row['route']}**  +{row['delta_pp']:.1f} pp "
                        f"({row['share_2024']*100:.1f}% → "
                        f"{row['share_2025']*100:.1f}%)")
                lines.append("")
                lines.append("**Running LESS of:**")
                for _, row in down.iterrows():
                    if row["delta_pp"] >= -0.05:
                        break
                    lines.append(
                        f"• **{row['route']}**  {row['delta_pp']:.1f} pp "
                        f"({row['share_2024']*100:.1f}% → "
                        f"{row['share_2025']*100:.1f}%)")
                st.markdown("\n".join(lines))

        # ── ROSTER SCHEME FIT ─────────────────────────────────
        st.markdown("---")
        st.markdown("##### 🧬  Who's winning the scheme shift")
        st.caption(
            "Each receiver's career route-profile fit against the team's "
            "current (2025) distribution. **Δ vs '24** > 0 means the "
            "offense moved toward this player's strengths."
        )
        team_fit = _fit[_fit["team"] == team].copy()
        if team_fit.empty:
            st.info("No receivers with sufficient career data on this team.")
        else:
            team_fit["fit_2024"] = team_fit["fit_2024"].round(3)
            team_fit["fit_2025"] = team_fit["fit_2025"].round(3)
            team_fit["fit_delta"] = team_fit["fit_delta"].round(3)
            team_fit["Verdict"] = team_fit["fit_delta"].apply(
                lambda d: ("🚀 SCHEME WINNER" if d >= 0.05
                           else "⬆️ small lift" if d > 0
                           else "⬇️ small headwind" if d > -0.05
                           else "⚠️ SCHEME LOSER")
                if pd.notna(d) else "—")
            team_fit_sorted = team_fit.sort_values(
                "fit_delta", ascending=False, na_position="last")
            display = team_fit_sorted[[
                "player_display_name", "position", "Verdict",
                "fit_2024", "fit_2025", "fit_delta",
            ]].rename(columns={
                "player_display_name": "Player",
                "position": "Pos",
                "fit_2024": "Fit '24",
                "fit_2025": "Fit '25",
                "fit_delta": "Δ Fit",
            })
            st.dataframe(display, use_container_width=True,
                            hide_index=True, height=380)

        # ── AUTO-NARRATIVE ────────────────────────────────────
        st.markdown("---")
        st.markdown("##### 📖  The one-paragraph story")
        # Compose a data-driven sentence from the most distinctive bits
        if not _change[_change["team"] == team].empty:
            tc = _change[_change["team"] == team]
            top_up = tc.nlargest(1, "delta")
            top_down = tc.nsmallest(1, "delta")
            up_route = top_up.iloc[0]["route"] if len(top_up) else None
            up_pp = top_up.iloc[0]["delta"] * 100 if len(top_up) else 0
            down_route = top_down.iloc[0]["route"] if len(top_down) else None
            down_pp = top_down.iloc[0]["delta"] * 100 if len(top_down) else 0
        else:
            up_route = down_route = None
            up_pp = down_pp = 0

        winner = loser = None
        if not team_fit.empty:
            tw = team_fit.dropna(subset=["fit_delta"]).sort_values(
                "fit_delta", ascending=False)
            if len(tw) > 0 and tw.iloc[0]["fit_delta"] > 0:
                winner = tw.iloc[0]["player_display_name"]
            tl = team_fit.dropna(subset=["fit_delta"]).sort_values(
                "fit_delta")
            if len(tl) > 0 and tl.iloc[0]["fit_delta"] < 0:
                loser = tl.iloc[0]["player_display_name"]

        magnitude_word = (
            "evolved sharply" if jsd >= 0.030 else
            "shifted noticeably" if jsd >= 0.020 else
            "shifted modestly" if jsd >= 0.012 else
            "stayed largely consistent"
        )
        narrative = (
            f"**{team_name}'s offense has {magnitude_word}** between "
            f"2024 and 2025"
        )
        if up_route and up_pp > 0.5:
            narrative += (
                f" — running **{up_route}** "
                f"{up_pp:+.1f} percentage points more"
            )
            if down_route and down_pp < -0.5:
                narrative += (
                    f" while cutting **{down_route}** "
                    f"{down_pp:+.1f} pp"
                )
        narrative += "."
        if winner:
            narrative += f" **{winner}**'s career profile fits the new look best."
        if loser:
            narrative += f" **{loser}**'s usage projects to shrink as the offense moves away."
        narrative += (
            "  \n_⚠️ OC continuity for 2026 not yet tracked — "
            "this story projects 2025-style scheme to continue. "
            "If the OC changes, expect different numbers; flag via "
            "Camp Battles when ready._"
        )
        st.markdown(narrative)


# ─── 🦌 ROSTER ─────────────────────────────────────────────
with tab_roster:
    st.markdown("### 🦌  Roster — top performers")
    st.caption(
        "Top 3 players per position by all-stats z-score, this team-season. "
        "Click any player to drill into their full rater page."
    )
    _roster = _team_roster_top(team, int(season))
    if not _roster:
        st.info("No roster data for this team-season yet.")
    else:
        pos_keys = list(_roster.keys())
        for i in range(0, len(pos_keys), 4):
            row_keys = pos_keys[i:i + 4]
            cols = st.columns(4)
            for col, pk in zip(cols, row_keys):
                with col:
                    st.markdown(
                        f'<div style="font-size: 11px; font-weight: 800; '
                        f'letter-spacing: 1.5px; opacity: 0.55; '
                        f'margin: 4px 0 4px 4px;">{pk.upper()}</div>',
                        unsafe_allow_html=True,
                    )
                    for r in _roster[pk]:
                        sign = "+" if r["score"] >= 0 else ""
                        label = f"{r['name']}  ·  {sign}{r['score']:.2f}"
                        if st.button(
                            label,
                            key=f"roster_{pk}_{r['pid'] or r['name']}",
                            use_container_width=True,
                            help=f"Open {r['name']}'s page",
                        ):
                            # Land directly on this player's detail
                            # view, not the position leaderboard. We
                            # set:
                            #  1. the team/season filters
                            #  2. the leaderboard's "selected player"
                            #     marker (so it boots into detail mode)
                            #  3. the leaderboard's filter-ctx key to
                            #     match the new (team, season) — without
                            #     this, the auto-clear in
                            #     render_master_detail_leaderboard sees
                            #     a context change and wipes the marker
                            #     we just set.
                            # Then we nuke the team/season widget keys
                            # so the new page picks up our state
                            # instead of any stale widget value.
                            kp = _POS_KEY_PREFIX.get(pk, pk.lower())
                            st.session_state["selected_team"] = team
                            st.session_state["selected_season"] = int(season)
                            st.session_state.pop(
                                "team_selector_widget_v2", None)
                            st.session_state.pop(
                                "season_selector_widget", None)
                            st.session_state[
                                f"{kp}_selected_player_"
                                f"{team}_{int(season)}"
                            ] = r["name"]
                            st.session_state[
                                f"_{kp}_filter_ctx"
                            ] = (str(team), int(season))
                            st.switch_page(r["page"])

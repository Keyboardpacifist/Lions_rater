"""
College team page — destination from the CFB landing grid.

Hero with school colors, position-group strength snapshot, comp
engine (top-3 historical comparable college team-seasons with
generated narrative), and roster click-through to the existing
College mode leaderboards filtered to this school.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from lib_shared import inject_css
from lib_college_team_comps import (
    find_college_team_comps,
    load_college_team_seasons,
)
from lib_college_grid import _TEAM_COLORS, _DEFAULT_COLORS, _readable_text

st.set_page_config(
    page_title="College Team Profile",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "college"


# ── Pick team / season ──────────────────────────────────────────
qp = st.query_params
qp_team = qp.get("team")
qp_season = qp.get("season")

team_df = load_college_team_seasons()
if team_df.empty:
    st.error(
        "College team data not loaded. Run "
        "`python tools/build_college_team_seasons.py`."
    )
    st.stop()

teams_avail = sorted(team_df["team"].unique().tolist())
seasons_avail = sorted(team_df["season"].unique().tolist(), reverse=True)

# Apply pending nav intent (set by the "Open …" comp buttons) before
# the selectbox widgets render. We can't write to widget keys after
# they're instantiated, so the click handler stashes the target in
# `college_nav_intent` and we resolve it here, on the rerun.
_nav = st.session_state.pop("college_nav_intent", None)
if _nav:
    nav_team, nav_season = _nav
    if nav_team in teams_avail:
        st.session_state["college_team_pick"] = nav_team
    if nav_season in seasons_avail:
        st.session_state["college_season_pick"] = int(nav_season)

# Read query params into session_state ONLY on first render. After
# that, the selectbox is the source of truth — pushing stale qp
# values back in on every rerun would overwrite the user's pick
# (st.query_params.update() doesn't sync the URL until *after* the
# rerun, so on the rerun triggered by the selectbox change, qp
# still holds the old value).
if "college_team_pick" not in st.session_state:
    st.session_state["college_team_pick"] = (
        qp_team if (qp_team and qp_team in teams_avail) else "Michigan"
    )
if "college_season_pick" not in st.session_state:
    _s_int = None
    if qp_season:
        try:
            _s_int = int(qp_season)
        except (ValueError, TypeError):
            pass
    st.session_state["college_season_pick"] = (
        _s_int if _s_int in seasons_avail else seasons_avail[0]
    )

c1, c2 = st.columns([2, 1])
with c1:
    team = st.selectbox(
        "School",
        options=teams_avail,
        key="college_team_pick",
    )
with c2:
    season = st.selectbox(
        "Season",
        options=seasons_avail,
        key="college_season_pick",
    )

st.query_params.update({"team": team, "season": str(season)})

primary, secondary = _TEAM_COLORS.get(team, _DEFAULT_COLORS)
text_color = _readable_text(primary)

row = team_df[(team_df["team"] == team) & (team_df["season"] == season)]
if row.empty:
    st.warning(f"No data for {team} in {season}.")
    st.stop()
row = row.iloc[0]

# ── Hero header ────────────────────────────────────────────────
st.markdown(
    f"""
<div style="
    background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    color: {text_color};
">
    <div style="font-size: 38px; font-weight: 900; letter-spacing: -0.5px;
                 line-height: 1;">
        {team}
    </div>
    <div style="font-size: 16px; opacity: 0.85; margin-top: 8px;
                 font-weight: 500; letter-spacing: 1px;">
        {int(season)} SEASON · COLLEGE TEAM PROFILE
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── At-a-glance position-group strengths ──────────────────────
st.markdown("### 📊  Position-group strengths")
st.caption(
    "Z-scored against all college team-seasons in our database. "
    "Higher = stronger relative to the historical pool."
)

GROUPS_DISPLAY = [
    ("qb_strength_z",          "QB"),
    ("wr_te_pass_strength_z",  "Receivers"),
    ("rb_strength_z",          "Running backs"),
    ("ol_strength_z",          "Offensive line"),
    ("def_all_strength_z",     "Defense"),
    ("overall_strength",       "OVERALL"),
]
cols = st.columns(len(GROUPS_DISPLAY))
for col, (stat, label) in zip(cols, GROUPS_DISPLAY):
    with col:
        v = row.get(stat)
        if v is None or pd.isna(v):
            col.metric(label, "—")
        else:
            sign = "+" if v >= 0 else ""
            col.metric(label, f"{sign}{v:.2f}")

# ── Comp engine ────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔮  Most comparable college team-seasons")
st.caption(
    "Cosine similarity across our college team-season database. "
    "Filter set excludes small-sample programs to keep the comp pool "
    "meaningful."
)

scope = st.radio(
    "Compare against:",
    options=["full", "offense", "defense"],
    horizontal=True,
    format_func=lambda s: {"full": "Full team",
                             "offense": "Offensive twins",
                             "defense": "Defensive twins"}[s],
    key="college_comp_scope",
)
include_all_fbs = st.checkbox(
    "🌐  Compare against all FBS teams",
    value=False,
    help=("By default, comps stay within the same tier (Power-4 vs "
          "Group-of-5 vs FCS) so a Big Ten team isn't matched with a "
          "Sun Belt program. Check this to open the pool to every FBS "
          "team-season we have."),
    key="college_comp_all_fbs",
)
comps = find_college_team_comps(
    team=team, season=int(season),
    scope=scope, n=3,
    restrict_to_tier=not include_all_fbs,
)
if not comps:
    st.info("Not enough data to compute comps for this team-season yet.")
else:
    cc = st.columns(3)
    for col, c in zip(cc, comps):
        comp_primary, comp_secondary = _TEAM_COLORS.get(
            c["team"], _DEFAULT_COLORS)
        comp_text = _readable_text(comp_primary)
        with col:
            st.markdown(
                f"""
<div style="
    background: linear-gradient(135deg, {comp_primary} 0%, {comp_secondary} 100%);
    border-radius: 14px;
    padding: 20px;
    height: 100%;
    color: {comp_text};
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
">
    <div style="font-size: 11px; opacity: 0.7; letter-spacing: 1.5px;">
        SIMILARITY {c["similarity"]*100:.0f}%
    </div>
    <div style="font-size: 22px; font-weight: 800; line-height: 1.1;
                 margin-top: 4px;">
        {c["season"]} {c["team"]}
    </div>
    <div style="margin-top: 14px; font-size: 14px; line-height: 1.5;
                 opacity: 0.95;">
        {c["reason"]}
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.markdown("")
            if st.button(
                f"Open {c['season']} {c['team']} →",
                key=f"cgo_{c['team']}_{c['season']}",
                use_container_width=True,
            ):
                # Stash the target on a NAV-INTENT key (not the widget
                # keys) — those would raise StreamlitAPIException
                # because the selectboxes already rendered this run.
                # First-render logic above applies the intent next run.
                st.session_state["college_nav_intent"] = (
                    c["team"], int(c["season"])
                )
                st.query_params.update({
                    "team": c["team"],
                    "season": str(c["season"]),
                })
                st.rerun()


# ── Roster — top performers per position ───────────────────────
st.markdown("---")
st.markdown("### 🎓  Roster — top performers")
st.caption(
    "Top 3 players per position by all-stats average z-score, this "
    "team-season."
)

_ROSTER_SOURCES = [
    ("QB",   "college_qb_all_seasons.parquet"),
    ("WR",   "college_wr_all_seasons.parquet"),
    ("TE",   "college_te_all_seasons.parquet"),
    ("RB",   "college_rb_all_seasons.parquet"),
    ("OL",   "college_ol_roster.parquet"),
    ("Defense", "college_def_all_seasons.parquet"),
]

# Defense rolls up multiple positions into one parquet — map each
# defender's listed_position to the College mode position key so
# clicks land on the correct leaderboard.
_DEF_POS_MAP = {
    "CB": "CB",  "DB": "S",  "S": "S",
    "DE": "DE",  "EDGE": "DE",
    "DT": "DT",  "DL": "DT", "NT": "DT",
    "LB": "LB",  "ILB": "LB", "OLB": "LB",
}


@st.cache_data(show_spinner=False)
def _college_roster_top(team: str, season: int) -> dict:
    out: dict[str, list] = {}
    for pos_label, fname in _ROSTER_SOURCES:
        path = DATA / fname
        if not path.exists():
            continue
        try:
            df = pl.read_parquet(path).to_pandas()
        except Exception:
            continue
        if "team" not in df.columns or "season" not in df.columns:
            continue
        sub = df[(df["team"] == team) & (df["season"] == season)]
        if sub.empty:
            continue
        z_cols = [c for c in sub.columns if c.endswith("_z")]
        if not z_cols:
            continue
        sub = sub.copy()
        sub["_avg_z"] = sub[z_cols].mean(axis=1, skipna=True)
        for name_col in ("player_name", "name", "athlete", "player"):
            if name_col in sub.columns:
                break
        else:
            name_col = None
        if name_col is None:
            continue
        sub = sub.dropna(subset=["_avg_z", name_col])
        if sub.empty:
            continue
        top = sub.sort_values("_avg_z", ascending=False).head(3)
        rows = []
        for _, r in top.iterrows():
            if pos_label == "Defense":
                _listed = (r.get("listed_position")
                           or r.get("pos_group") or "LB")
                cm_pos = _DEF_POS_MAP.get(str(_listed).upper(), "LB")
            else:
                cm_pos = pos_label
            rows.append({
                "name": str(r[name_col]),
                "score": float(r["_avg_z"]),
                "cm_pos": cm_pos,
            })
        out[pos_label] = rows
    return out


_roster = _college_roster_top(team, int(season))
if not _roster:
    st.info("No roster data for this team-season yet.")
else:
    st.caption("Click any card to open the full College mode profile.")

    from lib_shared import college_theme, render_player_card

    # Per-position stat tiles for the card. Each entry is
    # (col, fmt, label) — render_player_card reads col from view_row.
    _ROSTER_STAT_SPECS: dict[str, list[tuple[str, str, str]]] = {
        "QB":  [("pass_yards", "{:.0f}", "Pass yds"),
                ("pass_tds",   "{:.0f}", "TD"),
                ("pass_ints",  "{:.0f}", "INT"),
                ("pass_ypa",   "{:.1f}", "Y/Att"),
                ("pass_pct",   "{:.1%}",  "Comp%")],
        "WR":  [("receptions", "{:.0f}", "Rec"),
                ("rec_yards",  "{:.0f}", "Rec yds"),
                ("rec_tds",    "{:.0f}", "TD"),
                ("rec_ypr",    "{:.1f}", "Y/R")],
        "TE":  [("receptions", "{:.0f}", "Rec"),
                ("rec_yards",  "{:.0f}", "Rec yds"),
                ("rec_tds",    "{:.0f}", "TD"),
                ("rec_ypr",    "{:.1f}", "Y/R")],
        "RB":  [("rush_carries", "{:.0f}", "Car"),
                ("rush_yards",   "{:.0f}", "Rush yds"),
                ("rush_tds",     "{:.0f}", "TD"),
                ("rush_ypc",     "{:.1f}", "YPC"),
                ("receptions",   "{:.0f}", "Rec")],
        "OL":  [],
        "Defense": [("tackles_total",     "{:.0f}", "Tkl"),
                    ("tfl",                "{:.1f}", "TFL"),
                    ("sacks",              "{:.1f}", "Sk"),
                    ("interceptions",      "{:.0f}", "INT"),
                    ("passes_deflected",   "{:.0f}", "PD")],
    }

    @st.cache_data(show_spinner=False)
    def _roster_row_for(team_arg: str, season_arg: int,
                          fname: str, name: str) -> dict | None:
        """Pull the full row for one player from the position parquet
        — needed to feed render_player_card with the raw stat
        columns it formats into tile values."""
        path = DATA / fname
        if not path.exists():
            return None
        df = pl.read_parquet(path).to_pandas()
        if "team" not in df.columns or "season" not in df.columns:
            return None
        for name_col in ("player_name", "name", "athlete", "player"):
            if name_col in df.columns:
                break
        else:
            return None
        m = df[(df["team"] == team_arg) & (df["season"] == season_arg)
               & (df[name_col] == name)]
        if m.empty:
            return None
        return dict(m.iloc[0])

    _theme = college_theme(team)

    pos_keys = list(_roster.keys())
    # One position per row — full-width banner cards are wide; placing
    # 3 across crowded the stat tiles. Keep top-3 stacked under each
    # position label.
    for pk in pos_keys:
        st.markdown(
            f"""
<div style="font-size: 13px; font-weight: 800; letter-spacing: 1.5px;
             opacity: 0.65; margin: 16px 0 6px 4px;">
    {pk.upper()}
</div>
""",
            unsafe_allow_html=True,
        )
        # Find the file for this position group
        fname = next((f for label, f in _ROSTER_SOURCES
                       if label == pk), None)
        stat_specs = _ROSTER_STAT_SPECS.get(pk, [])
        for r in _roster[pk]:
            full_row = (_roster_row_for(team, int(season), fname,
                                            r["name"]) if fname else None)
            view_row = pd.Series(full_row) if full_row else pd.Series()
            render_player_card(
                player_name=r["name"],
                position_label=r["cm_pos"],
                season_str=f"{team} · {int(season)}",
                score=r["score"],
                stat_specs=stat_specs,
                view_row=view_row,
                team_label=team,
                primary_color=_theme.get("primary"),
                secondary_color=_theme.get("secondary"),
                logo_url=_theme.get("logo") or "",
            )
            btn_key = f"roster_{team}_{season}_{pk}_{r['name']}"
            if st.button("Open profile  →", key=btn_key,
                            use_container_width=True):
                st.session_state["college_school_v2"] = team
                st.session_state["college_season_landing"] = int(season)
                st.session_state["college_position_top"] = r["cm_pos"]
                st.session_state["expand_college_player"] = r["name"]
                st.session_state[f"lb_selected_{r['cm_pos']}"] = r["name"]
                st.session_state["mode_toggle"] = "College"
                # Match app.py's _filter_ctx tuple exactly — its
                # auto-clear wipes lb_selected markers when prev_ctx
                # != cur_ctx, so the conference field has to use
                # the same default label app.py computes.
                st.session_state["_college_filter_ctx"] = (
                    team, "🏈 All conferences",
                    int(season), r["cm_pos"],
                )
                st.session_state["college_conference"] = (
                    "🏈 All conferences")
                # Force-detail token — app.py applies this AFTER all
                # auto-clears so the marker survives any state-wipe
                # logic. Survives one navigation, then gets popped.
                st.session_state["_force_college_detail"] = {
                    "school": team,
                    "season": int(season),
                    "position": r["cm_pos"],
                    "player": r["name"],
                }
                st.switch_page("app.py")

# ── Click-through to existing College mode leaderboards ───────
st.markdown("---")
_, mid, _ = st.columns([1, 2, 1])
with mid:
    if st.button(
        f"📋  See full {team} leaderboards in College mode",
        use_container_width=True,
    ):
        st.session_state["college_school_v2"] = team
        st.session_state["mode_toggle"] = "College"
        st.switch_page("app.py")

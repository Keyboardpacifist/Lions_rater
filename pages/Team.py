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
default_team = qp.get("abbr", "DET")
default_season = qp.get("season")

team_df = load_team_seasons()
if team_df.empty:
    st.error("Team data not loaded. Run `python tools/build_team_seasons.py`.")
    st.stop()

teams_avail = sorted(team_df["team"].unique().tolist())
seasons_avail = sorted(team_df["season"].unique().tolist(), reverse=True)

c1, c2 = st.columns([2, 1])
with c1:
    team = st.selectbox(
        "Team",
        options=teams_avail,
        index=(teams_avail.index(default_team)
                if default_team in teams_avail else 0),
        key="team_pick",
    )
with c2:
    season = st.selectbox(
        "Season",
        options=seasons_avail,
        index=(0 if default_season is None
                else seasons_avail.index(int(default_season))
                if (default_season and int(default_season) in seasons_avail)
                else 0),
        key="season_pick",
    )

# Update URL so the page is shareable
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

# ── Hero header ────────────────────────────────────────────────
st.markdown(
    f"""
<div style="
    background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    display: flex;
    align-items: center;
    gap: 24px;
    color: white;
">
    {f'<img src="{logo}" style="height: 110px; width: 110px; object-fit: contain; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.35));"/>' if logo else ''}
    <div>
        <div style="font-size: 38px; font-weight: 900; letter-spacing: -0.5px; line-height: 1;">
            {team_name}
        </div>
        <div style="font-size: 16px; opacity: 0.85; margin-top: 8px;
                     font-weight: 500; letter-spacing: 1px;">
            {season} SEASON · TEAM PROFILE
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── At-a-glance stat row ───────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Points/game",
           f"{row.get('points_per_game', float('nan')):.1f}",
           help="Regular season scoring average.")
m2.metric("Points allowed/game",
           f"{row.get('points_allowed_per_game', float('nan')):.1f}",
           help="Defensive scoring allowed.")
m3.metric("Off EPA/play",
           f"{row.get('off_epa_per_play', float('nan')):+.3f}",
           help="Expected points added per offensive play.")
m4.metric("Def EPA allowed/play",
           f"{row.get('def_epa_per_play', float('nan')):+.3f}",
           delta_color="inverse",
           help="Per-play EPA allowed (lower = better).")

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
                st.query_params.update({
                    "abbr": c["team"],
                    "season": str(c["season"]),
                })
                st.rerun()

# ── Roster / position click-through ────────────────────────────
st.markdown("---")
st.markdown("### 🦌  Drill into the roster")
st.caption(
    "Click a position to see this team's players ranked at that position. "
    "From there, build presets and trading cards for individual players."
)

POSITIONS = [
    ("QB",      "QB"),
    ("WR/TE",   "WR"),  # WR page is the entry; TE has its own page too
    ("RB",      "2_Running_backs"),
    ("OL",      "3_Offensive_Line"),
    ("DE",      "DE"),
    ("DT",      "DT"),
    ("LB",      "LB"),
    ("CB",      "CB"),
    ("S",       "Safety."),
    ("K",       "Kicker"),
    ("P",       "Punter"),
]

cols = st.columns(6)
for i, (label, page_slug) in enumerate(POSITIONS):
    with cols[i % 6]:
        if st.button(label,
                      key=f"goto_{page_slug}",
                      use_container_width=True,
                      help=f"Open the {label} rater"):
            # Stash team selection so the position page picks it up if it
            # supports cross-page team prefilling
            st.session_state["team_pick_from_team_page"] = team
            st.session_state["season_pick_from_team_page"] = int(season)
            st.switch_page(f"pages/{page_slug}.py")

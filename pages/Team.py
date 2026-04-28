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

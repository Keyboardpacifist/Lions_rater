"""2027 NFL Draft research — fan-built prospect big board.

Surfaces 2025-season FBS production filtered to 2027-eligible
prospects, with composite z-score per player. Click any prospect
to jump to their College mode detail (existing infrastructure).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib_draft_2027 import load_2027_prospects
from lib_shared import inject_css

st.set_page_config(
    page_title="2027 NFL Draft",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

# ── Hero ────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0a1f4a 0%,#1e3a8a 100%);
            border-radius:16px;padding:26px 30px;margin-bottom:18px;
            color:white;box-shadow:0 4px 14px rgba(0,0,0,0.18);">
  <div style="font-size:32px;font-weight:900;letter-spacing:-0.5px;
              line-height:1;">
    🏈 2027 NFL Draft — Big Board
  </div>
  <div style="font-size:14px;opacity:0.88;margin-top:8px;font-weight:500;
              letter-spacing:0.4px;">
    Fan-built prospect rankings · 2025-season FBS production · composite
    z-scores against position peers · click any prospect for the full
    college profile
  </div>
</div>
""", unsafe_allow_html=True)

df = load_2027_prospects()
if df.empty:
    st.error("No prospect data available — confirm the college parquets "
             "are loaded.")
    st.stop()

# ── Filters ─────────────────────────────────────────────────────
positions_avail = ["All", "QB", "RB", "WR", "TE", "OL", "DE", "DT",
                    "LB", "CB", "S"]
confs_avail = ["All"] + sorted(df["conference"].dropna().unique().tolist())
schools_avail = ["All"] + sorted(df["team"].dropna().unique().tolist())

f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
with f1:
    pos_filter = st.selectbox("Position", positions_avail,
                                key="draft_pos_filter")
with f2:
    conf_filter = st.selectbox("Conference", confs_avail,
                                 key="draft_conf_filter")
with f3:
    school_filter = st.selectbox("School", schools_avail,
                                   key="draft_school_filter")
with f4:
    class_filter = st.selectbox(
        "Class",
        ["All eligible",
         "True junior (2024 recruit)",
         "Senior+ (2023 recruit or earlier)",
         "Eligibility verified only"],
        key="draft_class_filter",
    )

filt = df.copy()
if pos_filter != "All":
    filt = filt[filt["position"] == pos_filter]
if conf_filter != "All":
    filt = filt[filt["conference"] == conf_filter]
if school_filter != "All":
    filt = filt[filt["team"] == school_filter]
if class_filter == "True junior (2024 recruit)":
    filt = filt[filt["recruit_year"] == 2024]
elif class_filter == "Senior+ (2023 recruit or earlier)":
    filt = filt[filt["recruit_year"] <= 2023]
elif class_filter == "Eligibility verified only":
    filt = filt[filt["eligibility_verified"]]

st.caption(
    f"**{len(filt):,} prospects** match · _Eligibility based on roster + "
    "recruiting year where available — players without recruiting data "
    "are included by default since the data only covers ~36% of the pool. "
    "Use the 'Eligibility verified only' class filter for the strict pool._"
)


# ── Click-through helper ────────────────────────────────────────
def _open_college_profile(team: str, season: int, position: str,
                            player: str) -> None:
    """Push session state for College mode and switch_page to app.py.
    Same handler the College team page uses for its roster click-through."""
    st.session_state["college_school_v2"] = team
    st.session_state["college_season_landing"] = int(season)
    st.session_state["college_position_top"] = position
    st.session_state["expand_college_player"] = player
    st.session_state[f"lb_selected_{position}"] = player
    st.session_state["mode_toggle"] = "College"
    st.session_state["_college_filter_ctx"] = (
        team, None, int(season), position,
    )
    st.switch_page("app.py")


def _render_prospect_row(rank: int, r: pd.Series, key_prefix: str) -> None:
    cols = st.columns([0.5, 5, 1.4, 1.4, 1.6])
    with cols[0]:
        st.markdown(
            f"<div style='font-size:1.5rem;font-weight:800;color:#666;"
            f"padding-top:6px;'>{rank}</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        sign = "+" if r["composite_z"] >= 0 else ""
        verified = "✓" if r.get("eligibility_verified") else "?"
        st.markdown(
            f"**{r['player']}** · `{r['position']}` · "
            f"{r['team']} _{r.get('conference', '') or ''}_"
        )
        st.caption(
            f"{sign}{r['composite_z']:.2f} composite z · "
            f"{verified} eligibility"
        )
    with cols[2]:
        if pd.notna(r.get("stars")):
            stars = "★" * int(r["stars"])
            st.caption(f"{stars} ({int(r['stars'])}-star)")
        else:
            st.caption("_no recruit data_")
    with cols[3]:
        if pd.notna(r.get("ranking")):
            st.caption(f"HS rank: #{int(r['ranking'])}")
        elif pd.notna(r.get("recruit_year")):
            st.caption(f"Class of {int(r['recruit_year'])}")
        else:
            st.caption("—")
    with cols[4]:
        if st.button(
            "Profile →",
            key=f"{key_prefix}_{r['player_id']}",
            use_container_width=True,
        ):
            _open_college_profile(
                team=r["team"], season=2025, position=r["position"],
                player=r["player"],
            )


# ── Tabs ────────────────────────────────────────────────────────
tab_board, tab_pos, tab_school, tab_conf = st.tabs([
    "📋 Big Board",
    "🎯 By Position",
    "🏫 By School",
    "🏟 By Conference",
])

# 📋 Big Board — top-100 across positions
with tab_board:
    st.markdown("### Top prospects (composite z, all positions)")
    st.caption(
        "Ranked by raw composite z-score — no positional-value multiplier "
        "yet. QBs and OTs typically under-rank here vs traditional draft "
        "boards because position value isn't applied. Adjust by filtering "
        "to a single position or coming back when the v1.1 board with "
        "positional value lands."
    )
    sorted_df = filt.sort_values("composite_z", ascending=False).head(100)
    if sorted_df.empty:
        st.info("No prospects match your filters.")
    else:
        for i, (_, r) in enumerate(sorted_df.iterrows(), start=1):
            _render_prospect_row(i, r, "bb")
            if i < len(sorted_df):
                st.divider()

# 🎯 By Position
with tab_pos:
    st.markdown("### Top 15 per position")
    st.caption(
        "Same composite z, grouped by position. Each section is "
        "expandable. Tier 1 prospects are typically those with z ≥ 1.5; "
        "tier 2 with z ≥ 1.0; below that gets noisier."
    )
    positions_order = ["QB", "RB", "WR", "TE", "OL",
                        "DE", "DT", "LB", "CB", "S"]
    for pos in positions_order:
        pos_df = filt[filt["position"] == pos].sort_values(
            "composite_z", ascending=False
        ).head(15)
        if pos_df.empty:
            continue
        with st.expander(
            f"**{pos}** — top {len(pos_df)} of "
            f"{(filt['position'] == pos).sum()} eligible",
            expanded=(pos == "QB"),
        ):
            for i, (_, r) in enumerate(pos_df.iterrows(), start=1):
                _render_prospect_row(i, r, f"pos_{pos}")
                if i < len(pos_df):
                    st.divider()

# 🏫 By School — prospect counts + drill-down
with tab_school:
    st.markdown("### Schools by 2027 prospect count")
    st.caption(
        "How many 2027-eligible players each school has in our composite "
        "pool, plus the average composite z of their top 5 prospects. "
        "Click any school in the filter at the top to drill down."
    )
    # Top-5 per school for the avg-z calculation
    sch = (filt.sort_values("composite_z", ascending=False)
           .groupby("team", as_index=False)
           .agg(prospects=("player_id", "count"),
                top5_avg_z=("composite_z",
                              lambda s: s.head(5).mean())))
    sch["top5_avg_z"] = sch["top5_avg_z"].round(2)
    sch = sch.sort_values(["top5_avg_z", "prospects"],
                            ascending=[False, False]).head(50)
    sch = sch.rename(columns={
        "team": "School",
        "prospects": "# eligible",
        "top5_avg_z": "Top-5 avg composite z",
    })
    st.dataframe(sch, use_container_width=True, hide_index=True)

# 🏟 By Conference — same idea, conference scope
with tab_conf:
    st.markdown("### Conferences by 2027 prospect strength")
    st.caption(
        "Counts and average top-25-prospect composite z per conference. "
        "A rough read on which conference is sending the most talent to "
        "the 2027 draft."
    )
    cnf = (filt.sort_values("composite_z", ascending=False)
           .groupby("conference", as_index=False)
           .agg(prospects=("player_id", "count"),
                top25_avg_z=("composite_z",
                              lambda s: s.head(25).mean())))
    cnf["top25_avg_z"] = cnf["top25_avg_z"].round(2)
    cnf = cnf.sort_values(["top25_avg_z", "prospects"],
                            ascending=[False, False])
    cnf = cnf.rename(columns={
        "conference": "Conference",
        "prospects": "# eligible",
        "top25_avg_z": "Top-25 avg composite z",
    })
    st.dataframe(cnf, use_container_width=True, hide_index=True)


# ── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Composite z-scores are the average of position-specific z-scores "
    "(production stats z-scored against FBS peers, 2025 season). "
    "Eligibility = recruit-year ≤ 2024 AND not already drafted; "
    "players without recruiting data are included by default since "
    "the dataset only covers ~36% of FBS rosters. "
    "Data via [nflverse](https://github.com/nflverse) · "
    "[CFBData](https://collegefootballdata.com)."
)

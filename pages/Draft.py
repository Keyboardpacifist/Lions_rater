"""2027 NFL Draft research — fan-built prospect big board.

Big Board + By Position tabs are driven by the expert seed board
(top-100, generated from `tools/seed_draft_2027_consensus.py`).
For each pick we attach our composite z-score from the College mode
production pool, so fans see expert rank alongside our model. Click
a prospect → jumps to their full College profile.

Schools tab + Conference tab summarize the same expert pool.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib_draft_2027 import (
    attach_composite_z,
    attach_nfl_comps,
    load_2027_prospects,
    load_consensus_board,
)
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
    Top 100 expert big board · seeded April 2026 · we attach our composite
    z-score from 2025 college production · click any prospect for the
    full college profile
  </div>
</div>
""", unsafe_allow_html=True)

consensus = load_consensus_board()
prospects = load_2027_prospects()

if consensus.empty:
    st.error("Consensus board not loaded — run "
             "`python tools/seed_draft_2027_consensus.py`.")
    st.stop()

board = attach_composite_z(consensus, prospects)

# NFL comps — cached on a signature of the consensus list so it
# invalidates if you reorder/edit the seed.
_board_sig = tuple((int(r["expert_rank"]), r["player"], r["school"])
                    for _, r in consensus.iterrows())
nfl_comps_df = attach_nfl_comps(_board_sig)
if not nfl_comps_df.empty:
    board = board.merge(nfl_comps_df, on="expert_rank", how="left")


# ── Filters ─────────────────────────────────────────────────────
positions_avail = ["All", "QB", "RB", "WR", "TE", "OL", "DE", "DT",
                    "LB", "CB", "S"]
schools_avail = ["All"] + sorted(board["school"].dropna().unique().tolist())

f1, f2 = st.columns([2, 2])
with f1:
    pos_filter = st.selectbox("Position", positions_avail,
                                key="draft_pos_filter")
with f2:
    school_filter = st.selectbox("School", schools_avail,
                                   key="draft_school_filter")

filt = board.copy()
if pos_filter != "All":
    filt = filt[filt["position"] == pos_filter]
if school_filter != "All":
    filt = filt[filt["school"] == school_filter]

st.caption(
    f"**{len(filt):,} of {len(board)} consensus prospects** match · "
    "_The expert board is one published source; composite z is our "
    "z-score against 2025 FBS peers. Mismatches between the two are "
    "the conversation._"
)


# ── Click-through helper ────────────────────────────────────────
def _open_college_profile(team: str, season: int, position: str,
                            player: str) -> None:
    """Push session state for College mode and switch_page to app.py.
    Same handler the College Team page uses for its roster click-through."""
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


def _render_prospect_row(rank_label: str, r: pd.Series,
                            key_prefix: str) -> None:
    cols = st.columns([0.7, 5, 1.6, 1.4, 1.4])
    with cols[0]:
        st.markdown(
            f"<div style='font-size:1.4rem;font-weight:800;color:#1e3a8a;"
            f"padding-top:6px;line-height:1.1;'>{rank_label}</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        actual_team = r.get("team")
        if (pd.notna(actual_team)
                and str(actual_team).lower() != str(r["school"]).lower()):
            sub_school = (f"{r['school']} on board · "
                          f"_{actual_team}_ in our 2025 stats")
        else:
            sub_school = r["school"]
        st.markdown(
            f"**{r['player']}** · `{r['board_position']}` · {sub_school}"
        )
        if pd.notna(r.get("composite_z")):
            sign = "+" if r["composite_z"] >= 0 else ""
            verified = "✓" if r.get("eligibility_verified") else "?"
            st.caption(
                f"Our composite z: {sign}{r['composite_z']:.2f} · "
                f"{verified} 2027 eligibility"
            )
        else:
            st.caption(
                "_No 2025 production data yet (likely a true freshman "
                "from the 2025 recruit class)._"
            )
    with cols[2]:
        if pd.notna(r.get("stars")):
            stars = "★" * int(r["stars"])
            st.caption(f"{stars} ({int(r['stars'])}-star recruit)")
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
        if pd.notna(r.get("player_id")):
            target_team = (r.get("team") if pd.notna(r.get("team"))
                           else r["school"])
            target_pos = r["position"]
            if st.button(
                "Profile →",
                key=f"{key_prefix}_{int(r['expert_rank'])}",
                use_container_width=True,
            ):
                _open_college_profile(
                    team=str(target_team),
                    season=2025,
                    position=target_pos,
                    player=r["player"],
                )
        else:
            st.caption("_profile pending_")

    # ── NFL statistical comps (collapsed by default) ──────────
    comps = r.get("nfl_comps")
    if isinstance(comps, list) and comps:
        top_comp_label = r.get("top_comp") or "—"
        with st.expander(
            f"🎯 Statistical composition most like: {top_comp_label}",
            expanded=False,
        ):
            for c in comps:
                yr = c.get("draft_year") or "—"
                rd = c.get("draft_round") or "—"
                pk = c.get("draft_overall") or "—"
                st.markdown(
                    f"**{c['similarity']*100:.0f}%** · {c['player']} "
                    f"({yr} {c['school']}) → "
                    f"R{rd} P{pk} {c['nfl_team']}"
                )
            # Hit-rate distribution
            r1 = r.get("hit_rate_r1")
            r2_3 = r.get("hit_rate_r2_3")
            r4_7 = r.get("hit_rate_r4_7")
            if pd.notna(r1):
                st.caption(
                    f"_Top-50 most-similar profiles: "
                    f"**{r1*100:.0f}% went R1** · "
                    f"{r2_3*100:.0f}% R2-3 · "
                    f"{r4_7*100:.0f}% R4-7. "
                    "Empirical hit-rate framing — not a prediction._"
                )
    elif r["position"] == "OL":
        st.caption(
            "🛠 _OL NFL comps coming v1.1 — historical OL linkage "
            "parquet not yet built._"
        )


# ── Tabs ────────────────────────────────────────────────────────
tab_board, tab_pos, tab_school, tab_conf = st.tabs([
    "📋 Big Board",
    "🎯 By Position",
    "🏫 By School",
    "🏟 By Conference",
])

# 📋 Big Board — expert rank order
with tab_board:
    if filt.empty:
        st.info("No prospects match your filters.")
    else:
        for _, r in filt.sort_values("expert_rank").iterrows():
            _render_prospect_row(f"{int(r['expert_rank'])}", r, "bb")
            st.divider()

# 🎯 By Position — grouped by normalized position
with tab_pos:
    st.markdown("### Top prospects per position (expert board)")
    st.caption(
        "Expert ranks grouped by position. Each section is "
        "expandable; QB opens by default."
    )
    positions_order = ["QB", "RB", "WR", "TE", "OL",
                        "DE", "DT", "LB", "CB", "S"]
    for pos in positions_order:
        pos_df = filt[filt["position"] == pos].sort_values("expert_rank")
        if pos_df.empty:
            continue
        with st.expander(
            f"**{pos}** — {len(pos_df)} prospects on the expert board",
            expanded=(pos == "QB"),
        ):
            for i, (_, r) in enumerate(pos_df.iterrows(), start=1):
                rank_html = (
                    f"#{i}<br><span style='font-size:0.7rem;color:#888;"
                    f"font-weight:500;'>(overall #{int(r['expert_rank'])})"
                    f"</span>"
                )
                _render_prospect_row(rank_html, r, f"pos_{pos}")
                st.divider()

# 🏫 By School
with tab_school:
    st.markdown("### Schools by expert-board prospects")
    st.caption(
        "Count of consensus-board prospects per school. Use the school "
        "filter at top to drill in."
    )
    sch = (filt.groupby("school", as_index=False)
           .agg(prospects=("player", "count"),
                top_rank=("expert_rank", "min"),
                avg_rank=("expert_rank", "mean")))
    sch["avg_rank"] = sch["avg_rank"].round(1)
    sch = sch.sort_values(["prospects", "top_rank"],
                            ascending=[False, True]).rename(columns={
        "school": "School",
        "prospects": "# on board",
        "top_rank": "Highest pick",
        "avg_rank": "Avg pick",
    })
    st.dataframe(sch, use_container_width=True, hide_index=True)

# 🏟 By Conference
with tab_conf:
    st.markdown("### Conferences by expert-board prospects")
    st.caption(
        "Conference is from our 2025-season parquet match. Prospects "
        "who didn't match (true freshmen with no production yet) are "
        "pooled as 'unmatched'."
    )
    cnf_pool = filt.copy()
    cnf_pool["conference"] = cnf_pool["conference"].fillna("(unmatched)")
    cnf = (cnf_pool.groupby("conference", as_index=False)
           .agg(prospects=("player", "count"),
                top_rank=("expert_rank", "min"),
                avg_rank=("expert_rank", "mean")))
    cnf["avg_rank"] = cnf["avg_rank"].round(1)
    cnf = cnf.sort_values(["prospects", "top_rank"],
                            ascending=[False, True]).rename(columns={
        "conference": "Conference",
        "prospects": "# on board",
        "top_rank": "Highest pick",
        "avg_rank": "Avg pick",
    })
    st.dataframe(cnf, use_container_width=True, hide_index=True)


# ── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Expert board: hand-curated top-100 (April 2026). Composite z: our "
    "z-score across 2025-season FBS production stats joined by name + "
    "school (with a name-only fallback for transfer-portal moves). "
    "Position keys: OT/IOL → OL · DL → DT · EDGE → DE. Data via "
    "[nflverse](https://github.com/nflverse) · "
    "[CFBData](https://collegefootballdata.com)."
)

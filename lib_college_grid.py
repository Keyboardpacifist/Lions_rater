"""Stylized CFB team grid — landing-page front door for College mode.

Top 5 programs per Power-4 conference + Notre Dame + 3 other notables
(24 teams total visible). "Show all" button expands to every school
in the data.
"""
from __future__ import annotations

import streamlit as st

# ── Featured tier (24 teams) ──────────────────────────────────
# Conference top-5s per current alignment, plus the must-have non-P4s.
_FEATURED = {
    "SEC":          ["Alabama",   "Georgia",       "Texas",        "LSU",          "Tennessee"],
    "Big Ten":      ["Ohio State","Michigan",      "Penn State",   "USC",          "Oregon"],
    "ACC":          ["Clemson",   "Florida State", "Miami",        "North Carolina","Louisville"],
    "Big 12":       ["Utah",      "BYU",           "Iowa State",   "Kansas State", "Texas Tech"],
    "Independents & notables": ["Notre Dame", "Boise State", "Army", "Navy"],
}

# Hand-picked program colors (primary, secondary). Used until we add
# an authoritative college team_colors.json source.
_TEAM_COLORS: dict[str, tuple[str, str]] = {
    "Alabama":        ("#9E1B32", "#FFFFFF"),
    "Georgia":        ("#BA0C2F", "#000000"),
    "Texas":          ("#BF5700", "#FFFFFF"),
    "LSU":            ("#461D7C", "#FDD023"),
    "Tennessee":      ("#FF8200", "#FFFFFF"),
    "Ohio State":     ("#BB0000", "#666666"),
    "Michigan":       ("#00274C", "#FFCB05"),
    "Penn State":     ("#041E42", "#FFFFFF"),
    "USC":            ("#990000", "#FFCC00"),
    "Oregon":         ("#154733", "#FEE123"),
    "Clemson":        ("#F66733", "#522D80"),
    "Florida State":  ("#782F40", "#CEB888"),
    "Miami":          ("#F47321", "#005030"),
    "North Carolina": ("#7BAFD4", "#FFFFFF"),
    "Louisville":     ("#AD0000", "#000000"),
    "Utah":           ("#CC0000", "#FFFFFF"),
    "BYU":            ("#002E5D", "#FFFFFF"),
    "Iowa State":     ("#C8102E", "#F1BE48"),
    "Kansas State":   ("#512888", "#A7A9AC"),
    "Texas Tech":     ("#CC0000", "#000000"),
    "Notre Dame":     ("#0C2340", "#C99700"),
    "Boise State":    ("#0033A0", "#D64309"),
    "Army":           ("#1F1F1F", "#D4BF7C"),
    "Navy":           ("#00205C", "#C5B783"),
}
_DEFAULT_COLORS = ("#1F2A44", "#0B1730")


def _route_to_team(team: str, on_pick_session_key: str) -> None:
    """Click handler for grid tiles — set session_state for both the
    legacy school filter (so College mode still works as a fallback)
    AND the CollegeTeam page selectbox, then navigate."""
    st.session_state[on_pick_session_key] = team
    st.session_state["college_team_pick"] = team
    st.query_params.update({"team": team})
    try:
        st.switch_page("pages/CollegeTeam.py")
    except Exception:
        # Fallback if the page hasn't been added yet
        st.rerun()


def _readable_text(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "white"
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.6 else "#0d1530"


def _team_tile_html(team: str) -> str:
    primary, secondary = _TEAM_COLORS.get(team, _DEFAULT_COLORS)
    text_color = _readable_text(primary)
    return f"""
<div style="
    background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
    border-radius: 12px;
    padding: 14px 8px;
    height: 96px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: {text_color};
    box-shadow: 0 3px 8px rgba(0,0,0,0.18);
    text-align: center;
">
    <div style="font-size: 14px; font-weight: 800; letter-spacing: 0.3px;
                 line-height: 1.15;">
        {team}
    </div>
</div>
"""


def render_college_grid(*,
                          all_schools: list[str] | None = None,
                          on_pick_session_key: str = "college_school_v2",
                          title: str = "🎓  Pick your school") -> None:
    """Render the conference-grouped CFB grid.

    Clicking a team sets `st.session_state[on_pick_session_key]` to the
    team name and triggers a rerun, so the rest of College mode picks
    it up as the active school filter.

    `all_schools` — full school list from the data, used to populate
    the "Show all schools" expansion. If None, only the featured 24
    are clickable.
    """
    st.markdown(
        f"""
<div style="text-align: center; margin-bottom: 12px;">
    <div style="font-size: 28px; font-weight: 800; letter-spacing: -0.5px;">
        {title}
    </div>
    <div style="font-size: 14px; opacity: 0.7; margin-top: 4px;">
        Top 5 from each Power-4 conference + Notre Dame + non-P4 standouts.
        Click any school to drill into its roster.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Sort/group toggle — placed above the grid
    _, sort_col, _ = st.columns([1, 2, 1])
    with sort_col:
        sort_mode = st.radio(
            "Sort",
            options=["By conference", "A → Z (featured)"],
            horizontal=True,
            label_visibility="collapsed",
            key="cfb_grid_sort_mode",
        )

    if sort_mode == "A → Z (featured)":
        all_featured = sorted([t for ts in _FEATURED.values() for t in ts])
        n_per_row = 6
        for i in range(0, len(all_featured), n_per_row):
            row = all_featured[i:i + n_per_row]
            cols = st.columns(n_per_row)
            for col, team in zip(cols, row):
                with col:
                    st.markdown(_team_tile_html(team), unsafe_allow_html=True)
                    if st.button(f"Open {team}",
                                  key=f"cfb_grid_az_{team.replace(' ', '_')}",
                                  use_container_width=True):
                        _route_to_team(team, on_pick_session_key)
        # Still offer show-all below
        _render_show_all_section(all_schools, on_pick_session_key)
        return

    for conf, team_list in _FEATURED.items():
        st.markdown(
            f"""
<div style="font-size: 12px; font-weight: 800; letter-spacing: 2px;
             opacity: 0.6; margin: 14px 0 8px 4px;">
    {conf.upper()}
</div>
""",
            unsafe_allow_html=True,
        )
        cols = st.columns(5)
        for col, team in zip(cols, team_list):
            with col:
                st.markdown(_team_tile_html(team), unsafe_allow_html=True)
                if st.button(f"Open {team}",
                              key=f"cfb_grid_{team.replace(' ', '_')}",
                              use_container_width=True):
                    _route_to_team(team, on_pick_session_key)
        # Empty slots in shorter rows (e.g. Independents has 4 not 5)
        for empty_col in cols[len(team_list):]:
            with empty_col:
                st.markdown("&nbsp;", unsafe_allow_html=True)

    _render_show_all_section(all_schools, on_pick_session_key)


def _render_show_all_section(all_schools: list[str] | None,
                                on_pick_session_key: str) -> None:
    if not all_schools:
        return
    if "_cfb_grid_show_all" not in st.session_state:
        st.session_state._cfb_grid_show_all = False

    if not st.session_state._cfb_grid_show_all:
        st.markdown("---")
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            if st.button(f"📋  Show all {len(all_schools)} schools",
                          use_container_width=True,
                          key="cfb_grid_show_all_btn"):
                st.session_state._cfb_grid_show_all = True
                st.rerun()
    else:
        st.markdown("---")
        st.markdown(
            """
<div style="font-size: 12px; font-weight: 800; letter-spacing: 2px;
             opacity: 0.6; margin: 14px 0 8px 4px;">
    ALL SCHOOLS
</div>
""",
            unsafe_allow_html=True,
        )
        featured_set = {t for ts in _FEATURED.values() for t in ts}
        other = [s for s in all_schools if s not in featured_set]
        n_per_row = 6
        for i in range(0, len(other), n_per_row):
            row = other[i:i + n_per_row]
            cols = st.columns(n_per_row)
            for col, team in zip(cols, row):
                with col:
                    if st.button(team,
                                  key=f"cfb_grid_other_{team.replace(' ', '_')}",
                                  use_container_width=True):
                        _route_to_team(team, on_pick_session_key)
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            if st.button("⬆️  Collapse",
                          use_container_width=True,
                          key="cfb_grid_collapse_btn"):
                st.session_state._cfb_grid_show_all = False
                st.rerun()

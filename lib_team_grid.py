"""Stylized NFL team grid — the landing-page front door.

Renders all 32 teams organized by conference + division with team
colors and logos. Click any team to open its Team profile page.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lib_shared import team_theme

# AFC / NFC × 4 divisions × 4 teams
_DIVISIONS = {
    "AFC": {
        "EAST":  ["BUF", "MIA", "NE",  "NYJ"],
        "NORTH": ["BAL", "CIN", "CLE", "PIT"],
        "SOUTH": ["HOU", "IND", "JAX", "TEN"],
        "WEST":  ["DEN", "KC",  "LAC", "LV"],
    },
    "NFC": {
        "EAST":  ["DAL", "NYG", "PHI", "WAS"],
        "NORTH": ["CHI", "DET", "GB",  "MIN"],
        "SOUTH": ["ATL", "CAR", "NO",  "TB"],
        "WEST":  ["ARI", "LA",  "SEA", "SF"],
    },
}


def _readable_text(hex_color: str) -> str:
    """Return white or black text color for legibility against bg."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "white"
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.6 else "#0d1530"


def _team_tile_html(team: str, season: int) -> str:
    """One team tile — logo + colors. Clickable via Streamlit button below."""
    theme = team_theme(team)
    primary = theme.get("primary", "#1F2A44")
    secondary = theme.get("secondary", "#0B1730")
    logo = theme.get("logo", "")
    text_color = _readable_text(primary)
    return f"""
<div style="
    background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
    border-radius: 12px;
    padding: 14px 10px;
    height: 110px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: {text_color};
    box-shadow: 0 3px 8px rgba(0,0,0,0.18);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    cursor: pointer;
">
    {f'<img src="{logo}" style="height: 56px; width: 56px; object-fit: contain; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.3));"/>' if logo else ''}
    <div style="font-size: 13px; font-weight: 700; letter-spacing: 1px;
                 margin-top: 6px;">
        {team}
    </div>
</div>
"""


def render_team_grid(*, default_season: int = 2025,
                       title: str = "Pick your team") -> None:
    """Render the AFC | NFC stylized grid. Clicking a team opens
    pages/Team.py?abbr=X&season=Y."""
    st.markdown(
        f"""
<div style="text-align: center; margin-bottom: 16px;">
    <div style="font-size: 28px; font-weight: 800; letter-spacing: -0.5px;">
        {title}
    </div>
    <div style="font-size: 14px; opacity: 0.7; margin-top: 4px;">
        Each team page has historical comparables, story-card builders,
        and full roster click-through.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    conf_cols = st.columns(2)
    for i, (conf, divs) in enumerate(_DIVISIONS.items()):
        with conf_cols[i]:
            st.markdown(
                f"""
<div style="text-align: center; font-size: 13px; font-weight: 800;
             letter-spacing: 2px; opacity: 0.6; margin-bottom: 8px;">
    {conf} CONFERENCE
</div>
""",
                unsafe_allow_html=True,
            )
            for div_name, team_list in divs.items():
                st.markdown(
                    f"""
<div style="font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
             opacity: 0.55; margin: 6px 0 4px 4px;">
    {conf} {div_name}
</div>
""",
                    unsafe_allow_html=True,
                )
                cols = st.columns(4)
                for tcol, team in zip(cols, team_list):
                    with tcol:
                        st.markdown(_team_tile_html(team, default_season),
                                     unsafe_allow_html=True)
                        # Streamlit nav button — has to be a button (links
                        # in markdown can't trigger st.switch_page).
                        if st.button(
                            f"Open {team}",
                            key=f"grid_open_{team}_{conf}_{div_name}",
                            use_container_width=True,
                            help=f"View the {team} team profile",
                        ):
                            st.query_params.update({
                                "abbr": team,
                                "season": str(default_season),
                            })
                            st.switch_page("pages/Team.py")

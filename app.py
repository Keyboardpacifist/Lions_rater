"""
NFL Rater — Landing page
"""
import streamlit as st
from team_selector import get_team_and_season, NFL_TEAMS

st.set_page_config(
    page_title="NFL Rater",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

selected_team, selected_season = get_team_and_season()
team_name = NFL_TEAMS.get(selected_team, selected_team)

st.markdown(
    f"Pick a position from the sidebar to see how the **{selected_season} {team_name}** "
    f"stack up against every player in the league. You control what matters."
)

st.divider()

st.markdown("### Pick a position")
st.markdown(
    """
**Offense:** QB · WR · TE · RB · OL

**Defense:** DE · DT · LB · CB · S

**Special teams:** Kicker · Punter

**Front office:** Coaches · OC · DC · GM

Each position has slider-based ratings. Adjust what you value and
the rankings change in real time.
"""
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Our ethos")
    st.markdown(
        """
Every stat on every page has its formula, its data source, and its
known weaknesses on display. If something can't be measured honestly
from the data we have, we say so.
"""
    )

with col2:
    st.markdown("### Why this exists")
    st.markdown(
        """
Free data, open methodology, community-built. No grade is final,
no stat is beyond questioning.
"""
    )

st.divider()
st.caption("Built with Streamlit · Data from nflverse · Open source on GitHub")

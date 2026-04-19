"""
NFL Rater — Landing page
=========================
A transparent, customizable alternative to PFF.
Fans build and share their own rating methodologies.
"""

import streamlit as st
from team_selector import get_team_and_season, NFL_TEAMS

st.set_page_config(
    page_title="NFL Rater",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Team selector — first thing on every page
selected_team, selected_season = get_team_and_season()
team_name = NFL_TEAMS.get(selected_team, selected_team)

st.title(f"🏈 {team_name}")
st.subheader("A transparent, fan-built alternative to PFF")

st.markdown(
    f"""
Pick a position from the sidebar to see how the **{selected_season} {team_name}**
stack up against every player in the league. You control what matters —
adjust the sliders, and the rankings update instantly.

**Every score is yours.** You pick the stats. You pick the weights. You see
exactly how every number was computed. No black boxes, no proprietary grades.
"""
)

st.divider()

st.markdown("### Pick a position to rate")
st.markdown(
    """
Use the sidebar on the left to jump into any position:

**Offense:** QB · WR · TE · RB · OL

**Defense:** DE · DT · LB · CB · S

**Special teams:** Kicker · Punter

**Front office:** Coaches · OC · DC · GM

Each position has its own slider-based rater. Adjust what you value and
the rankings change in real time.
"""
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Our ethos")
    st.markdown(
        """
We strive for the greatest accuracy while being transparent about
limitations and how we addressed them. Every stat on every page has its
formula, its data source, and its known weaknesses on display. If something
can't be measured honestly from the data we have, we say so.
"""
    )

with col2:
    st.markdown("### Why this exists")
    st.markdown(
        """
Paid services like PFF do charting work the public can't easily replicate
and sell the results as proprietary grades. That's fine, but it means nobody
outside those companies can check their work. This is the opposite:
free data, open methodology, community-built. No grade is final, no stat is
beyond questioning.
"""
    )

st.divider()

st.caption("Built with Streamlit · Data from nflverse · Open source on GitHub")

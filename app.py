"""
Lions Rater — Landing page
===========================
A transparent, customizable alternative to PFF.
Fans build and share their own rating methodologies.

This file is just a landing page. All the actual rating pages live in
the `pages/` folder and are auto-discovered by Streamlit.

Ethos: strive for the greatest accuracy while being transparent about
limitations and how we addressed them.
"""

import streamlit as st

st.set_page_config(
    page_title="Lions Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🦁 Lions Rater")
st.subheader("A transparent, fan-built alternative to PFF")

st.markdown(
    """
The goal here is simple and a little ambitious: a **Wikipedia of athletic
performance.** A place where fans — and eventually players and coaches — build
their own rating methodologies, share them, argue about them, and watch the
arguments play out in the open.

Most public rating systems hand you a number and ask you to trust it. We do
the opposite. You pick the stats that matter to you. You pick how much each
one counts. You see exactly how every score was computed. If you disagree
with a methodology, you fork it and try your own.
"""
)

st.divider()

st.markdown("### Pick a position to rate")
st.markdown(
    """
Use the sidebar on the left to jump into a rating page:

- **Receivers** — rate Lions WRs and TEs
- **Running backs** — rate Lions RBs
- **(more positions coming)**

Each position has its own slider-based rater. Save your methodology and
share it with other fans. Browse what others have built. Fork and remix.
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
can't be measured honestly from the data we have, we say so — we don't make
it up.
"""
    )

with col2:
    st.markdown("### Why this exists")
    st.markdown(
        """
Paid services like PFF do charting work the public can't easily replicate
and sell the results as proprietary grades. That's fine, but it means nobody
outside those companies can check their work. Lions Rater is the opposite:
free data, open methodology, community-built. No grade is final, no stat is
beyond questioning.
"""
    )

st.divider()

st.caption("Built with Streamlit · Data from nflverse · Open source on GitHub")

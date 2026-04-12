"""
Lions Rater — landing page
==========================
Entry point for the multi-page app. Streamlit auto-discovers pages
from the pages/ folder and shows them in the sidebar nav.
"""

import streamlit as st

st.set_page_config(
    page_title="Lions Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
h1, h2, h3 { color: #0076B6 !important; }
.position-card {
    border: 2px solid #0076B6;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    background: #fafbfc;
}
.position-card h3 { margin-top: 0; }
</style>
""", unsafe_allow_html=True)

st.title("🦁 Lions Rater")

st.markdown("""
**Build your own algorithm.** Drag sliders to weight what you value, and
watch the Lions re-rank in real time. No "best player" — just **your**
best player.

👈 **Pick a position from the sidebar to start.**
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='position-card'>
    <h3>🎯 Receivers</h3>
    <p>Rate Lions WRs and TEs on reliability, explosive plays, field
    stretching, volume, and yards after catch. 11 underlying stats from
    nflverse and Next Gen Stats.</p>
    <p><em>Click "Receivers" in the sidebar →</em></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='position-card'>
    <h3>🏃 Running backs</h3>
    <p>Rate Lions RBs on efficiency, RB skill above scheme, explosive
    plays, workload, receiving, and short yardage. Includes RYOE from
    Next Gen Stats.</p>
    <p><em>Click "Running backs" in the sidebar →</em></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
### How it works

Every player gets a z-score on each stat, comparing them to all NFL players
at the same position. Z-scores are pulled toward the league average for
small samples (a guy with 12 carries doesn't get to be RB1 just because his
yards-per-carry is high). Your sliders weight which stats matter, and the
final score is just a weighted sum of those z-scores.

A score of **0** is exactly league average. **+1** is roughly top 16%.
**+2** is roughly top 2.5%.

### Roadmap

- **Stage 5 (now):** Running backs page with traditional and advanced stats
- **Stage 6:** Strength-of-schedule adjustment — separate raw production from production-given-the-defenses-faced
- **Stage 7:** Forward-looking matchup projections
""")

st.caption(
    "Data via [nflverse](https://github.com/nflverse) • "
    "Built as a fan project, not affiliated with the NFL or the Detroit Lions."
)

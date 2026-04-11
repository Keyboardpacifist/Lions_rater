"""
Lions Receiver Rater - Stage 3 prototype
=========================================
A Streamlit app that lets you build your own player rating algorithm
by dragging sliders, then watch Lions receivers re-rank in real time.

This is the v0 prototype: Lions WRs and TEs only, 2024 regular season,
shrunken z-scores against the league. No accounts, no saving yet -
those come in stage 4.
"""

import streamlit as st
import pandas as pd
import polars as pl
from pathlib import Path

# ============================================================
# Page config & styling
# ============================================================
st.set_page_config(
    page_title="Lions Receiver Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lions-themed CSS - Honolulu blue + silver, clean and bold
st.markdown("""
<style>
    /* Lions blue for headers and accents */
    h1, h2, h3 { color: #0076B6 !important; }

    /* Slider track in Lions blue */
    .stSlider [data-baseweb="slider"] > div > div > div > div {
        background-color: #0076B6;
    }

    /* Section dividers */
    .section-divider {
        border-top: 2px solid #B0B7BC;
        margin: 1.5rem 0 1rem 0;
    }

    /* Subtle player name styling */
    .player-rank {
        font-size: 0.9rem;
        color: #6c757d;
    }

    /* Give the dataframe a bit more breathing room */
    .stDataFrame { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_data():
    """Load the master Lions table built in stage 2."""
    data_path = Path(__file__).parent / "data" / "master_lions_with_z.parquet"
    df = pl.read_parquet(data_path).to_pandas()
    return df

# ============================================================
# Stat catalog - the user-facing list of stats with metadata
# ============================================================
# Each entry: (display_name, z_column, raw_column, description, category)
STAT_CATALOG = [
    # --- Efficiency stats ---
    ("Yards per target", "yards_per_target_z", "yards_per_target",
     "How many yards he produces every time the ball comes his way.", "Efficiency"),
    ("EPA per target", "epa_per_target_z", "epa_per_target",
     "Average expected points added on each target. The single best efficiency metric.", "Efficiency"),
    ("Success rate", "success_rate_z", "success_rate",
     "% of targets that produced a 'successful' play by EPA standards.", "Efficiency"),
    ("Catch rate", "catch_rate_z", "catch_rate",
     "% of targets that became receptions. Hands and reliability.", "Efficiency"),
    ("CPOE", "avg_cpoe_z", "avg_cpoe",
     "Completion % above expected. Catches the catches you're 'supposed' to.", "Efficiency"),

    # --- Production stats ---
    ("First down rate", "first_down_rate_z", "first_down_rate",
     "% of targets that picked up a first down. Chain-mover.", "Production"),
    ("YAC per reception", "yac_per_reception_z", "yac_per_reception",
     "Average yards after catch. Run-after-catch ability.", "Production"),
    ("YAC over expected", "yac_above_exp_z", "yac_above_exp",
     "YAC vs. what an average receiver would produce on the same catches.", "Production"),

    # --- Volume / opportunity stats ---
    ("Targets per snap", "targets_per_snap_z", "targets_per_snap",
     "How often he gets the ball when he's on the field. Trust from the QB.", "Volume"),
    ("Yards per snap", "yards_per_snap_z", "yards_per_snap",
     "Total yards produced per snap played. Combines efficiency and volume.", "Volume"),

    # --- Separation / route running ---
    ("Average separation", "avg_separation_z", "avg_separation",
     "Average yards of separation from nearest defender at the catch point.", "Route Running"),
]

CATEGORIES = ["Efficiency", "Production", "Volume", "Route Running"]
CATEGORY_COLORS = {
    "Efficiency": "#0076B6",     # Lions blue
    "Production": "#B0B7BC",     # Lions silver
    "Volume": "#003F5C",         # darker blue
    "Route Running": "#8B6F47",  # warm accent
}

# ============================================================
# Header
# ============================================================
st.title("🦁 Lions Receiver Rater")
st.markdown(
    "**Build your own algorithm.** Drag the sliders to weight what you value, "
    "and watch the Lions receivers re-rank in real time. "
    "_No 'best receiver' — just **your** best receiver._"
)

st.caption(
    "2024 regular season • Z-scores computed against all NFL WRs and TEs • "
    "Shrinkage applied to small samples"
)

# ============================================================
# Load data
# ============================================================
try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Couldn't find the data file. Make sure `data/master_lions_with_z.parquet` "
        "is committed to the repo."
    )
    st.stop()

# ============================================================
# Sidebar: filters & sliders
# ============================================================
st.sidebar.header("Filters")

# Position filter
positions = st.sidebar.multiselect(
    "Positions",
    options=["WR", "TE"],
    default=["WR", "TE"],
)

# Minimum snap threshold - keeps the small-sample noise out by default
min_snaps = st.sidebar.slider(
    "Minimum offensive snaps",
    min_value=0,
    max_value=1000,
    value=100,
    step=25,
    help="Players below this snap count won't appear in the ranking. "
         "Set to 0 to see everyone (with shrinkage applied).",
)

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.sidebar.header("Algorithm weights")
st.sidebar.caption(
    "How much should each stat matter? "
    "0 = ignore, 100 = fully weighted. Set everything to 0 to start fresh."
)

# Build sliders, grouped by category
weights = {}
for category in CATEGORIES:
    color = CATEGORY_COLORS[category]
    st.sidebar.markdown(
        f"**<span style='color:{color}'>{category}</span>**",
        unsafe_allow_html=True,
    )
    for display_name, z_col, raw_col, desc, cat in STAT_CATALOG:
        if cat != category:
            continue
        weights[z_col] = st.sidebar.slider(
            display_name,
            min_value=0,
            max_value=100,
            value=50,  # default: equal weight
            step=5,
            help=desc,
            key=f"slider_{z_col}",
        )

# ============================================================
# Compute weighted score
# ============================================================
# Filter the dataframe by position and snap threshold
filtered = df[df["position"].isin(positions)].copy()
filtered = filtered[filtered["off_snaps"].fillna(0) >= min_snaps]

if len(filtered) == 0:
    st.warning("No players match the current filters. Try lowering the snap threshold.")
    st.stop()

# Weighted sum of z-scores. Missing z-scores (e.g. NGS for low-snap players)
# are treated as 0 (= league average) so we don't penalize unfairly.
total_weight = sum(weights.values())
if total_weight == 0:
    filtered["score"] = 0.0
    st.info("All weights are zero — drag some sliders to start ranking.")
else:
    score = pd.Series(0.0, index=filtered.index)
    for z_col, w in weights.items():
        if w == 0 or z_col not in filtered.columns:
            continue
        score += filtered[z_col].fillna(0) * (w / total_weight)
    filtered["score"] = score

# Sort by score
filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)
filtered.index = filtered.index + 1  # 1-indexed rank

# ============================================================
# Main panel: ranked table
# ============================================================
st.subheader("Ranking")

# Build a clean display table
display_df = pd.DataFrame({
    "Rank": filtered.index,
    "Player": filtered["player_display_name"],
    "Pos": filtered["position"],
    "Snaps": filtered["off_snaps"].fillna(0).astype(int),
    "Targets": filtered["targets"].fillna(0).astype(int),
    "Yards": filtered["rec_yards"].fillna(0).astype(int),
    "TDs": filtered["rec_tds"].fillna(0).astype(int),
    "Score": filtered["score"].round(2),
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score": st.column_config.NumberColumn(
            "Your Score",
            help="Weighted sum of z-scores using your slider weights. "
                 "Higher = better. Roughly: 0 = league average, +1 = top ~16%, +2 = top ~2.5%.",
            format="%.2f",
        ),
    },
)

# ============================================================
# Detail panel: pick a player to see their full stat profile
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")

selected_player = st.selectbox(
    "Pick a player to see how the score breaks down",
    options=filtered["player_display_name"].tolist(),
    index=0,
)

player_row = filtered[filtered["player_display_name"] == selected_player].iloc[0]

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Position", player_row["position"])
    st.metric("Snaps", int(player_row["off_snaps"]) if pd.notna(player_row["off_snaps"]) else 0)
    st.metric("Targets", int(player_row["targets"]) if pd.notna(player_row["targets"]) else 0)
    st.metric("Receiving yards", int(player_row["rec_yards"]) if pd.notna(player_row["rec_yards"]) else 0)
    st.metric("Your score", f"{player_row['score']:.2f}")

with col2:
    st.markdown("**Stat-by-stat breakdown** (z-score vs league)")
    breakdown_rows = []
    for display_name, z_col, raw_col, desc, cat in STAT_CATALOG:
        z = player_row.get(z_col)
        raw = player_row.get(raw_col)
        weight = weights.get(z_col, 0)
        contribution = (z if pd.notna(z) else 0) * (weight / total_weight) if total_weight > 0 else 0
        breakdown_rows.append({
            "Stat": display_name,
            "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
            "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
            "Weight": f"{weight}",
            "Contribution": f"{contribution:+.2f}",
        })
    breakdown_df = pd.DataFrame(breakdown_rows)
    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

# ============================================================
# Footer
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    "Data via [nflverse](https://github.com/nflverse) • "
    "FTN charting via FTN Data via nflverse (CC-BY-SA 4.0) • "
    "Built as a fan project, not affiliated with the NFL or the Detroit Lions."
)

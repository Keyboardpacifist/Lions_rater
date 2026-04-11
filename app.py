"""
Lions Receiver Rater - Stage 3.1: Two-layer interface
======================================================
Casual fans get plain-English bundle sliders ("Reliability",
"Explosive plays", etc.). Power users can flip on Advanced mode
to see and tweak the underlying individual stats.

Same math as before under the hood: weighted sum of shrunken
z-scores against league-wide WR/TE peers. Bundles are just
named groups of underlying stats with internal weights.
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

st.markdown("""
<style>
    h1, h2, h3 { color: #0076B6 !important; }
    .stSlider [data-baseweb="slider"] > div > div > div > div {
        background-color: #0076B6;
    }
    .section-divider {
        border-top: 2px solid #B0B7BC;
        margin: 1.5rem 0 1rem 0;
    }
    .bundle-desc {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: -0.5rem;
        margin-bottom: 0.5rem;
    }
    .stDataFrame { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_data():
    data_path = Path(__file__).parent / "data" / "master_lions_with_z.parquet"
    df = pl.read_parquet(data_path).to_pandas()
    return df

# ============================================================
# Stat catalog - underlying individual stats
# ============================================================
INDIVIDUAL_STATS = {
    "yards_per_target_z": ("Yards per target", "yards_per_target",
        "How many yards he produces every time the ball comes his way."),
    "epa_per_target_z": ("EPA per target", "epa_per_target",
        "Expected points added per target. The single best efficiency metric."),
    "success_rate_z": ("Success rate", "success_rate",
        "% of targets that produced a 'successful' play by EPA standards."),
    "catch_rate_z": ("Catch rate", "catch_rate",
        "% of targets that became receptions. Hands and reliability."),
    "avg_cpoe_z": ("CPOE", "avg_cpoe",
        "Completion % above expected. Catches the catches you're supposed to."),
    "first_down_rate_z": ("First down rate", "first_down_rate",
        "% of targets that picked up a first down. Chain-mover."),
    "yac_per_reception_z": ("YAC per reception", "yac_per_reception",
        "Average yards after catch. Run-after-catch ability."),
    "yac_above_exp_z": ("YAC over expected", "yac_above_exp",
        "YAC vs. what an average receiver would produce on the same catches."),
    "targets_per_snap_z": ("Targets per snap", "targets_per_snap",
        "How often he gets the ball when on the field. QB trust."),
    "yards_per_snap_z": ("Yards per snap", "yards_per_snap",
        "Total yards produced per snap. Combines efficiency and volume."),
    "avg_separation_z": ("Average separation", "avg_separation",
        "Average yards of separation from nearest defender at the catch point."),
}

# ============================================================
# Bundles - the casual-fan-friendly layer
# ============================================================
BUNDLES = {
    "reliability": {
        "label": "🎯 Reliability",
        "description": "Catches what's thrown his way and keeps drives alive.",
        "stats": {
            "catch_rate_z": 0.30,
            "avg_cpoe_z": 0.20,
            "success_rate_z": 0.30,
            "first_down_rate_z": 0.20,
        },
    },
    "explosive": {
        "label": "💥 Explosive plays",
        "description": "Turns targets into chunk plays. Big gains, not just dump-offs.",
        "stats": {
            "yards_per_target_z": 0.50,
            "yac_above_exp_z": 0.30,
            "yards_per_snap_z": 0.20,
        },
    },
    "deep_threat": {
        "label": "🔥 Field stretcher",
        "description": "Takes the top off the defense. The 'go deep' guy.",
        "stats": {
            "yards_per_target_z": 0.40,
            "avg_separation_z": 0.30,
            "yards_per_snap_z": 0.30,
        },
    },
    "volume": {
        "label": "📊 Volume & usage",
        "description": "How much of the offense runs through him.",
        "stats": {
            "targets_per_snap_z": 0.50,
            "yards_per_snap_z": 0.50,
        },
    },
    "after_catch": {
        "label": "🏃 After the catch",
        "description": "What happens once he's got the ball in his hands.",
        "stats": {
            "yac_per_reception_z": 0.50,
            "yac_above_exp_z": 0.50,
        },
    },
}

DEFAULT_BUNDLE_WEIGHTS = {
    "reliability": 60,
    "explosive": 50,
    "deep_threat": 30,
    "volume": 60,
    "after_catch": 30,
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
    "2024 regular season • Compared against all NFL WRs and TEs • "
    "Small samples adjusted toward league average"
)

# ============================================================
# Load data
# ============================================================
try:
    df = load_data()
except FileNotFoundError:
    st.error("Couldn't find the data file.")
    st.stop()

# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Filters")

positions = st.sidebar.multiselect(
    "Positions",
    options=["WR", "TE"],
    default=["WR", "TE"],
)

min_snaps = st.sidebar.slider(
    "Minimum offensive snaps",
    min_value=0,
    max_value=1000,
    value=100,
    step=25,
    help="Hide players who barely played. Set to 0 to see everyone.",
)

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

advanced_mode = st.sidebar.toggle(
    "🔬 Advanced mode",
    value=False,
    help="Show individual stat sliders (EPA, CPOE, separation, etc.) "
         "instead of plain-English bundles.",
)

st.sidebar.header("What do you value?")

# ------------------------------------------------------------
# Compute the effective per-stat weights based on current mode
# ------------------------------------------------------------
effective_weights = {}
bundle_weights = {}

if not advanced_mode:
    st.sidebar.caption("Drag to weight what matters to you. 0 = ignore, 100 = max.")
    for bundle_key, bundle in BUNDLES.items():
        st.sidebar.markdown(f"**{bundle['label']}**")
        st.sidebar.markdown(
            f"<div class='bundle-desc'>{bundle['description']}</div>",
            unsafe_allow_html=True,
        )
        bundle_weights[bundle_key] = st.sidebar.slider(
            bundle["label"],
            min_value=0,
            max_value=100,
            value=DEFAULT_BUNDLE_WEIGHTS[bundle_key],
            step=5,
            key=f"bundle_{bundle_key}",
            label_visibility="collapsed",
        )

    for bundle_key, bundle_weight in bundle_weights.items():
        if bundle_weight == 0:
            continue
        for z_col, internal_weight in BUNDLES[bundle_key]["stats"].items():
            effective_weights[z_col] = effective_weights.get(z_col, 0) + bundle_weight * internal_weight

else:
    st.sidebar.caption("Direct control over every underlying stat.")
    for z_col, (display_name, raw_col, desc) in INDIVIDUAL_STATS.items():
        effective_weights[z_col] = st.sidebar.slider(
            display_name,
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help=desc,
            key=f"adv_{z_col}",
        )

# ============================================================
# Filter players & compute weighted score
# ============================================================
filtered = df[df["position"].isin(positions)].copy()
filtered = filtered[filtered["off_snaps"].fillna(0) >= min_snaps]

if len(filtered) == 0:
    st.warning("No players match the current filters. Try lowering the snap threshold.")
    st.stop()

total_weight = sum(effective_weights.values())
if total_weight == 0:
    filtered["score"] = 0.0
    st.info("All weights are zero — drag some sliders to start ranking.")
else:
    score = pd.Series(0.0, index=filtered.index)
    for z_col, w in effective_weights.items():
        if w == 0 or z_col not in filtered.columns:
            continue
        score += filtered[z_col].fillna(0) * (w / total_weight)
    filtered["score"] = score

filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)
filtered.index = filtered.index + 1

# ============================================================
# Main panel: ranked table
# ============================================================
st.subheader("Ranking")

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
            help="Higher = better. Roughly: 0 = league average, "
                 "+1 = top ~16%, +2 = top ~2.5%.",
            format="%.2f",
        ),
    },
)

# ============================================================
# Detail panel
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")

selected_player = st.selectbox(
    "Pick a player to see how their score breaks down",
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
    if not advanced_mode:
        st.markdown("**How your score breaks down**")
        bundle_rows = []
        for bundle_key, bundle in BUNDLES.items():
            bundle_weight = bundle_weights.get(bundle_key, 0)
            if bundle_weight == 0:
                continue
            contribution = 0
            for z_col, internal_weight in bundle["stats"].items():
                z = player_row.get(z_col)
                if pd.notna(z):
                    eff_weight = bundle_weight * internal_weight
                    contribution += z * (eff_weight / total_weight)
            bundle_rows.append({
                "Bundle": bundle["label"],
                "Your weight": f"{bundle_weight}",
                "Contribution": f"{contribution:+.2f}",
            })
        if bundle_rows:
            st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No bundles weighted — drag some sliders.")

        with st.expander("🔬 See the underlying stats"):
            st.caption(
                "Each bundle is a mix of these individual stats. "
                "Z-scores show how each player compares to the league "
                "(0 = average, +1 = top ~16%, +2 = top ~2.5%)."
            )
            stat_rows = []
            for z_col, (display_name, raw_col, desc) in INDIVIDUAL_STATS.items():
                z = player_row.get(z_col)
                raw = player_row.get(raw_col)
                stat_rows.append({
                    "Stat": display_name,
                    "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
                    "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                })
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("**Stat-by-stat breakdown** (z-score vs league)")
        breakdown_rows = []
        for z_col, (display_name, raw_col, desc) in INDIVIDUAL_STATS.items():
            z = player_row.get(z_col)
            raw = player_row.get(raw_col)
            weight = effective_weights.get(z_col, 0)
            contribution = (z if pd.notna(z) else 0) * (weight / total_weight) if total_weight > 0 else 0
            breakdown_rows.append({
                "Stat": display_name,
                "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
                "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                "Weight": f"{weight}",
                "Contribution": f"{contribution:+.2f}",
            })
        st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

# ============================================================
# Footer
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    "Data via [nflverse](https://github.com/nflverse) • "
    "FTN charting via FTN Data via nflverse (CC-BY-SA 4.0) • "
    "Built as a fan project, not affiliated with the NFL or the Detroit Lions."
)

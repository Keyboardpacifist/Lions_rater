"""
Lions Receiver Rater — Receivers page
=====================================
Two-layer slider UI for WR/TE rankings, with save/load/browse community
algorithms scoped to position_group='receiver'.
"""

from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from lib_shared import (
    apply_algo_weights,
    community_section,
    compute_effective_weights,
    get_algorithm_by_slug,
    inject_css,
    score_players,
)

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Lions Receiver Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

POSITION_GROUP = "receiver"
PAGE_URL = "https://lions-rater.streamlit.app/Receivers"

# ============================================================
# Data
# ============================================================
@st.cache_data
def load_data():
    data_path = (
        Path(__file__).resolve().parent.parent / "data" / "master_lions_with_z.parquet"
    )
    return pl.read_parquet(data_path).to_pandas()


# ============================================================
# Stat catalog
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
# Session state
# ============================================================
if "loaded_algo" not in st.session_state:
    st.session_state.loaded_algo = None
if "upvoted_ids" not in st.session_state:
    st.session_state.upvoted_ids = set()

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
    st.error("Couldn't find the receivers data file.")
    st.stop()

# ============================================================
# ?algo= deep link
# ============================================================
if "algo" in st.query_params and st.session_state.loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group", "receiver") == POSITION_GROUP:
        apply_algo_weights(linked, BUNDLES)
        st.rerun()

# ============================================================
# Sidebar — filters
# ============================================================
st.sidebar.header("Filters")
positions = st.sidebar.multiselect(
    "Positions", options=["WR", "TE"], default=["WR", "TE"]
)
min_snaps = st.sidebar.slider(
    "Minimum offensive snaps", 0, 1000, 100, step=25,
    help="Hide players who barely played. Set to 0 to see everyone.",
)

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle(
    "🔬 Advanced mode", value=False,
    help="Show individual stat sliders instead of plain-English bundles.",
)

st.sidebar.header("What do you value?")

if st.session_state.loaded_algo:
    la = st.session_state.loaded_algo
    st.sidebar.info(
        f"Loaded: **{la['name']}** by {la['author']}\n\n"
        f"_{la.get('description', '')}_"
    )
    if st.sidebar.button("Clear loaded algorithm"):
        st.session_state.loaded_algo = None

# ============================================================
# Sliders
# ============================================================
bundle_weights = {}
effective_weights = {}

if not advanced_mode:
    st.sidebar.caption("Drag to weight what matters to you. 0 = ignore, 100 = max.")
    for bk, bundle in BUNDLES.items():
        st.sidebar.markdown(f"**{bundle['label']}**")
        st.sidebar.markdown(
            f"<div class='bundle-desc'>{bundle['description']}</div>",
            unsafe_allow_html=True,
        )
        bundle_weights[bk] = st.sidebar.slider(
            bundle["label"], 0, 100,
            value=DEFAULT_BUNDLE_WEIGHTS.get(bk, 50),
            step=5,
            key=f"bundle_{bk}",
            label_visibility="collapsed",
        )
    effective_weights = compute_effective_weights(BUNDLES, bundle_weights)
else:
    st.sidebar.caption("Direct control over every underlying stat.")
    for z_col, (display_name, raw_col, desc) in INDIVIDUAL_STATS.items():
        effective_weights[z_col] = st.sidebar.slider(
            display_name, 0, 100, 50, step=5,
            help=desc, key=f"adv_rec_{z_col}",
        )

# ============================================================
# Filter & score
# ============================================================
filtered = df[df["position"].isin(positions)].copy()
filtered = filtered[filtered["off_snaps"].fillna(0) >= min_snaps]

if len(filtered) == 0:
    st.warning("No players match the current filters. Try lowering the snap threshold.")
    st.stop()

filtered = score_players(filtered, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0:
    st.info("All weights are zero — drag some sliders to start ranking.")

filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)
filtered.index = filtered.index + 1

# ============================================================
# Ranking table
# ============================================================
st.subheader("Ranking")
st.caption(
    "⚠️ Players with very few targets have noisy scores — extreme values "
    "reflect small sample sizes, not skill. Use the 'Minimum offensive snaps' "
    "filter in the sidebar to hide low-volume players if desired."
)
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
# Player detail
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")

selected = st.selectbox(
    "Pick a player to see how their score breaks down",
    options=filtered["player_display_name"].tolist(),
    index=0,
)
player = filtered[filtered["player_display_name"] == selected].iloc[0]

c1, c2 = st.columns([1, 2])

with c1:
    st.metric("Position", player["position"])
    st.metric("Snaps", int(player["off_snaps"]) if pd.notna(player["off_snaps"]) else 0)
    st.metric("Targets", int(player["targets"]) if pd.notna(player["targets"]) else 0)
    st.metric("Receiving yards", int(player["rec_yards"]) if pd.notna(player["rec_yards"]) else 0)
    st.metric("Your score", f"{player['score']:.2f}")

with c2:
    if not advanced_mode:
        st.markdown("**How your score breaks down**")
        bundle_rows = []
        for bk, bundle in BUNDLES.items():
            bw = bundle_weights.get(bk, 0)
            if bw == 0:
                continue
            contribution = 0
            for z_col, internal in bundle["stats"].items():
                z = player.get(z_col)
                if pd.notna(z) and total_weight > 0:
                    contribution += z * (bw * internal / total_weight)
            bundle_rows.append({
                "Bundle": bundle["label"],
                "Your weight": f"{bw}",
                "Contribution": f"{contribution:+.2f}",
            })
        if bundle_rows:
            st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No bundles weighted — drag some sliders.")

        with st.expander("🔬 See the underlying stats"):
            stat_rows = []
            for z_col, (display_name, raw_col, desc) in INDIVIDUAL_STATS.items():
                z = player.get(z_col)
                raw = player.get(raw_col)
                stat_rows.append({
                    "Stat": display_name,
                    "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
                    "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                })
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("**Stat-by-stat breakdown** (z-score vs league)")
        rows = []
        for z_col, (display_name, raw_col, desc) in INDIVIDUAL_STATS.items():
            z = player.get(z_col)
            raw = player.get(raw_col)
            w = effective_weights.get(z_col, 0)
            contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            rows.append({
                "Stat": display_name,
                "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
                "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                "Weight": f"{w}",
                "Contribution": f"{contrib:+.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================
# Community algorithms
# ============================================================
community_section(
    position_group=POSITION_GROUP,
    bundles=BUNDLES,
    bundle_weights=bundle_weights,
    advanced_mode=advanced_mode,
    page_url=PAGE_URL,
)

# ============================================================
# Footer
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    "Data via [nflverse](https://github.com/nflverse) • "
    "FTN charting via FTN Data via nflverse (CC-BY-SA 4.0) • "
    "Built as a fan project, not affiliated with the NFL or the Detroit Lions."
)

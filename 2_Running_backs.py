"""
Lions Running Back Rater
========================
Two-layer slider UI for RB rankings, with save/load/browse community
algorithms scoped to position_group='rb'.
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
    page_title="Lions Running Back Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

POSITION_GROUP = "rb"
PAGE_URL = "https://lions-rater.streamlit.app/Running_backs"

# ============================================================
# Data
# ============================================================
@st.cache_data
def load_data():
    data_path = (
        Path(__file__).resolve().parent.parent
        / "data" / "master_lions_rbs_with_z.parquet"
    )
    return pl.read_parquet(data_path).to_pandas()


# ============================================================
# Stat catalog (RB-specific)
# ============================================================
INDIVIDUAL_STATS = {
    "yards_per_carry_z": ("Yards per carry", "yards_per_carry",
        "Plain old yards per carry. Volume-adjusted across the league."),
    "epa_per_rush_z": ("EPA per rush", "epa_per_rush",
        "Expected points added per rush attempt. Best single efficiency stat."),
    "rush_success_rate_z": ("Rush success rate", "rush_success_rate",
        "% of carries that produced a 'successful' play by EPA standards."),
    "ryoe_per_att_z": ("RYOE per attempt", "ryoe_per_att",
        "Rush yards over expected. NGS measures what an average back would "
        "have gained on the same carry, given blocking and box count."),
    "broken_tackles_per_att_z": ("Broken tackles per att", "broken_tackles_per_att",
        "How often he makes the first defender miss. FTN charting."),
    "yards_before_contact_per_att_z": ("Yards before contact", "yards_before_contact_per_att",
        "How much room the offensive line gives him before he gets touched."),
    "yards_after_contact_per_att_z": ("Yards after contact", "yards_after_contact_per_att",
        "How much he gains AFTER first contact. Pure RB ability."),
    "explosive_run_rate_z": ("Explosive run rate", "explosive_run_rate",
        "% of carries that go for 10+ yards."),
    "explosive_15_rate_z": ("15+ yard run rate", "explosive_15_rate",
        "% of carries that go for 15+ yards. The big-play threshold."),
    "carries_per_game_z": ("Carries per game", "carries_per_game",
        "Workhorse score. How much he carries the rock."),
    "snap_share_z": ("Snap share", "snap_share",
        "% of offensive snaps he was on the field."),
    "touches_per_game_z": ("Touches per game", "touches_per_game",
        "Carries plus receptions. Total involvement in the offense."),
    "rz_carry_share_z": ("Red zone carry share", "rz_carry_share",
        "% of his carries that came inside the 20. Trusted near the goal line."),
    "rec_yards_per_target_z": ("Rec yards per target", "rec_yards_per_target",
        "Receiving production efficiency for backs."),
    "yac_per_reception_z": ("YAC per reception", "yac_per_reception",
        "Run-after-catch ability on outlet passes and screens."),
    "targets_per_game_z": ("Targets per game", "targets_per_game",
        "How often the offense looks his way as a receiver."),
    "rec_epa_per_target_z": ("Receiving EPA per target", "rec_epa_per_target",
        "Expected points added per target. Receiving efficiency."),
    "short_yardage_conv_rate_z": ("Short yardage conv rate", "short_yardage_conv_rate",
        "% of 3rd/4th & 1-2 carries he converted into a first down."),
    "goal_line_td_rate_z": ("Goal line TD rate", "goal_line_td_rate",
        "% of his inside-the-5 carries that scored."),
}

# 6 bundles
BUNDLES = {
    "efficiency": {
        "label": "⚡ Efficiency",
        "description": "Productive on a per-carry basis. Doesn't waste touches.",
        "stats": {
            "yards_per_carry_z": 0.25,
            "epa_per_rush_z": 0.35,
            "rush_success_rate_z": 0.20,
            "ryoe_per_att_z": 0.20,
        },
    },
    "tackle_breaking": {
        "label": "💪 Tackle breaking",
        "description": "Makes defenders miss and grinds out yards after contact.",
        "stats": {
            "broken_tackles_per_att_z": 0.40,
            "yards_after_contact_per_att_z": 0.45,
            "yards_before_contact_per_att_z": 0.15,
        },
    },
    "explosive": {
        "label": "💥 Explosive plays",
        "description": "Hits the home run. Big-play threat every carry.",
        "stats": {
            "explosive_run_rate_z": 0.50,
            "explosive_15_rate_z": 0.50,
        },
    },
    "volume": {
        "label": "📊 Volume & usage",
        "description": "Workhorse. The offense runs through him.",
        "stats": {
            "carries_per_game_z": 0.35,
            "snap_share_z": 0.30,
            "touches_per_game_z": 0.35,
        },
    },
    "receiving": {
        "label": "🤲 Receiving back",
        "description": "Dual threat out of the backfield as a pass catcher.",
        "stats": {
            "rec_yards_per_target_z": 0.25,
            "yac_per_reception_z": 0.20,
            "targets_per_game_z": 0.30,
            "rec_epa_per_target_z": 0.25,
        },
    },
    "short_yardage": {
        "label": "🎯 Short yardage & goal line",
        "description": "Gets the tough yards when the team needs them most.",
        "stats": {
            "short_yardage_conv_rate_z": 0.50,
            "goal_line_td_rate_z": 0.30,
            "rz_carry_share_z": 0.20,
        },
    },
}

DEFAULT_BUNDLE_WEIGHTS = {
    "efficiency": 70,
    "tackle_breaking": 50,
    "explosive": 40,
    "volume": 60,
    "receiving": 30,
    "short_yardage": 30,
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
st.title("🦁 Lions Running Back Rater")
st.markdown(
    "**Build your own algorithm.** Drag the sliders to weight what you "
    "value, and watch the Lions running backs re-rank in real time. "
    "_No 'best back' — just **your** best back._"
)
st.caption(
    "2024 regular season • Compared against all NFL RBs • "
    "Small samples adjusted toward league average"
)

# ============================================================
# Load data
# ============================================================
try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Couldn't find the running backs data file. "
        "Run `python scripts/build_data.py` to build it."
    )
    st.stop()

# ============================================================
# ?algo= deep link
# ============================================================
if "algo" in st.query_params and st.session_state.loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP:
        apply_algo_weights(linked, BUNDLES)
        st.rerun()

# ============================================================
# Sidebar — filters
# ============================================================
st.sidebar.header("Filters")
min_carries = st.sidebar.slider(
    "Minimum carries", 0, 300, 20, step=5,
    help="Hide backs who barely touched the ball.",
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
            help=desc, key=f"adv_rb_{z_col}",
        )

# ============================================================
# Filter & score
# ============================================================
filtered = df[df["carries"].fillna(0) >= min_carries].copy()

if len(filtered) == 0:
    st.warning("No backs match the current filters. Try lowering the carry threshold.")
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
display_df = pd.DataFrame({
    "Rank": filtered.index,
    "Player": filtered["player_display_name"],
    "Carries": filtered["carries"].fillna(0).astype(int),
    "Rush yds": filtered["rush_yards"].fillna(0).astype(int),
    "Rush TDs": filtered["rush_tds"].fillna(0).astype(int),
    "Rec": filtered["receptions"].fillna(0).astype(int),
    "Rec yds": filtered["rec_yards"].fillna(0).astype(int),
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
    "Pick a back to see how their score breaks down",
    options=filtered["player_display_name"].tolist(),
    index=0,
)
player = filtered[filtered["player_display_name"] == selected].iloc[0]

c1, c2 = st.columns([1, 2])

with c1:
    st.metric("Carries", int(player["carries"]) if pd.notna(player["carries"]) else 0)
    st.metric("Rush yards", int(player["rush_yards"]) if pd.notna(player["rush_yards"]) else 0)
    st.metric("Rush TDs", int(player["rush_tds"]) if pd.notna(player["rush_tds"]) else 0)
    st.metric("Receptions", int(player["receptions"]) if pd.notna(player["receptions"]) else 0)
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

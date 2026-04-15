"""
Lions Running Back Rater — RB page (tier migration)
===================================================
Tier-based slider UI for RB rankings, matching the Receivers and OL pages.

What the tier system does:
- Loads stat tiers and methodology from data/rb_stat_metadata.json.
- Tier checkboxes at the top of the page let users filter which stats
  participate in scoring. Tier 4 off by default; there are no Tier 4
  stats for RBs, so that checkbox is a no-op for now but we keep
  it for consistency with WR and OL.
- When a tier is disabled, any stat in that tier is removed from every
  bundle. Bundles that end up empty disappear from the sidebar.
- Advanced mode shows per-stat sliders with methodology in help tooltips.
- Leaderboard scores carry a label like "+0.47 (above group)".
- "How is this score calculated?" expander below the leaderboard.

Design note: Tier 1 raw counts (rush_yards_z, rush_tds_z, carries_z,
receptions_z, rec_yards_z, rec_tds_z) are NOT added to any existing
bundle. Adding raw volume to "Efficiency" or "Tackle breaking" would
break the bundle's meaning. Tier 1 stats are still accessible — they
show up in Advanced mode. Bundle mode keeps its clean original design.
"""

import json
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

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "master_lions_rbs_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "rb_stat_metadata.json"


# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_data():
    return pl.read_parquet(DATA_PATH).to_pandas()


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH) as f:
        return json.load(f)


# ============================================================
# Stat catalog — raw column names for Player Detail display
# ============================================================
RAW_COL_MAP = {
    "rush_yards_z": "rush_yards",
    "rush_tds_z": "rush_tds",
    "carries_z": "carries",
    "receptions_z": "receptions",
    "rec_yards_z": "rec_yards",
    "rec_tds_z": "rec_tds",
    "yards_per_carry_z": "yards_per_carry",
    "rush_success_rate_z": "rush_success_rate",
    "carries_per_game_z": "carries_per_game",
    "snap_share_z": "snap_share",
    "touches_per_game_z": "touches_per_game",
    "targets_per_game_z": "targets_per_game",
    "explosive_run_rate_z": "explosive_run_rate",
    "explosive_15_rate_z": "explosive_15_rate",
    "rz_carry_share_z": "rz_carry_share",
    "goal_line_td_rate_z": "goal_line_td_rate",
    "short_yardage_conv_rate_z": "short_yardage_conv_rate",
    "rec_yards_per_target_z": "rec_yards_per_target",
    "yac_per_reception_z": "yac_per_reception",
    "broken_tackles_per_att_z": "broken_tackles_per_att",
    "yards_before_contact_per_att_z": "yards_before_contact_per_att",
    "yards_after_contact_per_att_z": "yards_after_contact_per_att",
    "epa_per_rush_z": "epa_per_rush",
    "rec_epa_per_target_z": "rec_epa_per_target",
    "ryoe_per_att_z": "ryoe_per_att",
}


# ============================================================
# Bundles — Tier 2/3 organized, unchanged from previous version
# ============================================================
# Tier 1 raw counts are deliberately excluded from bundles (see module
# docstring). They're still accessible via Advanced mode.
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
# Tier helpers
# ============================================================
TIER_LABELS = {
    1: "Tier 1 — Counted",
    2: "Tier 2 — Contextualized",
    3: "Tier 3 — Adjusted",
    4: "Tier 4 — Inferred",
}
TIER_DESCRIPTIONS = {
    1: "Pure recorded facts. No modeling.",
    2: "Counts divided by opportunity. Still no modeling.",
    3: "Compared against a modeled baseline. Model is simple and visible.",
    4: "Inferred from patterns the data can't directly see. Use with skepticism.",
}


def tier_badge(tier: int) -> str:
    return {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠"}.get(tier, "⚪")


def filter_bundles_by_tier(bundles: dict, stat_tiers: dict, enabled_tiers: list) -> dict:
    """Strip disabled-tier stats out of each bundle. Empty bundles drop out."""
    filtered = {}
    for bk, bdef in bundles.items():
        kept_stats = {
            z: w for z, w in bdef["stats"].items()
            if stat_tiers.get(z, 2) in enabled_tiers
        }
        if kept_stats:
            filtered[bk] = {
                "label": bdef["label"],
                "description": bdef["description"],
                "stats": kept_stats,
            }
    return filtered


def bundle_tier_summary(bundle_stats: dict, stat_tiers: dict) -> str:
    counts = {}
    for z in bundle_stats:
        t = stat_tiers.get(z, 2)
        counts[t] = counts.get(t, 0) + 1
    return " ".join(f"{tier_badge(t)}×{c}" for t, c in sorted(counts.items()))


# ============================================================
# Score labels
# ============================================================
def score_label(score: float) -> str:
    if pd.isna(score):
        return "—"
    if score >= 1.0:
        return "well above group"
    if score >= 0.4:
        return "above group"
    if score >= -0.4:
        return "about average"
    if score >= -1.0:
        return "below group"
    return "well below group"


def format_score(score: float) -> str:
    if pd.isna(score):
        return "—"
    sign = "+" if score >= 0 else ""
    return f"{sign}{score:.2f} ({score_label(score)})"


SCORE_EXPLAINER = """
**What this number means.** The score is a weighted average of z-scores —
standardized stats where 0 is the league average, +1 is one standard
deviation above, and −1 is one standard deviation below. Your slider
weights control how much each bundle contributes.

**How to read it:**
- `+1.0` or higher → well above the league average on what you weighted
- `+0.4` to `+1.0` → above average
- `−0.4` to `+0.4` → roughly average
- `−1.0` or lower → well below average

**What this is not.** It's not a PFF-style 0-100 grade. It's a
**comparative** number telling you how Lions running backs stack up
against the top 32 RBs in the league, under the methodology *you* chose.

**League population:** z-scores are computed against the top 32 RBs by
offensive snaps (min 6 games played). Every Lions RB with at least one
offensive snap is visible, but players with very few carries will have
noisy scores — read extreme values on low-volume players as "small
sample, not skill."
"""


# ============================================================
# Session state
# ============================================================
if "loaded_algo" not in st.session_state:
    st.session_state.loaded_algo = None
if "upvoted_ids" not in st.session_state:
    st.session_state.upvoted_ids = set()
if "rb_tiers_enabled" not in st.session_state:
    st.session_state.rb_tiers_enabled = [1, 2, 3]  # Tier 4 off by default


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
    "2024 regular season • Compared against top 32 RBs by snaps • "
    "Every Lions RB visible"
)


# ============================================================
# Load data
# ============================================================
try:
    df = load_data()
except FileNotFoundError:
    st.error("Couldn't find the running backs data file.")
    st.stop()

meta = load_metadata()
stat_tiers = meta.get("stat_tiers", {})
stat_labels = meta.get("stat_labels", {})
stat_methodology = meta.get("stat_methodology", {})


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
    help="Show individual stat sliders with methodology tooltips instead of plain-English bundles.",
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
# Tier filter (main content area)
# ============================================================
st.markdown("### How speculative do you want to get?")
st.caption(
    "Each stat is labeled by how much trust it asks from you. "
    "Uncheck tiers you don't want to include. Philosophy in a checkbox."
)
tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        checked = st.checkbox(
            f"{tier_badge(tier)} {TIER_LABELS[tier]}",
            value=(tier in st.session_state.rb_tiers_enabled),
            help=TIER_DESCRIPTIONS[tier],
            key=f"rb_tier_checkbox_{tier}",
        )
        if checked:
            new_enabled.append(tier)
st.session_state.rb_tiers_enabled = new_enabled

if not new_enabled:
    st.warning("Enable at least one tier to see ratings.")
    st.stop()

active_bundles = filter_bundles_by_tier(BUNDLES, stat_tiers, new_enabled)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ============================================================
# Sliders
# ============================================================
bundle_weights = {}
effective_weights = {}

if not advanced_mode:
    if not active_bundles:
        st.info("No bundles have stats in the enabled tiers. Try enabling more tiers.")
        st.stop()

    st.sidebar.caption("Drag to weight what matters to you. 0 = ignore, 100 = max.")
    for bk, bundle in active_bundles.items():
        tier_summary = bundle_tier_summary(bundle["stats"], stat_tiers)
        st.sidebar.markdown(f"**{bundle['label']}**")
        st.sidebar.markdown(
            f"<div class='bundle-desc'>{bundle['description']}<br>"
            f"<small>{tier_summary}</small></div>",
            unsafe_allow_html=True,
        )
        if f"bundle_{bk}" not in st.session_state:
            st.session_state[f"bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(
            bundle["label"], 0, 100,
            step=5,
            key=f"bundle_{bk}",
            label_visibility="collapsed",
        )
    # Bundles not in active_bundles still need a zero entry for save
    for bk in BUNDLES:
        if bk not in bundle_weights:
            bundle_weights[bk] = 0
    effective_weights = compute_effective_weights(active_bundles, bundle_weights)
else:
    st.sidebar.caption(
        "Direct control over every underlying stat. Hover the ⓘ icon next to "
        "each slider for methodology."
    )
    # Build list of all stats in enabled tiers, sorted by tier then by label
    all_enabled_stats = [
        z for z, t in stat_tiers.items() if t in new_enabled
    ]
    all_enabled_stats.sort(key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))

    for z_col in all_enabled_stats:
        tier = stat_tiers.get(z_col, 2)
        label = stat_labels.get(z_col, z_col)
        meth = stat_methodology.get(z_col, {})

        # Build a rich help tooltip with What/How/Limits
        help_parts = []
        if meth.get("what"):
            help_parts.append(f"What: {meth['what']}")
        if meth.get("how"):
            help_parts.append(f"How: {meth['how']}")
        if meth.get("limits"):
            help_parts.append(f"Limits: {meth['limits']}")
        help_text = "\n\n".join(help_parts) if help_parts else None

        w = st.sidebar.slider(
            f"{tier_badge(tier)} {label}",
            min_value=0, max_value=100, value=50, step=5,
            key=f"adv_rb_{z_col}",
            help=help_text,
        )
        if w > 0:
            effective_weights[z_col] = w

    # For save compatibility — advanced mode doesn't save, but community_section
    # expects bundle_weights to exist
    bundle_weights = {bk: 0 for bk in BUNDLES}


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
st.caption(
    "⚠️ Backs with very few carries have noisy scores — extreme values "
    "reflect small sample sizes, not skill. Use the 'Minimum carries' "
    "filter in the sidebar to hide low-volume backs if desired."
)
display_df = pd.DataFrame({
    "Rank": filtered.index,
    "Player": filtered["player_display_name"],
    "Carries": filtered["carries"].fillna(0).astype(int),
    "Rush yds": filtered["rush_yards"].fillna(0).astype(int),
    "Rush TDs": filtered["rush_tds"].fillna(0).astype(int),
    "Rec": filtered["receptions"].fillna(0).astype(int),
    "Rec yds": filtered["rec_yards"].fillna(0).astype(int),
    "Score": filtered["score"].apply(format_score),
})
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

with st.expander("ℹ️ How is this score calculated?"):
    st.markdown(SCORE_EXPLAINER)


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
    st.metric("Your score", format_score(player["score"]))

with c2:
    if not advanced_mode:
        st.markdown("**How your score breaks down**")
        bundle_rows = []
        for bk, bundle in active_bundles.items():
            bw = bundle_weights.get(bk, 0)
            if bw == 0:
                continue
            contribution = 0.0
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
            shown_stats = set()
            for bundle in active_bundles.values():
                shown_stats.update(bundle["stats"].keys())
            if 1 in new_enabled:
                for z_col, t in stat_tiers.items():
                    if t == 1:
                        shown_stats.add(z_col)

            for z_col in sorted(shown_stats, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                tier = stat_tiers.get(z_col, 2)
                label = stat_labels.get(z_col, z_col)
                raw_col = RAW_COL_MAP.get(z_col)
                z = player.get(z_col)
                raw = player.get(raw_col) if raw_col else None
                stat_rows.append({
                    "Tier": tier_badge(tier),
                    "Stat": label,
                    "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
                    "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                })
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("**Stat-by-stat breakdown** (z-score vs league)")
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            tier = stat_tiers.get(z_col, 2)
            label = stat_labels.get(z_col, z_col)
            raw_col = RAW_COL_MAP.get(z_col)
            z = player.get(z_col)
            raw = player.get(raw_col) if raw_col else None
            w = effective_weights.get(z_col, 0)
            contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            rows.append({
                "Tier": tier_badge(tier),
                "Stat": label,
                "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
                "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                "Weight": f"{w}",
                "Contribution": f"{contrib:+.2f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No stats weighted — drag some sliders.")


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

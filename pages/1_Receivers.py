"""
Lions Receiver Rater — Receivers page (tier migration)
======================================================
Tier-based slider UI for WR/TE rankings, matching the OL page's structure.

What the tier system does:
- Loads stat tiers and methodology from data/wr_stat_metadata.json.
- Tier checkboxes at the top of the page let users filter which stats
  participate in scoring. Tier 4 off by default; there are no Tier 4
  stats for receivers, so that checkbox is a no-op for now but we keep
  it for consistency with OL.
- When a tier is disabled, any stat in that tier is removed from every
  bundle. Bundles that end up empty disappear from the sidebar.
- Advanced mode shows per-stat sliders with ℹ️ methodology popovers
  (what/how/limits) for every stat.
- Leaderboard scores carry a label like "+0.47 (above group)".
- "How is this score calculated?" expander below the leaderboard.

What we preserved from the previous version:
- Positions filter (WR/TE multiselect)
- Minimum snaps filter
{
  "position_group": "receiver",
  "season": 2024,
  "team": "DET",
  "reference_population": "Top 64 WR + top 32 TE by offensive snaps, min 6 games",
  "output_population": "All DET WR/TE with 1+ offensive snaps",
  "n_players_output": 10,
  "stat_tiers": {
    "rec_yards_z": 1,
    "receptions_z": 1,
    "rec_tds_z": 1,
    "targets_z": 1,
    "catch_rate_z": 2,
    "success_rate_z": 2,
    "first_down_rate_z": 2,
    "yards_per_target_z": 2,
    "yards_per_snap_z": 2,
    "targets_per_snap_z": 2,
    "yac_per_reception_z": 2,
    "epa_per_target_z": 3,
    "avg_cpoe_z": 3,
    "yac_above_exp_z": 3,
    "avg_separation_z": 3
  },
  "stat_labels": {
    "rec_yards_z": "Receiving yards (raw)",
    "receptions_z": "Receptions (raw)",
    "rec_tds_z": "Receiving TDs (raw)",
    "targets_z": "Targets (raw)",
    "catch_rate_z": "Catch rate",
    "success_rate_z": "Success rate",
    "first_down_rate_z": "First-down rate",
    "yards_per_target_z": "Yards per target",
    "yards_per_snap_z": "Yards per snap",
    "targets_per_snap_z": "Targets per snap",
    "yac_per_reception_z": "YAC per reception",
    "epa_per_target_z": "EPA per target",
    "avg_cpoe_z": "CPOE",
    "yac_above_exp_z": "YAC over expected",
    "avg_separation_z": "Average separation"
  },
  "stat_methodology": {
    "rec_yards_z": {
      "what": "Total raw receiving yards.",
      "how": "Sum of PBP receiving_yards, z-scored against the reference population (top 64 WR + top 32 TE by snaps).",
      "limits": "Raw volume stat \u2014 rewards opportunity as much as skill. High-volume WR1s will always outrank efficient role players here."
    },
    "receptions_z": {
      "what": "Total raw receptions.",
      "how": "Count of complete passes where this player was the receiver, z-scored against the reference population.",
      "limits": "Volume stat. A possession receiver with 110 catches outranks a deep threat with 50 catches even if the deep threat averaged more yards."
    },
    "rec_tds_z": {
      "what": "Total raw receiving touchdowns.",
      "how": "Count of TDs on pass plays where this player was the receiver.",
      "limits": "Small integer samples are noisy. Four TDs vs. six TDs is a 50% difference but could easily be luck over 17 games."
    },
    "targets_z": {
      "what": "Total raw targets.",
      "how": "Count of pass plays where this player was the intended receiver.",
      "limits": "Pure opportunity \u2014 this is \"how much did the QB look your way,\" not a skill measure."
    },
    "catch_rate_z": {
      "what": "Percentage of targets caught.",
      "how": "receptions / targets.",
      "limits": "Drops and defended passes both count as incomplete. Doesn't account for target difficulty."
    },
    "success_rate_z": {
      "what": "Percentage of targets that produced a \"successful\" play by EPA standards.",
      "how": "nflverse tags each play with a binary success flag; we take the mean across this player's targets.",
      "limits": "Success is defined by an EPA threshold that varies by down/distance. The binary cutoff hides near-misses and runaway successes."
    },
    "first_down_rate_z": {
      "what": "Percentage of targets that gained a first down.",
      "how": "first_downs / targets.",
      "limits": "Chain-moving is valuable but depends on how the offense uses you \u2014 slot receivers on 3rd-and-short will post big numbers here."
    },
    "yards_per_target_z": {
      "what": "Average yards per target (not per reception).",
      "how": "total receiving yards / total targets.",
      "limits": "Penalizes drops as zeros. Rewards big plays disproportionately."
    },
    "yards_per_snap_z": {
      "what": "Receiving yards per offensive snap on the field.",
      "how": "total receiving yards / offensive snaps.",
      "limits": "Best efficiency-of-role metric from free data, but blocking TEs who rarely get targeted will look bad."
    },
    "targets_per_snap_z": {
      "what": "How often the QB looks your way per snap on the field.",
      "how": "targets / offensive snaps.",
      "limits": "Measures role, not skill. Schemed targets count the same as earned targets."
    },
    "yac_per_reception_z": {
      "what": "Average yards gained after the catch, per reception.",
      "how": "total yards_after_catch / receptions.",
      "limits": "Credit for YAC is shared between the receiver (did you break tackles / run well) and the scheme / blockers. Not purely a receiver stat."
    },
    "epa_per_target_z": {
      "what": "Expected Points Added per target.",
      "how": "mean of nflverse epa on this player's targets.",
      "limits": "EPA is a modeled value built from historical down/distance/field-position outcomes. Your score depends on trusting the EPA model."
    },
    "avg_cpoe_z": {
      "what": "Completion Percentage Over Expected.",
      "how": "nflverse computes expected completion probability based on throw difficulty, then this stat is actual_completion - expected. We average across the player's targets.",
      "limits": "Measures catching catches you're supposed to catch. A model decides what \"supposed to\" means, using throw distance, separation, etc."
    },
    "yac_above_exp_z": {
      "what": "Yards After Catch vs. what a league-average receiver would produce in the same situations.",
      "how": "NFL Next Gen Stats computes expected YAC from tracking data (defender proximity, angle, etc.); this is actual - expected.",
      "limits": "Requires NGS tracking data. Small-sample receivers may have missing or unstable values."
    },
    "avg_separation_z": {
      "what": "Average yards of separation from nearest defender at the moment of the catch.",
      "how": "NFL Next Gen Stats tracking data, season average.",
      "limits": "Depth-blind \u2014 a 2-yard separation on a deep route is more impressive than on a hitch. Doesn't tell you if separation came from route running or scheme."
    }
  },
  "invert_stats": [],
  "zscore_params": {
    "rec_yards": {
      "mean": 759.8645833333334,
      "std": 333.8034284391328
    },
    "receptions": {
      "mean": 62.229166666666664,
      "std": 26.405932015629276
    },
    "rec_tds": {
      "mean": 5.302083333333333,
      "std": 3.353300939817729
    },
    "targets": {
      "mean": 93.23958333333333,
      "std": 37.64816198537709
    },
    "catch_rate": {
      "mean": 0.669952487385956,
      "std": 0.08371328892138102
    },
    "success_rate": {
      "mean": 0.5314245071879173,
      "std": 0.07668027099833089
    },
    "first_down_rate": {
      "mean": 0.3940054842868282,
      "std": 0.07154238004457444
    },
    "yards_per_target": {
      "mean": 8.123640624463725,
      "std": 1.4050733105408424
    },
    "yac_per_reception": {
      "mean": 4.529667462627493,
      "std": 1.4070272822716339
    },
    "targets_per_snap": {
      "mean": 0.11074180775960829,
      "std": 0.037979721418954344
    },
    "yards_per_snap": {
      "mean": 0.8994516800380724,
      "std": 0.3376238608733308
    },
    "epa_per_target": {
      "mean": 0.2896097432721018,
      "std": 0.20274281751164158
    },
    "avg_cpoe": {
      "mean": 2.6641945233925335,
      "std": 6.042195592436554
    },
    "yac_above_exp": {
      "mean": 0.680665211673874,
      "std": 0.8193431150110553
    },
    "avg_separation": {
      "mean": 3.1251617528215068,
      "std": 0.5582120770916534
    }
  }
}
- Advanced mode toggle
- ?algo= deep link loading
- Community save/browse/fork/upvote via community_section
- Player Detail section with per-bundle and per-stat drill-down
- Small-sample caution caption above the leaderboard
- Footer with data credits

Design notes:
- Tier 1 raw counts (rec_yards_z etc.) are NOT added to any existing
  bundle. Adding raw volume to a "Reliability" bundle would break the
  bundle's meaning. Tier 1 stats are still accessible — they show up in
  Advanced mode. Bundle mode keeps its clean original design.
- Bundles carry a tier summary label like "🟢×0 🔵×3 🟡×1" so users can
  see at a glance which tiers they're trusting when they drag a slider.
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
    page_title="Lions Receiver Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

POSITION_GROUP = "receiver"
PAGE_URL = "https://lions-rater.streamlit.app/Receivers"

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "master_lions_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "wr_stat_metadata.json"


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
# Stat catalog — raw column names for Advanced mode display
# ============================================================
# Maps z-col → raw-col so the Player Detail section can show "raw 8.23 / z +0.47".
# This only needs the raw counterparts; labels/methodology come from the JSON.
RAW_COL_MAP = {
    "rec_yards_z": "rec_yards",
    "receptions_z": "receptions",
    "rec_tds_z": "rec_tds",
    "targets_z": "targets",
    "yards_per_target_z": "yards_per_target",
    "epa_per_target_z": "epa_per_target",
    "success_rate_z": "success_rate",
    "catch_rate_z": "catch_rate",
    "avg_cpoe_z": "avg_cpoe",
    "first_down_rate_z": "first_down_rate",
    "yac_per_reception_z": "yac_per_reception",
    "yac_above_exp_z": "yac_above_exp",
    "targets_per_snap_z": "targets_per_snap",
    "yards_per_snap_z": "yards_per_snap",
    "avg_separation_z": "avg_separation",
}


# ============================================================
# Bundles — Tier 2/3 organized, unchanged from previous version
# ============================================================
# Tier 1 raw counts are deliberately excluded from bundles (see module
# docstring). They're still accessible via Advanced mode.
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
**comparative** number telling you how Lions receivers stack up against
the top WR/TE population in the league, under the methodology *you* chose.

**League population:** z-scores are computed against the top 64 WRs and
top 32 TEs by offensive snaps (min 6 games played). Every Lions receiver
with at least one offensive snap is visible, but players with very few
targets will have noisy scores — read extreme values on low-volume players
as "small sample, not skill."
"""


# ============================================================
# Session state
# ============================================================
if "loaded_algo" not in st.session_state:
    st.session_state.loaded_algo = None
if "upvoted_ids" not in st.session_state:
    st.session_state.upvoted_ids = set()
if "rec_tiers_enabled" not in st.session_state:
    st.session_state.rec_tiers_enabled = [1, 2, 3]  # Tier 4 off by default


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
    "2024 regular season • Compared against top 64 WR + top 32 TE by snaps • "
    "Every Lions receiver visible"
)


# ============================================================
# Load data
# ============================================================
try:
    df = load_data()
except FileNotFoundError:
    st.error("Couldn't find the receivers data file.")
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
    help="Show individual stat sliders with methodology popovers instead of plain-English bundles.",
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
            value=(tier in st.session_state.rec_tiers_enabled),
            help=TIER_DESCRIPTIONS[tier],
            key=f"rec_tier_checkbox_{tier}",
        )
        if checked:
            new_enabled.append(tier)
st.session_state.rec_tiers_enabled = new_enabled

if not new_enabled:
    st.warning("Enable at least one tier to see ratings.")
    st.stop()

# Filter bundles to only those with stats in enabled tiers
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
            key=f"adv_rec_{z_col}",
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
            # Show stats from active bundles + Tier 1 stats if Tier 1 enabled
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

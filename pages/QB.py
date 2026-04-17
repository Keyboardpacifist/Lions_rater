"""
Lions QB Rater — Quarterbacks page
====================================
Tier-based slider UI for quarterback rankings. Parallel structure to the
Receivers, Running Backs, Offensive Line, Coaches, and GMs pages.

Default view:
- Active QBs with 2+ qualified seasons (200+ dropbacks/season)
- Stats are career-weighted averages (weighted by dropbacks per season)

Checkboxes:
- Include 1st-year QBs (shown with 🔴 small-sample flag)
- Include recent historical QBs

Data expected at data/master_qbs_with_z.parquet.
"""

import json
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm

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
    page_title="Lions QB Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

POSITION_GROUP = "qb"
PAGE_URL = "https://lions-rater.streamlit.app/QB"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "master_qbs_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "qb_stat_metadata.json"


# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_qbs_data():
    return pl.read_parquet(DATA_PATH).to_pandas()


@st.cache_data
def load_qbs_metadata():
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH) as f:
        return json.load(f)


# ============================================================
# Stat catalog
# ============================================================
RAW_COL_MAP = {
    "epa_per_play_z": "epa_per_play",
    "cpoe_z": "cpoe",
    "yards_per_attempt_z": "yards_per_attempt",
    "deep_ball_accuracy_z": "deep_ball_accuracy",
    "int_rate_z": "int_rate",
    "sack_rate_z": "sack_rate",
    "pressure_to_sack_rate_z": "pressure_to_sack_rate",
    "twp_rate_z": "twp_rate",
    "fourth_q_epa_z": "fourth_q_epa",
    "third_down_conv_rate_z": "third_down_conv_rate",
    "red_zone_td_rate_z": "red_zone_td_rate",
    "rush_epa_per_play_z": "rush_epa_per_play",
    "rush_yards_per_game_z": "rush_yards_per_game",
}


# ============================================================
# Bundles
# ============================================================
BUNDLES = {
    "accuracy": {
        "label": "🎯 Accuracy & efficiency",
        "description": "EPA per play, completion over expected, yards per attempt, deep ball accuracy.",
        "stats": {
            "epa_per_play_z": 0.30,
            "cpoe_z": 0.30,
            "yards_per_attempt_z": 0.20,
            "deep_ball_accuracy_z": 0.20,
        },
    },
    "decisions": {
        "label": "🧠 Decision-making",
        "description": "Avoids interceptions, escapes pressure, limits turnovers.",
        "stats": {
            "int_rate_z": 0.30,
            "sack_rate_z": 0.25,
            "pressure_to_sack_rate_z": 0.20,
            "twp_rate_z": 0.25,
        },
    },
    "clutch": {
        "label": "⚡ Clutch",
        "description": "4th-quarter performance, moves the chains on 3rd down, finishes in the red zone.",
        "stats": {
            "fourth_q_epa_z": 0.40,
            "third_down_conv_rate_z": 0.35,
            "red_zone_td_rate_z": 0.25,
        },
    },
    "mobility": {
        "label": "🏃 Mobility",
        "description": "Rushing value and volume. Rewards dual-threat QBs.",
        "stats": {
            "rush_epa_per_play_z": 0.60,
            "rush_yards_per_game_z": 0.40,
        },
    },
}

DEFAULT_BUNDLE_WEIGHTS = {
    "accuracy": 70,
    "decisions": 60,
    "clutch": 50,
    "mobility": 30,
}


# ============================================================
# Radar chart config
# ============================================================
RADAR_STATS = [
    "epa_per_play_z",
    "cpoe_z",
    "yards_per_attempt_z",
    "deep_ball_accuracy_z",
    "int_rate_z",
    "sack_rate_z",
    "pressure_to_sack_rate_z",
    "twp_rate_z",
    "fourth_q_epa_z",
    "third_down_conv_rate_z",
    "red_zone_td_rate_z",
    "rush_epa_per_play_z",
    "rush_yards_per_game_z",
]

RADAR_INVERT = set()  # Already inverted in the data pull

RADAR_LABEL_OVERRIDES = {
    "epa_per_play_z": "EPA/play",
    "cpoe_z": "CPOE",
    "yards_per_attempt_z": "YPA",
    "deep_ball_accuracy_z": "Deep ball",
    "int_rate_z": "INT avoidance",
    "sack_rate_z": "Sack avoidance",
    "pressure_to_sack_rate_z": "Pressure escape",
    "twp_rate_z": "TO avoidance",
    "fourth_q_epa_z": "4th Q EPA",
    "third_down_conv_rate_z": "3rd down",
    "red_zone_td_rate_z": "Red zone",
    "rush_epa_per_play_z": "Rush EPA",
    "rush_yards_per_game_z": "Rush yds/gm",
}


def zscore_to_percentile(z):
    if pd.isna(z):
        return None
    return float(norm.cdf(z) * 100)


def build_radar_figure(qb, stat_labels, stat_methodology):
    axes, values, descriptions = [], [], []
    for z_col in RADAR_STATS:
        if z_col not in qb.index:
            continue
        z = qb.get(z_col)
        if pd.isna(z):
            continue
        if z_col in RADAR_INVERT:
            z = -z
        pct = zscore_to_percentile(z)
        label = RADAR_LABEL_OVERRIDES.get(z_col, stat_labels.get(z_col, z_col))
        desc = stat_methodology.get(z_col, {}).get("what", "")
        axes.append(label)
        values.append(pct)
        descriptions.append(desc)

    if not axes:
        return None

    axes_closed = axes + [axes[0]]
    values_closed = values + [values[0]]
    descriptions_closed = descriptions + [descriptions[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=axes_closed,
        customdata=descriptions_closed,
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.25)",
        line=dict(color="rgba(31, 119, 180, 0.9)", width=2),
        marker=dict(size=6, color="rgba(31, 119, 180, 1)"),
        hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<br><br><i>%{customdata}</i><extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                ticktext=["25", "50", "75", "100"],
                tickfont=dict(size=9, color="#888"),
                gridcolor="#ddd",
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                gridcolor="#ddd",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=20, b=20),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


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


def filter_bundles_by_tier(bundles, stat_tiers, enabled_tiers):
    filtered = {}
    for bk, bdef in bundles.items():
        kept = {z: w for z, w in bdef["stats"].items() if stat_tiers.get(z, 2) in enabled_tiers}
        if kept:
            filtered[bk] = {"label": bdef["label"], "description": bdef["description"], "stats": kept}
    return filtered


def bundle_tier_summary(bundle_stats, stat_tiers):
    counts = {}
    for z in bundle_stats:
        t = stat_tiers.get(z, 2)
        counts[t] = counts.get(t, 0) + 1
    return " ".join(f"{tier_badge(t)}×{c}" for t, c in sorted(counts.items()))


# ============================================================
# Score labels
# ============================================================
def score_label(score):
    if pd.isna(score): return "—"
    if score >= 1.0: return "well above group"
    if score >= 0.4: return "above group"
    if score >= -0.4: return "about average"
    if score >= -1.0: return "below group"
    return "well below group"


def format_score(score):
    if pd.isna(score): return "—"
    sign = "+" if score >= 0 else ""
    return f"{sign}{score:.2f} ({score_label(score)})"


def sample_size_badge(seasons):
    if pd.isna(seasons): return ""
    if seasons < 2: return "🔴"
    if seasons < 4: return "🟡"
    return ""


def sample_size_caption(seasons):
    if pd.isna(seasons): return ""
    if seasons < 2:
        return f"⚠️ Only {int(seasons)} qualified season. Career stats are based on a very small sample. Treat as directional only."
    if seasons < 4:
        return f"⚠️ {int(seasons)} qualified seasons. Career averages may shift significantly with more data."
    return ""


SCORE_EXPLAINER = """
**What this number means.** The score is a weighted average of z-scores —
standardized stats where 0 is the QB-group average, +1 is one standard
deviation above, and −1 is one standard deviation below. Your slider
weights control how much each bundle contributes.

**How to read it:**
- `+1.0` or higher → well above the group average on what you weighted
- `+0.4` to `+1.0` → above average
- `−0.4` to `+0.4` → roughly average
- `−1.0` or lower → well below average

**What this is not.** It's not a PFF-style grade. It's a **comparative**
number telling you how quarterbacks stack up against each other under
the methodology *you* chose.

**QB population:** z-scores are computed against QBs with 2+ qualified
seasons (200+ dropbacks/season) from 2016-2024. All stats are per-play
rates, so volume doesn't matter — game managers and gunslingers are
compared on efficiency, not counting stats.

**Per-play normalization.** Every stat is a rate: EPA per dropback, INTs
per attempt, yards per attempt, etc. A QB who throws 400 times and one
who throws 600 times are compared on the same scale.

**Inverted stats.** Interception rate, sack rate, pressure-to-sack rate,
and turnover-worthy play rate are inverted so that a positive z-score
always means "good." Higher = fewer turnovers / fewer sacks.
"""


# ============================================================
# Session state
# ============================================================
if "qb_loaded_algo" not in st.session_state:
    st.session_state.qb_loaded_algo = None
if "upvoted_ids" not in st.session_state:
    st.session_state.upvoted_ids = set()
if "qb_tiers_enabled" not in st.session_state:
    st.session_state.qb_tiers_enabled = [1, 2, 3]


# ============================================================
# Header
# ============================================================
st.title("🦁 Lions QB Rater")
st.markdown(
    "**Build your own algorithm.** Drag the sliders to weight what you value, "
    "and watch the quarterbacks re-rank in real time. "
    "_No 'best QB' — just **your** best QB._"
)
st.caption(
    "Lions QBs shown by default • Z-scores computed league-wide for sample size • "
    "Per-play rates (volume-neutral) • "
    "Every stat has a methodology popover"
)


# ============================================================
# Load data
# ============================================================
try:
    df = load_qbs_data()
except FileNotFoundError:
    st.error(f"Couldn't find the QBs data file at {DATA_PATH}.")
    st.caption(
        "Run the QB data-pull script and upload the parquet + metadata "
        "files to `data/` in the repo."
    )
    st.stop()

meta = load_qbs_metadata()
stat_tiers = meta.get("stat_tiers", {})
stat_labels = meta.get("stat_labels", {})
stat_methodology = meta.get("stat_methodology", {})


# ============================================================
# ?algo= deep link
# ============================================================
if "algo" in st.query_params and st.session_state.qb_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP:
        apply_algo_weights(linked, BUNDLES)
        st.rerun()


# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Filters")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle(
    "🔬 Advanced mode", value=False,
    help="Show individual stat sliders with methodology tooltips instead of plain-English bundles.",
)

st.sidebar.header("What do you value?")

if st.session_state.qb_loaded_algo:
    la = st.session_state.qb_loaded_algo
    st.sidebar.info(f"Loaded: **{la['name']}** by {la['author']}\n\n_{la.get('description', '')}_")
    if st.sidebar.button("Clear loaded algorithm"):
        st.session_state.qb_loaded_algo = None


# ============================================================
# Tier filter
# ============================================================
st.markdown("### How speculative do you want to get?")
st.caption("Each stat is labeled by how much trust it asks from you. Uncheck tiers you don't want to include. Philosophy in a checkbox.")

tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        checked = st.checkbox(
            f"{tier_badge(tier)} {TIER_LABELS[tier]}",
            value=(tier in st.session_state.qb_tiers_enabled),
            help=TIER_DESCRIPTIONS[tier],
            key=f"qb_tier_checkbox_{tier}",
        )
        if checked:
            new_enabled.append(tier)

st.session_state.qb_tiers_enabled = new_enabled
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
            f"<div class='bundle-desc'>{bundle['description']}<br><small>{tier_summary}</small></div>",
            unsafe_allow_html=True,
        )
        if f"qb_bundle_{bk}" not in st.session_state:
            st.session_state[f"qb_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(
            bundle["label"], 0, 100, step=5, key=f"qb_bundle_{bk}", label_visibility="collapsed",
        )
    for bk in BUNDLES:
        if bk not in bundle_weights:
            bundle_weights[bk] = 0
    effective_weights = compute_effective_weights(active_bundles, bundle_weights)
else:
    st.sidebar.caption("Direct control over every underlying stat. Hover the ⓘ icon next to each slider for methodology.")
    st.sidebar.markdown(
        "<div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#888;margin-bottom:-0.5rem'>"
        "<span>\u2190 Low priority</span><span>High priority \u2192</span></div>",
        unsafe_allow_html=True,
    )
    all_enabled_stats = [z for z, t in stat_tiers.items() if t in new_enabled]
    all_enabled_stats.sort(key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))
    for z_col in all_enabled_stats:
        tier = stat_tiers.get(z_col, 2)
        label = stat_labels.get(z_col, z_col)
        meth = stat_methodology.get(z_col, {})
        help_parts = []
        if meth.get("what"): help_parts.append(f"What: {meth['what']}")
        if meth.get("how"): help_parts.append(f"How: {meth['how']}")
        if meth.get("limits"): help_parts.append(f"Limits: {meth['limits']}")
        help_text = "\n\n".join(help_parts) if help_parts else None
        w = st.sidebar.slider(
            f"{tier_badge(tier)} {label}", min_value=0, max_value=100, value=50, step=5,
            key=f"adv_qb_{z_col}", help=help_text,
        )
        if w > 0:
            effective_weights[z_col] = w
    bundle_weights = {bk: 0 for bk in BUNDLES}


# ============================================================
# Filter QB population
# ============================================================
st.markdown("### Who's in the pool?")
st.caption(
    "💡 **Why league-wide z-scores?** With only one Lions QB in the data, "
    "we need the full league to compute meaningful comparisons. Goff's scores "
    "tell you where he ranks among all qualified QBs. Check the box below to "
    "see the full league rankings."
)
c_opts = st.columns(3)
with c_opts[0]:
    show_league = st.checkbox(
        "Show league-wide QBs",
        value=False, key="qb_show_league",
        help="Reveals all qualified QBs across the league. Lions QBs are always included.",
    )
with c_opts[1]:
    include_rookies = st.checkbox(
        "Include 1st-year QBs",
        value=False, key="qb_include_rookies",
        help="Adds QBs with only 1 qualified season. Flagged 🔴 for small sample.",
    )
with c_opts[2]:
    include_historical = st.checkbox(
        "Include historical QBs",
        value=False, key="qb_include_historical",
        help="Adds QBs whose last qualified season was before 2024.",
    )

qbs = df.copy()

# Start with Lions-only, expand if user opts in
if show_league:
    keep_mask = (qbs["current_status"] == "active") & (qbs["seasons"].fillna(0) >= 2)
else:
    keep_mask = (qbs["current_status"] == "active") & (qbs["seasons"].fillna(0) >= 2) & (qbs["team"] == "DET")

# Always include Lions QBs regardless of other filters
lions_mask = qbs["team"] == "DET"

if include_rookies:
    if show_league:
        keep_mask = keep_mask | ((qbs["current_status"] == "active") & (qbs["seasons"].fillna(0) < 2))
    else:
        keep_mask = keep_mask | (lions_mask & (qbs["current_status"] == "active") & (qbs["seasons"].fillna(0) < 2))

if include_historical:
    if show_league:
        keep_mask = keep_mask | (qbs["current_status"] == "historical")
    else:
        keep_mask = keep_mask | (lions_mask & (qbs["current_status"] == "historical"))

# Always include any Lions QB
keep_mask = keep_mask | lions_mask

qbs = qbs[keep_mask].copy()

if len(qbs) == 0:
    st.warning("No QBs match the current filters.")
    st.stop()


# ============================================================
# Score
# ============================================================
qbs = score_players(qbs, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0:
    st.info("All weights are zero — drag some sliders to start ranking.")
qbs = qbs.sort_values("score", ascending=False).reset_index(drop=True)
qbs.index = qbs.index + 1


# ============================================================
# Ranking
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Ranking")

hide_small = st.checkbox(
    "Hide QBs with severe small samples (<2 seasons)",
    value=False, key="qb_hide_small",
    help="Hides red-flagged QBs. Yellow-flagged QBs still show with a caution.",
)

ranked = qbs.copy()
if hide_small:
    ranked = ranked[ranked["seasons"].fillna(0) >= 2].copy()
    if len(ranked) == 0:
        st.warning("All QBs are below the 2-season threshold. Uncheck the filter.")
        st.stop()
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1

if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("qb_name", "—")
    top_team = top.get("team", "")
    top_score = top["score"]
    top_seasons = top.get("seasons", 0)
    badge = sample_size_badge(top_seasons)
    sign = "+" if top_score >= 0 else ""
    team_part = f" ({top_team})" if top_team else ""
    st.markdown(
        f"<div style='background:#0076B6;color:white;padding:14px 20px;"
        f"border-radius:8px;margin-bottom:8px;font-size:1.1rem;'>"
        f"<span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span>"
        f" &nbsp;·&nbsp; <strong>{top_name}</strong>{team_part} {badge}"
        f" &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{sign}{top_score:.2f}</span>"
        f" <span style='opacity:0.85;'>({score_label(top_score)})</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    warn = sample_size_caption(top_seasons)
    if warn:
        st.warning(warn)

st.caption(
    "⚠️ QBs with fewer qualified seasons have noisier scores. "
    "🔴 = severe small sample (1 season), 🟡 = caution (2-3 seasons)."
)

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "": ranked["seasons"].apply(sample_size_badge),
    "QB": ranked["qb_name"],
    "Team": ranked.get("team", pd.Series(["—"] * len(ranked))),
    "Seasons": ranked["seasons"].fillna(0).astype(int),
    "Dropbacks": ranked.get("total_dropbacks", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "Games": ranked.get("total_games", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "Score": ranked["score"].apply(format_score),
})

st.dataframe(display_df, use_container_width=True, hide_index=True)

with st.expander("ℹ️ How is this score calculated?"):
    st.markdown(SCORE_EXPLAINER)


# ============================================================
# QB detail
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("QB detail")

selected = st.selectbox("Pick a QB to see how their score breaks down", options=ranked["qb_name"].tolist(), index=0)
qb = ranked[ranked["qb_name"] == selected].iloc[0]

warn = sample_size_caption(qb.get("seasons", 0))
if warn:
    st.warning(warn)

c1, c2 = st.columns([1, 1])
with c1:
    team = qb.get("team", "") if pd.notna(qb.get("team")) else ""
    st.markdown(f"### {selected}")
    st.caption(
        f"**{team}** · "
        f"{int(qb.get('seasons') or 0)} seasons · "
        f"{int(qb.get('total_games') or 0)} games · "
        f"{int(qb.get('total_dropbacks') or 0)} dropbacks"
    )
    st.markdown(f"**Your score:** {format_score(qb['score'])}")
    st.markdown("---")
    st.markdown("**How your score breaks down**")

    if not advanced_mode:
        bundle_rows = []
        for bk, bundle in active_bundles.items():
            bw = bundle_weights.get(bk, 0)
            if bw == 0: continue
            contribution = 0.0
            for z_col, internal in bundle["stats"].items():
                z = qb.get(z_col)
                if pd.notna(z) and total_weight > 0:
                    contribution += z * (bw * internal / total_weight)
            bundle_rows.append({"Bundle": bundle["label"], "Your weight": f"{bw}", "Contribution": f"{contribution:+.2f}"})
        if bundle_rows:
            st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No bundles weighted — drag some sliders.")

        with st.expander("🔬 See the underlying stats"):
            stat_rows = []
            shown_stats = set()
            for bundle in active_bundles.values():
                shown_stats.update(bundle["stats"].keys())
            for z_col in sorted(shown_stats, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                tier = stat_tiers.get(z_col, 2)
                label = stat_labels.get(z_col, z_col)
                raw_col = RAW_COL_MAP.get(z_col)
                z = qb.get(z_col)
                raw = qb.get(raw_col) if raw_col else None
                raw_str = f"{raw:.3f}" if pd.notna(raw) else "—"
                z_str = f"{z:+.2f}" if pd.notna(z) else "—"
                stat_rows.append({"Tier": tier_badge(tier), "Stat": label, "Raw": raw_str, "Z-score": z_str})
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Stat-by-stat breakdown (z-score vs QB group)")
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            tier = stat_tiers.get(z_col, 2)
            label = stat_labels.get(z_col, z_col)
            raw_col = RAW_COL_MAP.get(z_col)
            z = qb.get(z_col)
            raw = qb.get(raw_col) if raw_col else None
            w = effective_weights.get(z_col, 0)
            contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            raw_str = f"{raw:.3f}" if pd.notna(raw) else "—"
            z_str = f"{z:+.2f}" if pd.notna(z) else "—"
            rows.append({"Tier": tier_badge(tier), "Stat": label, "Raw": raw_str, "Z-score": z_str, "Weight": f"{w}", "Contribution": f"{contrib:+.2f}"})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No stats weighted — drag some sliders.")

with c2:
    st.markdown("**QB profile** (percentiles vs. QB group)")
    fig = build_radar_figure(qb, stat_labels, stat_methodology)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No radar data available for this QB.")
    st.caption(
        "Each axis shows where this QB ranks among the group. "
        "50 = median. Inverted stats (INT rate, sack rate, etc.) are "
        "flipped so higher = better on all axes. "
        "Hover any data point for the stat description."
    )


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
    "Play-by-play via nflfastR • "
    "All stats are per-play rates from 2016-2024 regular season • "
    "Built as a fan project, not affiliated with the NFL or the Detroit Lions."
)

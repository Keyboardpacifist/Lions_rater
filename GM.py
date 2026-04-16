"""
Lions GM Rater — GMs page
=================================
Tier-based slider UI for general manager rankings. Parallel structure to the
Receivers, Running Backs, Offensive Line, and Coaches pages.

Default view:
- Active GMs with 3+ full seasons
- Stats reflect full career as GM (across teams if applicable)

Checkboxes:
- Include newer GMs (1-2 seasons, shown with 🔴 small-sample flag)
- Include recent historical GMs

Data expected at data/master_gms_with_z.parquet. Required columns:
- gm_name, team, current_status ("active" / "historical"),
  seasons_as_gm, total_picks, fa_count, trade_count, udfa_count
- Raw stats: draft_hit_rate, day3_gem_rate, first_round_success,
  fa_hit_rate, fa_value_efficiency, trade_surplus, trade_activity,
  core_retention, udfa_hit_rate
- Z-scored stats: draft_hit_rate_z, day3_gem_rate_z, first_round_success_z,
  fa_hit_rate_z, fa_value_efficiency_z, trade_surplus_z, trade_activity_z,
  core_retention_z, udfa_hit_rate_z
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
    page_title="Lions GM Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

POSITION_GROUP = "gm"
PAGE_URL = "https://lions-rater.streamlit.app/GMs"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "master_gms_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "gm_stat_metadata.json"


# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_gms_data():
    return pl.read_parquet(DATA_PATH).to_pandas()


@st.cache_data
def load_gms_metadata():
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH) as f:
        return json.load(f)


# ============================================================
# Stat catalog — raw column names for GM Detail display
# ============================================================
RAW_COL_MAP = {
    "draft_hit_rate_z": "draft_hit_rate",
    "day3_gem_rate_z": "day3_gem_rate",
    "first_round_success_z": "first_round_success",
    "fa_hit_rate_z": "fa_hit_rate",
    "fa_value_efficiency_z": "fa_value_efficiency",
    "trade_surplus_z": "trade_surplus",
    "trade_activity_z": "trade_activity",
    "core_retention_z": "core_retention",
    "udfa_hit_rate_z": "udfa_hit_rate",
}


# ============================================================
# Bundles
# ============================================================
BUNDLES = {
    "drafting": {
        "label": "🎯 Drafting",
        "description": "Hits on picks, finds Day 3 gems, nails the first round.",
        "stats": {
            "draft_hit_rate_z": 0.40,
            "day3_gem_rate_z": 0.30,
            "first_round_success_z": 0.30,
        },
    },
    "free_agency": {
        "label": "💰 Free agency",
        "description": "Signs guys who actually play. Gets value for the money.",
        "stats": {
            "fa_hit_rate_z": 0.60,
            "fa_value_efficiency_z": 0.40,
        },
    },
    "trades": {
        "label": "🔄 Trades",
        "description": "Comes out ahead in swaps. Active on the phones.",
        "stats": {
            "trade_surplus_z": 0.70,
            "trade_activity_z": 0.30,
        },
    },
    "roster_building": {
        "label": "🧱 Roster building",
        "description": "Re-signs own draft picks. Mines UDFAs that stick.",
        "stats": {
            "core_retention_z": 0.60,
            "udfa_hit_rate_z": 0.40,
        },
    },
}

DEFAULT_BUNDLE_WEIGHTS = {
    "drafting": 70,
    "free_agency": 50,
    "trades": 40,
    "roster_building": 40,
}


# ============================================================
# Radar chart config — all 9 stats
# ============================================================
RADAR_STATS = [
    "draft_hit_rate_z",
    "day3_gem_rate_z",
    "first_round_success_z",
    "fa_hit_rate_z",
    "fa_value_efficiency_z",
    "trade_surplus_z",
    "trade_activity_z",
    "core_retention_z",
    "udfa_hit_rate_z",
]

RADAR_INVERT = set()  # No stats need inverting — higher is better for all 9

RADAR_LABEL_OVERRIDES = {
    "draft_hit_rate_z": "Draft hits",
    "day3_gem_rate_z": "Day 3 gems",
    "first_round_success_z": "1st-round value",
    "fa_hit_rate_z": "FA hits",
    "fa_value_efficiency_z": "FA value $",
    "trade_surplus_z": "Trade surplus",
    "trade_activity_z": "Trade activity",
    "core_retention_z": "Core retention",
    "udfa_hit_rate_z": "UDFA hits",
}


def zscore_to_percentile(z):
    if pd.isna(z):
        return None
    return float(norm.cdf(z) * 100)


def build_radar_figure(gm, stat_labels, stat_methodology):
    axes, values, descriptions = [], [], []
    for z_col in RADAR_STATS:
        if z_col not in gm.index:
            continue
        z = gm.get(z_col)
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


def filter_bundles_by_tier(bundles: dict, stat_tiers: dict, enabled_tiers: list) -> dict:
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


def sample_size_badge(seasons: float) -> str:
    """Badge based on GM tenure length.
    🔴 severe (<3 seasons — not enough draft cycles matured),
    🟡 caution (3-4 seasons — some stats still noisy),
    '' otherwise (5+ seasons)."""
    if pd.isna(seasons):
        return ""
    if seasons < 3:
        return "🔴"
    if seasons < 5:
        return "🟡"
    return ""


def sample_size_caption(seasons: float) -> str:
    if pd.isna(seasons):
        return ""
    if seasons < 3:
        return (
            f"⚠️ Very short tenure: {int(seasons)} seasons as GM. "
            f"Draft picks haven't had time to develop; core retention requires "
            f"4-5 years to be meaningful. Treat as directional only."
        )
    if seasons < 5:
        return (
            f"⚠️ Short tenure: {int(seasons)} seasons as GM. "
            f"Some stats (core retention, first-round success) may not be fully settled."
        )
    return ""


SCORE_EXPLAINER = """
**What this number means.** The score is a weighted average of z-scores —
standardized stats where 0 is the GM-group average, +1 is one standard
deviation above, and −1 is one standard deviation below. Your slider
weights control how much each bundle contributes.

**How to read it:**
- `+1.0` or higher → well above the group average on what you weighted
- `+0.4` to `+1.0` → above average
- `−0.4` to `+0.4` → roughly average
- `−1.0` or lower → well below average

**GM population:** z-scores are computed against active GMs with 3+
full seasons. Stats reflect each GM's full career (across teams if
applicable). Newer GMs with <3 seasons are flagged 🔴 and have noisier
scores — their draft classes haven't had time to develop and their
core retention isn't measurable yet.

**Data window:** 2013-2024. Pre-2013 GM decisions are not counted
because OverTheCap contract data gets patchy before that.

**GM attribution caveats.** GM responsibility is often shared — with
owners (Jerry Jones, Mike Brown), head coaches (Rivera in WAS,
Belichick in NE pre-2020), and shared front-office structures. The
"GM" credited here is the person publicly associated with personnel
decisions, which is sometimes different from who held the title. Also:
players and coaches ultimately decide who hits. These stats try to
isolate GM signal but can't do so perfectly. Treat big gaps as
meaningful and small gaps as noise.
"""


# ============================================================
# Session state
# ============================================================
if "gm_loaded_algo" not in st.session_state:
    st.session_state.gm_loaded_algo = None
if "upvoted_ids" not in st.session_state:
    st.session_state.upvoted_ids = set()
if "gm_tiers_enabled" not in st.session_state:
    st.session_state.gm_tiers_enabled = [1, 2, 3]


# ============================================================
# Header
# ============================================================
st.title("🦁 Lions GM Rater")
st.markdown(
    "**Build your own algorithm.** Drag the sliders to weight what you value, "
    "and watch the general managers re-rank in real time. "
    "_No 'best GM' — just **your** best GM._"
)
st.caption(
    "Full career stats • Compared against active GMs with 3+ full seasons • "
    "Every stat has a methodology popover"
)


# ============================================================
# Load data
# ============================================================
try:
    df = load_gms_data()
except FileNotFoundError:
    st.error(f"Couldn't find the GMs data file at {DATA_PATH}.")
    st.caption(
        "Run the GM data-pull script and upload the parquet + metadata "
        "files to `data/` in the repo."
    )
    st.stop()

meta = load_gms_metadata()
stat_tiers = meta.get("stat_tiers", {})
stat_labels = meta.get("stat_labels", {})
stat_methodology = meta.get("stat_methodology", {})


# ============================================================
# ?algo= deep link
# ============================================================
if "algo" in st.query_params and st.session_state.gm_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP:
        apply_algo_weights(linked, BUNDLES)
        st.rerun()


# ============================================================
# Sidebar — filters
# ============================================================
st.sidebar.header("Filters")

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle(
    "🔬 Advanced mode", value=False,
    help="Show individual stat sliders with methodology tooltips instead of plain-English bundles.",
)

st.sidebar.header("What do you value?")

if st.session_state.gm_loaded_algo:
    la = st.session_state.gm_loaded_algo
    st.sidebar.info(
        f"Loaded: **{la['name']}** by {la['author']}\n\n"
        f"_{la.get('description', '')}_"
    )
    if st.sidebar.button("Clear loaded algorithm"):
        st.session_state.gm_loaded_algo = None


# ============================================================
# Tier filter
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
            value=(tier in st.session_state.gm_tiers_enabled),
            help=TIER_DESCRIPTIONS[tier],
            key=f"gm_tier_checkbox_{tier}",
        )
        if checked:
            new_enabled.append(tier)

st.session_state.gm_tiers_enabled = new_enabled

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
        if f"gm_bundle_{bk}" not in st.session_state:
            st.session_state[f"gm_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(
            bundle["label"], 0, 100,
            step=5,
            key=f"gm_bundle_{bk}",
            label_visibility="collapsed",
        )

    for bk in BUNDLES:
        if bk not in bundle_weights:
            bundle_weights[bk] = 0

    effective_weights = compute_effective_weights(active_bundles, bundle_weights)

else:
    st.sidebar.caption(
        "Direct control over every underlying stat. Hover the ⓘ icon next to "
        "each slider for methodology."
    )
    st.sidebar.markdown(
        "<div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#888;margin-bottom:-0.5rem'>"
        "<span>\u2190 Low priority</span><span>High priority \u2192</span></div>",
        unsafe_allow_html=True,
    )

    all_enabled_stats = [
        z for z, t in stat_tiers.items() if t in new_enabled
    ]
    all_enabled_stats.sort(key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))

    for z_col in all_enabled_stats:
        tier = stat_tiers.get(z_col, 2)
        label = stat_labels.get(z_col, z_col)
        meth = stat_methodology.get(z_col, {})

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
            key=f"adv_gm_{z_col}",
            help=help_text,
        )
        if w > 0:
            effective_weights[z_col] = w

    bundle_weights = {bk: 0 for bk in BUNDLES}


# ============================================================
# Filter the GM population
# ============================================================
st.markdown("### Who's in the pool?")
c_opts = st.columns(2)
with c_opts[0]:
    include_rookies = st.checkbox(
        "Include newer GMs (1-2 seasons)",
        value=False,
        key="gm_include_rookies",
        help="Adds GMs with fewer than 3 full seasons. Flagged 🔴 for small sample.",
    )
with c_opts[1]:
    include_historical = st.checkbox(
        "Include recent historical GMs",
        value=False,
        key="gm_include_historical",
        help="Adds retired/fired GMs from the 2013-2024 window for historical comparison.",
    )

gms = df.copy()

# Base: active GMs with 3+ full seasons
keep_mask = (gms["current_status"] == "active") & (gms["seasons_as_gm"].fillna(0) >= 3)

if include_rookies:
    keep_mask = keep_mask | (
        (gms["current_status"] == "active") & (gms["seasons_as_gm"].fillna(0) < 3)
    )

if include_historical:
    keep_mask = keep_mask | (gms["current_status"] == "historical")

gms = gms[keep_mask].copy()

if len(gms) == 0:
    st.warning("No GMs match the current filters.")
    st.stop()


# ============================================================
# Score
# ============================================================
gms = score_players(gms, effective_weights)

total_weight = sum(effective_weights.values())
if total_weight == 0:
    st.info("All weights are zero — drag some sliders to start ranking.")

gms = gms.sort_values("score", ascending=False).reset_index(drop=True)
gms.index = gms.index + 1


# ============================================================
# Ranking
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Ranking")

hide_small = st.checkbox(
    "Hide GMs with severe small samples (<3 seasons)",
    value=False,
    key="gm_hide_small",
    help="Hides red-flagged GMs. Yellow-flagged GMs still show with a caution.",
)

ranked = gms.copy()
if hide_small:
    ranked = ranked[ranked["seasons_as_gm"].fillna(0) >= 3].copy()
    if len(ranked) == 0:
        st.warning("All GMs are below the 3-season threshold. Uncheck the filter to see them.")
        st.stop()
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1

# Top-ranked highlight banner
if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("gm_name", "—")
    top_team = top.get("team", "")
    top_score = top["score"]
    top_seasons = top.get("seasons_as_gm", 0)
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
    "⚠️ GMs with short tenures have noisier scores — their draft picks haven't fully developed. "
    "🔴 = severe small sample (<3 seasons), 🟡 = caution (3-4 seasons)."
)

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "": ranked["seasons_as_gm"].apply(sample_size_badge),
    "GM": ranked["gm_name"],
    "Team": ranked.get("team", pd.Series(["—"] * len(ranked))),
    "Seasons": ranked["seasons_as_gm"].fillna(0).astype(int),
    "Picks": ranked.get("total_picks", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "FA signings": ranked.get("fa_count", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "Trades": ranked.get("trade_count", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "Score": ranked["score"].apply(format_score),
})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

with st.expander("ℹ️ How is this score calculated?"):
    st.markdown(SCORE_EXPLAINER)


# ============================================================
# GM detail
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("GM detail")

selected = st.selectbox(
    "Pick a GM to see how their score breaks down",
    options=ranked["gm_name"].tolist(),
    index=0,
)

gm = ranked[ranked["gm_name"] == selected].iloc[0]

warn = sample_size_caption(gm.get("seasons_as_gm", 0))
if warn:
    st.warning(warn)

c1, c2 = st.columns([1, 1])
with c1:
    team = gm.get("team", "") if pd.notna(gm.get("team")) else ""
    st.markdown(f"### {selected}")
    st.caption(
        f"**{team}** · "
        f"{int(gm.get('seasons_as_gm') or 0)} seasons · "
        f"{int(gm.get('total_picks') or 0)} picks · "
        f"{int(gm.get('fa_count') or 0)} FA signings · "
        f"{int(gm.get('trade_count') or 0)} trades"
    )
    st.markdown(f"**Your score:** {format_score(gm['score'])}")
    st.markdown("---")
    st.markdown("**How your score breaks down**")

    if not advanced_mode:
        bundle_rows = []
        for bk, bundle in active_bundles.items():
            bw = bundle_weights.get(bk, 0)
            if bw == 0:
                continue
            contribution = 0.0
            for z_col, internal in bundle["stats"].items():
                z = gm.get(z_col)
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
            for z_col in sorted(shown_stats, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                tier = stat_tiers.get(z_col, 2)
                label = stat_labels.get(z_col, z_col)
                raw_col = RAW_COL_MAP.get(z_col)
                z = gm.get(z_col)
                raw = gm.get(raw_col) if raw_col else None
                # Special message for core_retention NaN on newer GMs
                if z_col == "core_retention_z" and pd.isna(z):
                    raw_str = "N/A (tenure too short)"
                    z_str = "—"
                else:
                    raw_str = f"{raw:.3f}" if pd.notna(raw) else "—"
                    z_str = f"{z:+.2f}" if pd.notna(z) else "—"
                stat_rows.append({
                    "Tier": tier_badge(tier),
                    "Stat": label,
                    "Raw": raw_str,
                    "Z-score": z_str,
                })
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    else:
        st.caption("Stat-by-stat breakdown (z-score vs GM group)")
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            tier = stat_tiers.get(z_col, 2)
            label = stat_labels.get(z_col, z_col)
            raw_col = RAW_COL_MAP.get(z_col)
            z = gm.get(z_col)
            raw = gm.get(raw_col) if raw_col else None
            w = effective_weights.get(z_col, 0)
            contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            # Special message for core_retention NaN on newer GMs
            if z_col == "core_retention_z" and pd.isna(z):
                raw_str = "N/A (tenure too short)"
                z_str = "—"
            else:
                raw_str = f"{raw:.3f}" if pd.notna(raw) else "—"
                z_str = f"{z:+.2f}" if pd.notna(z) else "—"
            rows.append({
                "Tier": tier_badge(tier),
                "Stat": label,
                "Raw": raw_str,
                "Z-score": z_str,
                "Weight": f"{w}",
                "Contribution": f"{contrib:+.2f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No stats weighted — drag some sliders.")

with c2:
    st.markdown("**GM profile** (percentiles vs. GM group)")
    fig = build_radar_figure(gm, stat_labels, stat_methodology)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No radar data available for this GM.")
    st.caption(
        "Each axis shows where this GM ranks among the group. "
        "50 = median. Hover any data point for the stat description."
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
    "Contract data via OverTheCap via nflverse • "
    "GM tenure hand-compiled from public records • "
    "Built as a fan project, not affiliated with the NFL or the Detroit Lions."
)

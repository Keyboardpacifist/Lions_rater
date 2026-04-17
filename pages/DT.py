"""
Lions DT Rater — Defensive Tackles page
==========================================
Tier-based slider UI for DT/IDL rankings. Parallel structure to all other pages.

Default view:
- Active DTs with 2+ qualified seasons (200+ defensive snaps/season)
- Stats are career-weighted averages (weighted by snap count per season)

Data expected at data/master_dts_with_z.parquet.
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
    page_title="Lions DT Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

POSITION_GROUP = "dt"
PAGE_URL = "https://lions-rater.streamlit.app/DT"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "master_lions_dts_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dt_stat_metadata.json"


@st.cache_data
def load_dts_data():
    return pl.read_parquet(DATA_PATH).to_pandas()


@st.cache_data
def load_dts_metadata():
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH) as f:
        return json.load(f)


# ============================================================
# Stat catalog
# ============================================================
RAW_COL_MAP = {
    "sacks_per_game_z": "sacks_per_game",
    "qb_hits_per_game_z": "qb_hits_per_game",
    "pressure_rate_z": "pressure_rate",
    "tfl_per_game_z": "tfl_per_game",
    "solo_tackle_rate_z": "solo_tackle_rate",
    "tackles_per_snap_z": "tackles_per_snap",
    "forced_fumbles_per_game_z": "forced_fumbles_per_game",
    "passes_defended_per_game_z": "passes_defended_per_game",
}


# ============================================================
# Bundles
# ============================================================
BUNDLES = {
    "pass_rush": {
        "label": "🔥 Pass rush",
        "description": "Gets to the QB. Sacks, hits, and total pressures.",
        "stats": {
            "sacks_per_game_z": 0.35,
            "qb_hits_per_game_z": 0.30,
            "pressure_rate_z": 0.35,
        },
    },
    "run_defense": {
        "label": "🛡️ Run defense",
        "description": "Stops the run. Penetrates the backfield. Makes solo tackles.",
        "stats": {
            "tfl_per_game_z": 0.40,
            "solo_tackle_rate_z": 0.30,
            "tackles_per_snap_z": 0.30,
        },
    },
    "playmaking": {
        "label": "💥 Playmaking",
        "description": "Forces fumbles and bats passes. Game-changing disruption.",
        "stats": {
            "forced_fumbles_per_game_z": 0.50,
            "passes_defended_per_game_z": 0.50,
        },
    },
}

DEFAULT_BUNDLE_WEIGHTS = {
    "pass_rush": 60,
    "run_defense": 50,
    "playmaking": 30,
}


# ============================================================
# Radar chart config
# ============================================================
RADAR_STATS = [
    "sacks_per_game_z",
    "qb_hits_per_game_z",
    "pressure_rate_z",
    "tfl_per_game_z",
    "solo_tackle_rate_z",
    "tackles_per_snap_z",
    "forced_fumbles_per_game_z",
    "passes_defended_per_game_z",
]

RADAR_INVERT = set()

RADAR_LABEL_OVERRIDES = {
    "sacks_per_game_z": "Sacks",
    "qb_hits_per_game_z": "QB hits",
    "pressure_rate_z": "Pressure",
    "tfl_per_game_z": "TFLs",
    "solo_tackle_rate_z": "Solo tackle %",
    "tackles_per_snap_z": "Tackles/snap",
    "forced_fumbles_per_game_z": "Forced fumbles",
    "passes_defended_per_game_z": "Pass defense",
}


def zscore_to_percentile(z):
    if pd.isna(z):
        return None
    return float(norm.cdf(z) * 100)


def build_radar_figure(player, stat_labels, stat_methodology):
    axes, values, descriptions = [], [], []
    for z_col in RADAR_STATS:
        if z_col not in player.index:
            continue
        z = player.get(z_col)
        if pd.isna(z):
            continue
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
        r=values_closed, theta=axes_closed, customdata=descriptions_closed,
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.25)",
        line=dict(color="rgba(31, 119, 180, 0.9)", width=2),
        marker=dict(size=6, color="rgba(31, 119, 180, 1)"),
        hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<br><br><i>%{customdata}</i><extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[25, 50, 75, 100],
                            ticktext=["25", "50", "75", "100"], tickfont=dict(size=9, color="#888"), gridcolor="#ddd"),
            angularaxis=dict(tickfont=dict(size=11), gridcolor="#ddd"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False, margin=dict(l=60, r=60, t=20, b=20), height=380, paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ============================================================
# Tier helpers
# ============================================================
TIER_LABELS = {1: "Tier 1 — Counted", 2: "Tier 2 — Contextualized", 3: "Tier 3 — Adjusted", 4: "Tier 4 — Inferred"}
TIER_DESCRIPTIONS = {
    1: "Pure recorded facts. No modeling.",
    2: "Counts divided by opportunity. Still no modeling.",
    3: "Compared against a modeled baseline. Model is simple and visible.",
    4: "Inferred from patterns the data can't directly see. Use with skepticism.",
}


def tier_badge(tier): return {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠"}.get(tier, "⚪")


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
        return f"⚠️ Only {int(seasons)} qualified season. Treat as directional only."
    if seasons < 4:
        return f"⚠️ {int(seasons)} qualified seasons. Career averages may shift with more data."
    return ""


SCORE_EXPLAINER = """
**What this number means.** The score is a weighted average of z-scores —
standardized stats where 0 is the DT-group average, +1 is one standard
deviation above, and −1 is one standard deviation below.

**How to read it:**
- `+1.0` or higher → well above the group average on what you weighted
- `+0.4` to `+1.0` → above average
- `−0.4` to `+0.4` → roughly average
- `−1.0` or lower → well below average

**DT population:** z-scores are computed against DTs with 2+ qualified
seasons (200+ defensive snaps/season) from 2016-2024. Stats are
career-weighted by snap count so high-snap seasons count more.

**Position note:** This page covers interior defensive linemen — DTs
and NTs. Edge rushers (DEs/OLBs) are a different archetype and will
have their own page.

**Scheme caveats.** A 1-tech nose tackle asked to eat double teams
will naturally have fewer sacks and TFLs than a 3-tech penetrator.
These stats reward disruption more than run-stuffing, which is a real
limitation. Treat the rankings as "who disrupts more" rather than
"who is better at their role."
"""


# ============================================================
# Session state
# ============================================================
if "dt_loaded_algo" not in st.session_state:
    st.session_state.dt_loaded_algo = None
if "upvoted_ids" not in st.session_state:
    st.session_state.upvoted_ids = set()
if "dt_tiers_enabled" not in st.session_state:
    st.session_state.dt_tiers_enabled = [1, 2]


# ============================================================
# Header
# ============================================================
st.title("🦁 Lions DT Rater")
st.markdown(
    "**Build your own algorithm.** Drag the sliders to weight what you value, "
    "and watch the defensive tackles re-rank in real time. "
    "_No 'best DT' — just **your** best DT._"
)
st.caption(
    "Lions DTs shown • Z-scores computed against league-wide DTs (200+ snaps/season) • "
    "Per-game and per-snap rates • "
    "Every stat has a methodology popover"
)


# ============================================================
# Load data
# ============================================================
try:
    df = load_dts_data()
except FileNotFoundError:
    st.error(f"Couldn't find the DTs data file at {DATA_PATH}.")
    st.caption("Run the DT data-pull script and upload the parquet + metadata files to `data/` in the repo.")
    st.stop()

meta = load_dts_metadata()
stat_tiers = meta.get("stat_tiers", {})
stat_labels = meta.get("stat_labels", {})
stat_methodology = meta.get("stat_methodology", {})


# ============================================================
# ?algo= deep link
# ============================================================
if "algo" in st.query_params and st.session_state.dt_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP:
        apply_algo_weights(linked, BUNDLES)
        st.rerun()


# ============================================================
# Sidebar
# ============================================================
st.sidebar.header("Filters")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle("🔬 Advanced mode", value=False,
    help="Show individual stat sliders with methodology tooltips instead of plain-English bundles.")

st.sidebar.header("What do you value?")

if st.session_state.dt_loaded_algo:
    la = st.session_state.dt_loaded_algo
    st.sidebar.info(f"Loaded: **{la['name']}** by {la['author']}\n\n_{la.get('description', '')}_")
    if st.sidebar.button("Clear loaded algorithm"):
        st.session_state.dt_loaded_algo = None


# ============================================================
# Tier filter
# ============================================================
st.markdown("### How speculative do you want to get?")
st.caption("Each stat is labeled by how much trust it asks from you. Uncheck tiers you don't want to include.")

tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        checked = st.checkbox(f"{tier_badge(tier)} {TIER_LABELS[tier]}",
            value=(tier in st.session_state.dt_tiers_enabled), help=TIER_DESCRIPTIONS[tier], key=f"dt_tier_checkbox_{tier}")
        if checked:
            new_enabled.append(tier)

st.session_state.dt_tiers_enabled = new_enabled
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
            unsafe_allow_html=True)
        if f"dt_bundle_{bk}" not in st.session_state:
            st.session_state[f"dt_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(bundle["label"], 0, 100, step=5,
            key=f"dt_bundle_{bk}", label_visibility="collapsed")
    for bk in BUNDLES:
        if bk not in bundle_weights:
            bundle_weights[bk] = 0
    effective_weights = compute_effective_weights(active_bundles, bundle_weights)
else:
    st.sidebar.caption("Direct control over every underlying stat.")
    st.sidebar.markdown(
        "<div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#888;margin-bottom:-0.5rem'>"
        "<span>\u2190 Low priority</span><span>High priority \u2192</span></div>", unsafe_allow_html=True)
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
        w = st.sidebar.slider(f"{tier_badge(tier)} {label}", min_value=0, max_value=100, value=50, step=5,
            key=f"adv_dt_{z_col}", help=help_text)
        if w > 0:
            effective_weights[z_col] = w
    bundle_weights = {bk: 0 for bk in BUNDLES}


# ============================================================
# Filter population
# ============================================================
st.markdown("### Who's in the pool?")
st.caption(
    "All Lions DTs with 200+ defensive snaps in a season. "
    "Z-scores are computed against league-wide DTs for meaningful comparison."
)

dts = df.copy()

if len(dts) == 0:
    st.warning("No DTs found in the data.")
    st.stop()


# ============================================================
# Score
# ============================================================
dts = score_players(dts, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0:
    st.info("All weights are zero — drag some sliders to start ranking.")
dts = dts.sort_values("score", ascending=False).reset_index(drop=True)
dts.index = dts.index + 1


# ============================================================
# Ranking
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Ranking")

hide_small = st.checkbox("Hide DTs with severe small samples (<2 seasons)", value=False, key="dt_hide_small")

ranked = dts.copy()
if hide_small:
    ranked = ranked[ranked["seasons"].fillna(0) >= 2].copy()
    if len(ranked) == 0:
        st.warning("All DTs are below the 2-season threshold.")
        st.stop()
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1

if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("player_name", "—")
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
        f"</div>", unsafe_allow_html=True)
    warn = sample_size_caption(top_seasons)
    if warn:
        st.warning(warn)

st.caption("⚠️ DTs with fewer seasons have noisier scores. "
    "🔴 = severe small sample (1 season), 🟡 = caution (2-3 seasons).")

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "": ranked["seasons"].apply(sample_size_badge),
    "Player": ranked["player_name"],
    "Team": ranked.get("team", pd.Series(["—"] * len(ranked))),
    "Seasons": ranked["seasons"].fillna(0).astype(int),
    "Games": ranked.get("total_games", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "Snaps": ranked.get("total_snaps", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "Score": ranked["score"].apply(format_score),
})

st.dataframe(display_df, use_container_width=True, hide_index=True)

with st.expander("ℹ️ How is this score calculated?"):
    st.markdown(SCORE_EXPLAINER)


# ============================================================
# Player detail
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")

selected = st.selectbox("Pick a DT to see how their score breaks down",
    options=ranked["player_name"].tolist(), index=0)
player = ranked[ranked["player_name"] == selected].iloc[0]

warn = sample_size_caption(player.get("seasons", 0))
if warn:
    st.warning(warn)

c1, c2 = st.columns([1, 1])
with c1:
    team = player.get("team", "") if pd.notna(player.get("team")) else ""
    st.markdown(f"### {selected}")
    st.caption(
        f"**{team}** · "
        f"{int(player.get('seasons') or 0)} seasons · "
        f"{int(player.get('total_games') or 0)} games · "
        f"{int(player.get('total_snaps') or 0)} snaps"
    )
    st.markdown(f"**Your score:** {format_score(player['score'])}")
    st.markdown("---")
    st.markdown("**How your score breaks down**")

    if not advanced_mode:
        bundle_rows = []
        for bk, bundle in active_bundles.items():
            bw = bundle_weights.get(bk, 0)
            if bw == 0: continue
            contribution = 0.0
            for z_col, internal in bundle["stats"].items():
                z = player.get(z_col)
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
                z = player.get(z_col)
                raw = player.get(raw_col) if raw_col else None
                stat_rows.append({
                    "Tier": tier_badge(tier), "Stat": label,
                    "Raw": f"{raw:.3f}" if pd.notna(raw) else "—",
                    "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                })
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Stat-by-stat breakdown (z-score vs DT group)")
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
                "Tier": tier_badge(tier), "Stat": label,
                "Raw": f"{raw:.3f}" if pd.notna(raw) else "—",
                "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                "Weight": f"{w}", "Contribution": f"{contrib:+.2f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No stats weighted — drag some sliders.")

with c2:
    st.markdown("**DT profile** (percentiles vs. DT group)")
    fig = build_radar_figure(player, stat_labels, stat_methodology)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No radar data available for this DT.")
    st.caption("Each axis shows where this DT ranks among the group. 50 = median. "
        "Hover any data point for the stat description.")


# ============================================================
# Community algorithms
# ============================================================
community_section(
    position_group=POSITION_GROUP, bundles=BUNDLES,
    bundle_weights=bundle_weights, advanced_mode=advanced_mode, page_url=PAGE_URL,
)


# ============================================================
# Footer
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    "Data via [nflverse](https://github.com/nflverse) • "
    "Play-by-play via nflfastR • Snap counts via Pro Football Reference • "
    "All stats from 2016-2024 regular season • "
    "Built as a fan project, not affiliated with the NFL or the Detroit Lions."
)

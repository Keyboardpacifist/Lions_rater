"""
Lions CB Rater — Cornerbacks page
====================================
8 stats across 3 bundles + 1 Tier 4 bundle.
Tier 4 man coverage stats (2023-2024 only) disabled by default.
"""

import json
from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from lib_shared import apply_algo_weights, community_section, compute_effective_weights, get_algorithm_by_slug, inject_css, score_players

st.set_page_config(page_title="Lions CB Rater", page_icon="🦁", layout="wide", initial_sidebar_state="expanded")
inject_css()

POSITION_GROUP = "cb"
PAGE_URL = "https://lions-rater.streamlit.app/CB"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "master_lions_cbs_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cb_stat_metadata.json"

@st.cache_data
def load_cbs_data(): return pl.read_parquet(DATA_PATH).to_pandas()
@st.cache_data
def load_cbs_metadata():
    if not METADATA_PATH.exists(): return {}
    with open(METADATA_PATH) as f: return json.load(f)

RAW_COL_MAP = {
    "passes_defended_per_game_z": "passes_defended_per_game",
    "interceptions_per_game_z": "interceptions_per_game",
    "solo_tackle_rate_z": "solo_tackle_rate",
    "tackles_per_snap_z": "tackles_per_snap",
    "tfl_per_game_z": "tfl_per_game",
    "forced_fumbles_per_game_z": "forced_fumbles_per_game",
    "man_coverage_epa_z": "man_coverage_epa",
    "man_coverage_comp_pct_z": "man_coverage_comp_pct",
}

BUNDLES = {
    "coverage": {"label": "🛡️ Coverage", "description": "Breaks up passes and picks off the QB. The CB's primary job.", "stats": {"passes_defended_per_game_z": 0.55, "interceptions_per_game_z": 0.45}},
    "tackling": {"label": "🏋️ Tackling & run support", "description": "Makes tackles, gets to the backfield, supports the run game.", "stats": {"tackles_per_snap_z": 0.35, "solo_tackle_rate_z": 0.35, "tfl_per_game_z": 0.30}},
    "playmaking": {"label": "💥 Playmaking", "description": "Creates turnovers — forced fumbles and opportunistic plays.", "stats": {"forced_fumbles_per_game_z": 1.0}},
    "man_coverage": {"label": "🔬 Man coverage (theoretical)", "description": "⚠️ TIER 4: How the defense performs on pass plays in man coverage when this CB is on the field. Based on 2023-2024 participation data. We know the team was in man, but NOT which receiver this CB was covering.", "stats": {"man_coverage_epa_z": 0.50, "man_coverage_comp_pct_z": 0.50}},
}
DEFAULT_BUNDLE_WEIGHTS = {"coverage": 70, "tackling": 30, "playmaking": 40, "man_coverage": 40}

RADAR_STATS = ["passes_defended_per_game_z", "interceptions_per_game_z", "solo_tackle_rate_z", "tackles_per_snap_z", "tfl_per_game_z", "forced_fumbles_per_game_z", "man_coverage_epa_z", "man_coverage_comp_pct_z"]
RADAR_INVERT = set()
RADAR_LABEL_OVERRIDES = {"passes_defended_per_game_z": "Pass defense", "interceptions_per_game_z": "Interceptions", "solo_tackle_rate_z": "Solo tackle %", "tackles_per_snap_z": "Tackles/snap", "tfl_per_game_z": "TFLs", "forced_fumbles_per_game_z": "Forced fumbles", "man_coverage_epa_z": "Man cov EPA", "man_coverage_comp_pct_z": "Man cov comp%"}

def zscore_to_percentile(z):
    if pd.isna(z): return None
    return float(norm.cdf(z) * 100)

def build_radar_figure(player, stat_labels, stat_methodology):
    axes, values, descriptions = [], [], []
    for z_col in RADAR_STATS:
        if z_col not in player.index: continue
        z = player.get(z_col)
        if pd.isna(z): continue
        pct = zscore_to_percentile(z)
        label = RADAR_LABEL_OVERRIDES.get(z_col, stat_labels.get(z_col, z_col))
        desc = stat_methodology.get(z_col, {}).get("what", "")
        axes.append(label); values.append(pct); descriptions.append(desc)
    if not axes: return None
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values+[values[0]], theta=axes+[axes[0]], customdata=descriptions+[descriptions[0]], fill="toself", fillcolor="rgba(31, 119, 180, 0.25)", line=dict(color="rgba(31, 119, 180, 0.9)", width=2), marker=dict(size=6, color="rgba(31, 119, 180, 1)"), hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<br><br><i>%{customdata}</i><extra></extra>"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickvals=[25, 50, 75, 100], ticktext=["25", "50", "75", "100"], tickfont=dict(size=9, color="#888"), gridcolor="#ddd"), angularaxis=dict(tickfont=dict(size=11), gridcolor="#ddd"), bgcolor="rgba(0,0,0,0)"), showlegend=False, margin=dict(l=60, r=60, t=20, b=20), height=380, paper_bgcolor="rgba(0,0,0,0)")
    return fig

TIER_LABELS = {1: "Tier 1 — Counted", 2: "Tier 2 — Contextualized", 3: "Tier 3 — Adjusted", 4: "Tier 4 — Inferred"}
TIER_DESCRIPTIONS = {1: "Pure recorded facts.", 2: "Counts divided by opportunity.", 3: "Compared against a modeled baseline.", 4: "Inferred from patterns. Theoretical — use with skepticism. Man coverage stats require 2023-2024 participation data."}
def tier_badge(tier): return {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠"}.get(tier, "⚪")

def filter_bundles_by_tier(bundles, stat_tiers, enabled_tiers):
    filtered = {}
    for bk, bdef in bundles.items():
        kept = {z: w for z, w in bdef["stats"].items() if stat_tiers.get(z, 2) in enabled_tiers}
        if kept: filtered[bk] = {"label": bdef["label"], "description": bdef["description"], "stats": kept}
    return filtered

def bundle_tier_summary(bundle_stats, stat_tiers):
    counts = {}
    for z in bundle_stats: t = stat_tiers.get(z, 2); counts[t] = counts.get(t, 0) + 1
    return " ".join(f"{tier_badge(t)}×{c}" for t, c in sorted(counts.items()))

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
    if seasons < 2: return f"⚠️ Only {int(seasons)} qualified season. Treat as directional only."
    if seasons < 4: return f"⚠️ {int(seasons)} qualified seasons. Career averages may shift."
    return ""

SCORE_EXPLAINER = """
**What this number means.** Weighted average of z-scores — 0 is league-average CB, +1 is one SD above, −1 is one SD below.

**How to read it:** `+1.0` or higher → well above average • `+0.4` to `+1.0` → above average • `−0.4` to `+0.4` → roughly average • `−1.0` or lower → well below average

**CB population:** z-scores computed against all CBs league-wide with 2+ qualified seasons (200+ defensive snaps/season) from 2016-2024.

**What we can't measure (without PFF):** The best thing a CB can do is make the QB throw somewhere else. Free data can't capture "avoidance rate" — how often QBs refuse to throw at a CB. Our stats measure what happens when the ball comes near the CB, not how often it doesn't.

**Tier 4 — Man coverage (theoretical):** When enabled, two stats measure team defense performance on man coverage pass plays when this CB is on the field (2023-2024 only). We know the team was in man coverage, and we know this CB was playing, but we do NOT know which receiver was his assignment. Treat as a directional signal, not individual coverage grades.
"""

if "cb_loaded_algo" not in st.session_state: st.session_state.cb_loaded_algo = None
if "upvoted_ids" not in st.session_state: st.session_state.upvoted_ids = set()
if "cb_tiers_enabled" not in st.session_state: st.session_state.cb_tiers_enabled = [1, 2]

st.title("🦁 Lions CB Rater")
st.markdown("**Build your own algorithm.** Drag the sliders to weight what you value, and watch the Lions cornerbacks re-rank in real time. _No 'best CB' — just **your** best CB._")
st.caption("Lions CBs shown • Z-scores computed against league-wide CBs • Tier 4 man coverage stats available (2023-2024)")

try: df = load_cbs_data()
except FileNotFoundError: st.error(f"Couldn't find CBs data at {DATA_PATH}."); st.stop()

meta = load_cbs_metadata()
stat_tiers = meta.get("stat_tiers", {}); stat_labels = meta.get("stat_labels", {}); stat_methodology = meta.get("stat_methodology", {})

if "algo" in st.query_params and st.session_state.cb_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP: apply_algo_weights(linked, BUNDLES); st.rerun()

st.sidebar.header("Filters")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle("🔬 Advanced mode", value=False, help="Show individual stat sliders.")
st.sidebar.header("What do you value?")

if st.session_state.cb_loaded_algo:
    la = st.session_state.cb_loaded_algo
    st.sidebar.info(f"Loaded: **{la['name']}** by {la['author']}\n\n_{la.get('description', '')}_")
    if st.sidebar.button("Clear loaded algorithm"): st.session_state.cb_loaded_algo = None

st.markdown("### How speculative do you want to get?")
st.caption("Each stat is labeled by how much trust it asks from you. **Tier 4** includes theoretical man coverage stats — disabled by default.")
tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        checked = st.checkbox(f"{tier_badge(tier)} {TIER_LABELS[tier]}", value=(tier in st.session_state.cb_tiers_enabled), help=TIER_DESCRIPTIONS[tier], key=f"cb_tier_checkbox_{tier}")
        if checked: new_enabled.append(tier)
st.session_state.cb_tiers_enabled = new_enabled
if not new_enabled: st.warning("Enable at least one tier."); st.stop()
active_bundles = filter_bundles_by_tier(BUNDLES, stat_tiers, new_enabled)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

bundle_weights = {}; effective_weights = {}
if not advanced_mode:
    if not active_bundles: st.info("No bundles in enabled tiers."); st.stop()
    st.sidebar.caption("Drag to weight what matters to you. 0 = ignore, 100 = max.")
    for bk, bundle in active_bundles.items():
        tier_summary = bundle_tier_summary(bundle["stats"], stat_tiers)
        st.sidebar.markdown(f"**{bundle['label']}**")
        st.sidebar.markdown(f"<div class='bundle-desc'>{bundle['description']}<br><small>{tier_summary}</small></div>", unsafe_allow_html=True)
        if f"cb_bundle_{bk}" not in st.session_state: st.session_state[f"cb_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(bundle["label"], 0, 100, step=5, key=f"cb_bundle_{bk}", label_visibility="collapsed")
    for bk in BUNDLES:
        if bk not in bundle_weights: bundle_weights[bk] = 0
    effective_weights = compute_effective_weights(active_bundles, bundle_weights)
else:
    st.sidebar.caption("Direct control over every stat.")
    st.sidebar.markdown("<div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#888;margin-bottom:-0.5rem'><span>\u2190 Low priority</span><span>High priority \u2192</span></div>", unsafe_allow_html=True)
    all_enabled_stats = sorted([z for z, t in stat_tiers.items() if t in new_enabled], key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))
    for z_col in all_enabled_stats:
        tier = stat_tiers.get(z_col, 2); label = stat_labels.get(z_col, z_col); meth = stat_methodology.get(z_col, {}); help_parts = []
        if meth.get("what"): help_parts.append(f"What: {meth['what']}")
        if meth.get("how"): help_parts.append(f"How: {meth['how']}")
        if meth.get("limits"): help_parts.append(f"Limits: {meth['limits']}")
        w = st.sidebar.slider(f"{tier_badge(tier)} {label}", 0, 100, 50, 5, key=f"adv_cb_{z_col}", help="\n\n".join(help_parts) if help_parts else None)
        if w > 0: effective_weights[z_col] = w
    bundle_weights = {bk: 0 for bk in BUNDLES}

st.markdown("### Who's in the pool?")
st.caption("All Lions CBs with 200+ defensive snaps in a season. Z-scores computed against league-wide CBs. Man coverage data available for 2023-2024 only.")
cbs = df.copy()
if len(cbs) == 0: st.warning("No CBs found."); st.stop()

cbs = score_players(cbs, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0: st.info("All weights are zero — drag some sliders.")
cbs = cbs.sort_values("score", ascending=False).reset_index(drop=True)
cbs.index = cbs.index + 1

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Ranking")
ranked = cbs.copy()

if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("player_name", "—"); top_team = top.get("team", ""); top_score = top["score"]
    top_seasons = top.get("seasons", 0); badge = sample_size_badge(top_seasons)
    sign = "+" if top_score >= 0 else ""; team_part = f" ({top_team})" if top_team else ""
    st.markdown(f"<div style='background:#0076B6;color:white;padding:14px 20px;border-radius:8px;margin-bottom:8px;font-size:1.1rem;'><span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span> &nbsp;·&nbsp; <strong>{top_name}</strong>{team_part} {badge} &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{sign}{top_score:.2f}</span> <span style='opacity:0.85;'>({score_label(top_score)})</span></div>", unsafe_allow_html=True)
    warn = sample_size_caption(top_seasons)
    if warn: st.warning(warn)

st.caption("⚠️ CBs with fewer seasons have noisier scores. 🔴 = 1 season, 🟡 = 2-3 seasons.")
display_df = pd.DataFrame({"Rank": ranked.index, "": ranked["seasons"].apply(sample_size_badge), "Player": ranked["player_name"], "Team": ranked.get("team", "—"), "Seasons": ranked["seasons"].fillna(0).astype(int), "Games": ranked.get("total_games", pd.Series([0]*len(ranked))).fillna(0).astype(int), "Snaps": ranked.get("total_snaps", pd.Series([0]*len(ranked))).fillna(0).astype(int), "Score": ranked["score"].apply(format_score)})
st.dataframe(display_df, use_container_width=True, hide_index=True)
with st.expander("ℹ️ How is this score calculated?"): st.markdown(SCORE_EXPLAINER)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")
selected = st.selectbox("Pick a CB to see their breakdown", options=ranked["player_name"].tolist(), index=0)
player = ranked[ranked["player_name"] == selected].iloc[0]
warn = sample_size_caption(player.get("seasons", 0))
if warn: st.warning(warn)

c1, c2 = st.columns([1, 1])
with c1:
    team = player.get("team", "") if pd.notna(player.get("team")) else ""
    man_snaps = int(player.get("man_coverage_snaps") or 0)
    man_note = f" · {man_snaps} man coverage snaps" if man_snaps > 0 else ""
    st.markdown(f"### {selected}")
    st.caption(f"**{team}** · {int(player.get('seasons') or 0)} seasons · {int(player.get('total_games') or 0)} games · {int(player.get('total_snaps') or 0)} snaps{man_note}")
    st.markdown(f"**Your score:** {format_score(player['score'])}")
    st.markdown("---"); st.markdown("**How your score breaks down**")
    if not advanced_mode:
        bundle_rows = []
        for bk, bundle in active_bundles.items():
            bw = bundle_weights.get(bk, 0)
            if bw == 0: continue
            contribution = sum(player.get(z, 0) * (bw * internal / total_weight) for z, internal in bundle["stats"].items() if pd.notna(player.get(z)) and total_weight > 0)
            bundle_rows.append({"Bundle": bundle["label"], "Your weight": f"{bw}", "Contribution": f"{contribution:+.2f}"})
        if bundle_rows: st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True, hide_index=True)
        else: st.caption("No bundles weighted.")
        with st.expander("🔬 See the underlying stats"):
            stat_rows = []; shown = set()
            for bundle in active_bundles.values(): shown.update(bundle["stats"].keys())
            for z_col in sorted(shown, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                raw_col = RAW_COL_MAP.get(z_col); z = player.get(z_col); raw = player.get(raw_col) if raw_col else None
                stat_rows.append({"Tier": tier_badge(stat_tiers.get(z_col, 2)), "Stat": stat_labels.get(z_col, z_col), "Raw": f"{raw:.3f}" if pd.notna(raw) else "—", "Z-score": f"{z:+.2f}" if pd.notna(z) else "—"})
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Stat-by-stat breakdown")
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            raw_col = RAW_COL_MAP.get(z_col); z = player.get(z_col); raw = player.get(raw_col) if raw_col else None
            w = effective_weights.get(z_col, 0); contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            rows.append({"Tier": tier_badge(stat_tiers.get(z_col, 2)), "Stat": stat_labels.get(z_col, z_col), "Raw": f"{raw:.3f}" if pd.notna(raw) else "—", "Z-score": f"{z:+.2f}" if pd.notna(z) else "—", "Weight": f"{w}", "Contribution": f"{contrib:+.2f}"})
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with c2:
    st.markdown("**CB profile** (percentiles vs. league CBs)")
    fig = build_radar_figure(player, stat_labels, stat_methodology)
    if fig: st.plotly_chart(fig, use_container_width=True)
    else: st.caption("No radar data available.")
    st.caption("Each axis shows where this CB ranks among all qualified CBs league-wide. 50 = median. Man coverage stats only appear if Tier 4 is enabled and data exists.")

community_section(position_group=POSITION_GROUP, bundles=BUNDLES, bundle_weights=bundle_weights, advanced_mode=advanced_mode, page_url=PAGE_URL)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Data via [nflverse](https://github.com/nflverse) • Play-by-play via nflfastR • Snap counts via PFR • Man coverage data via FTN Data via nflverse (CC-BY-SA 4.0) • 2016-2024 regular season • Fan project, not affiliated with the NFL or Detroit Lions.")

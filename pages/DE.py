"""
Lions DE Rater — 2024 season
"""
import json
from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from lib_shared import apply_algo_weights, community_section, compute_effective_weights, get_algorithm_by_slug, inject_css, score_players

st.set_page_config(page_title="Lions DE Rater", page_icon="🦁", layout="wide", initial_sidebar_state="expanded")
inject_css()

POSITION_GROUP = "de"
PAGE_URL = "https://lions-rater.streamlit.app/DE"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "master_lions_des_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "de_stat_metadata.json"

@st.cache_data
def load_data(): return pl.read_parquet(DATA_PATH).to_pandas()
@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists(): return {}
    with open(METADATA_PATH) as f: return json.load(f)

RAW_COL_MAP = {
    "sacks_per_game_z": "sacks_per_game", "qb_hits_per_game_z": "qb_hits_per_game",
    "pressure_rate_z": "pressure_rate", "tfl_per_game_z": "tfl_per_game",
    "solo_tackle_rate_z": "solo_tackle_rate", "tackles_per_snap_z": "tackles_per_snap",
    "forced_fumbles_per_game_z": "forced_fumbles_per_game",
    "passes_defended_per_game_z": "passes_defended_per_game",
    "interceptions_per_game_z": "interceptions_per_game",
}

BUNDLES = {
    "pass_rush": {"label": "🔥 Pass rush", "description": "Gets to the QB. Sacks, hits, and total pressures.", "stats": {"sacks_per_game_z": 0.35, "qb_hits_per_game_z": 0.30, "pressure_rate_z": 0.35}},
    "run_defense": {"label": "🛡️ Run defense", "description": "Sets the edge and makes plays in the backfield.", "stats": {"tfl_per_game_z": 0.40, "solo_tackle_rate_z": 0.30, "tackles_per_snap_z": 0.30}},
    "playmaking": {"label": "💥 Playmaking", "description": "Forces fumbles, bats passes, and picks off throws.", "stats": {"forced_fumbles_per_game_z": 0.40, "passes_defended_per_game_z": 0.35, "interceptions_per_game_z": 0.25}},
}
DEFAULT_BUNDLE_WEIGHTS = {"pass_rush": 60, "run_defense": 40, "playmaking": 30}

RADAR_STATS = list(RAW_COL_MAP.keys())
RADAR_INVERT = set()
RADAR_LABEL_OVERRIDES = {"sacks_per_game_z": "Sacks", "qb_hits_per_game_z": "QB hits", "pressure_rate_z": "Pressure", "tfl_per_game_z": "TFLs", "solo_tackle_rate_z": "Solo tackle %", "tackles_per_snap_z": "Tackles/snap", "forced_fumbles_per_game_z": "Forced fumbles", "passes_defended_per_game_z": "Pass defense", "interceptions_per_game_z": "Interceptions"}

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
TIER_DESCRIPTIONS = {1: "Pure recorded facts.", 2: "Counts divided by opportunity.", 3: "Compared against a modeled baseline.", 4: "Inferred from patterns. Use with skepticism."}
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
def sample_size_badge(snaps):
    if pd.isna(snaps): return ""
    if snaps < 300: return "🔴"
    if snaps < 500: return "🟡"
    return ""
def sample_size_caption(snaps):
    if pd.isna(snaps): return ""
    if snaps < 300: return f"⚠️ Only {int(snaps)} defensive snaps. Small sample — treat as directional only."
    if snaps < 500: return f"⚠️ {int(snaps)} defensive snaps. Moderate sample."
    return ""

SCORE_EXPLAINER = """
**What this number means.** Weighted average of z-scores — 0 is league-average DE, +1 is one SD above, −1 is one SD below.

**How to read it:** `+1.0` or higher → well above average • `+0.4` to `+1.0` → above average • `−0.4` to `+0.4` → roughly average • `−1.0` or lower → well below average

**DE population:** 2024 regular season, z-scored against all DEs league-wide with 200+ defensive snaps.
"""

if "de_loaded_algo" not in st.session_state: st.session_state.de_loaded_algo = None
if "upvoted_ids" not in st.session_state: st.session_state.upvoted_ids = set()
if "de_tiers_enabled" not in st.session_state: st.session_state.de_tiers_enabled = [1, 2]

st.title("🦁 Lions DE Rater")
st.markdown("**Build your own algorithm.** Drag the sliders to weight what you value, and watch the Lions defensive ends re-rank in real time.")
st.caption("2024 regular season • Z-scores vs all DEs league-wide (200+ snaps)")

try: df = load_data()
except FileNotFoundError: st.error(f"Couldn't find DE data at {DATA_PATH}."); st.stop()

meta = load_metadata()
stat_tiers = meta.get("stat_tiers", {}); stat_labels = meta.get("stat_labels", {}); stat_methodology = meta.get("stat_methodology", {})

if "algo" in st.query_params and st.session_state.de_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP: apply_algo_weights(linked, BUNDLES); st.rerun()

st.sidebar.header("Filters")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle("🔬 Advanced mode", value=False)
st.sidebar.header("What do you value?")

if st.session_state.de_loaded_algo:
    la = st.session_state.de_loaded_algo
    st.sidebar.info(f"Loaded: **{la['name']}** by {la['author']}\n\n_{la.get('description', '')}_")
    if st.sidebar.button("Clear loaded algorithm"): st.session_state.de_loaded_algo = None

st.markdown("### How speculative do you want to get?")
tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        checked = st.checkbox(f"{tier_badge(tier)} {TIER_LABELS[tier]}", value=(tier in st.session_state.de_tiers_enabled), help=TIER_DESCRIPTIONS[tier], key=f"de_tier_checkbox_{tier}")
        if checked: new_enabled.append(tier)
st.session_state.de_tiers_enabled = new_enabled
if not new_enabled: st.warning("Enable at least one tier."); st.stop()
active_bundles = filter_bundles_by_tier(BUNDLES, stat_tiers, new_enabled)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

bundle_weights = {}; effective_weights = {}
if not advanced_mode:
    if not active_bundles: st.info("No bundles in enabled tiers."); st.stop()
    st.sidebar.caption("Drag to weight what matters to you.")
    for bk, bundle in active_bundles.items():
        tier_summary = bundle_tier_summary(bundle["stats"], stat_tiers)
        st.sidebar.markdown(f"**{bundle['label']}**")
        st.sidebar.markdown(f"<div class='bundle-desc'>{bundle['description']}<br><small>{tier_summary}</small></div>", unsafe_allow_html=True)
        if f"de_bundle_{bk}" not in st.session_state: st.session_state[f"de_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(bundle["label"], 0, 100, step=5, key=f"de_bundle_{bk}", label_visibility="collapsed")
    for bk in BUNDLES:
        if bk not in bundle_weights: bundle_weights[bk] = 0
    effective_weights = compute_effective_weights(active_bundles, bundle_weights)
else:
    st.sidebar.caption("Direct control over every stat.")
    all_enabled_stats = sorted([z for z, t in stat_tiers.items() if t in new_enabled], key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))
    for z_col in all_enabled_stats:
        tier = stat_tiers.get(z_col, 2); label = stat_labels.get(z_col, z_col); meth = stat_methodology.get(z_col, {}); help_parts = []
        if meth.get("what"): help_parts.append(f"What: {meth['what']}")
        if meth.get("how"): help_parts.append(f"How: {meth['how']}")
        if meth.get("limits"): help_parts.append(f"Limits: {meth['limits']}")
        w = st.sidebar.slider(f"{tier_badge(tier)} {label}", 0, 100, 50, 5, key=f"adv_de_{z_col}", help="\n\n".join(help_parts) if help_parts else None)
        if w > 0: effective_weights[z_col] = w
    bundle_weights = {bk: 0 for bk in BUNDLES}

st.markdown("### Who's in the pool?")
st.caption("All Lions DEs with 25+ defensive snaps in 2024. Z-scores computed against all DEs league-wide (200+ snaps).")
des = df.copy()
if len(des) == 0: st.warning("No DEs found."); st.stop()
des = score_players(des, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0: st.info("All weights are zero — drag some sliders.")
des = des.sort_values("score", ascending=False).reset_index(drop=True)
des.index = des.index + 1

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Ranking")
ranked = des.copy()

if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("player_name", "—"); top_score = top["score"]
    top_snaps = top.get("def_snaps", 0); badge = sample_size_badge(top_snaps)
    sign = "+" if top_score >= 0 else ""
    st.markdown(f"<div style='background:#0076B6;color:white;padding:14px 20px;border-radius:8px;margin-bottom:8px;font-size:1.1rem;'><span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span> &nbsp;·&nbsp; <strong>{top_name}</strong> {badge} &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{sign}{top_score:.2f}</span> <span style='opacity:0.85;'>({score_label(top_score)})</span></div>", unsafe_allow_html=True)
    warn = sample_size_caption(top_snaps)
    if warn: st.warning(warn)

st.caption("🔴 = <300 snaps, 🟡 = 300-499 snaps.")
display_df = pd.DataFrame({
    "Rank": ranked.index, "": ranked.get("def_snaps", pd.Series([0]*len(ranked))).apply(sample_size_badge),
    "Player": ranked["player_name"],
    "Games": ranked.get("games", pd.Series([0]*len(ranked))).fillna(0).astype(int),
    "Snaps": ranked.get("def_snaps", pd.Series([0]*len(ranked))).fillna(0).astype(int),
    "Score": ranked["score"].apply(format_score),
})
st.dataframe(display_df, use_container_width=True, hide_index=True)
with st.expander("ℹ️ How is this score calculated?"): st.markdown(SCORE_EXPLAINER)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")
selected = st.selectbox("Pick a DE to see their breakdown", options=ranked["player_name"].tolist(), index=0)
player = ranked[ranked["player_name"] == selected].iloc[0]
warn = sample_size_caption(player.get("def_snaps", 0))
if warn: st.warning(warn)

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f"### {selected}")
    st.caption(f"{int(player.get('games') or 0)} games · {int(player.get('def_snaps') or 0)} snaps")
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
        with st.expander("🔬 See the underlying stats"):
            stat_rows = []; shown = set()
            for bundle in active_bundles.values(): shown.update(bundle["stats"].keys())
            for z_col in sorted(shown, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                raw_col = RAW_COL_MAP.get(z_col); z = player.get(z_col); raw = player.get(raw_col) if raw_col else None
                stat_rows.append({"Tier": tier_badge(stat_tiers.get(z_col, 2)), "Stat": stat_labels.get(z_col, z_col), "Raw": f"{raw:.3f}" if pd.notna(raw) else "—", "Z-score": f"{z:+.2f}" if pd.notna(z) else "—"})
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            raw_col = RAW_COL_MAP.get(z_col); z = player.get(z_col); raw = player.get(raw_col) if raw_col else None
            w = effective_weights.get(z_col, 0); contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            rows.append({"Tier": tier_badge(stat_tiers.get(z_col, 2)), "Stat": stat_labels.get(z_col, z_col), "Raw": f"{raw:.3f}" if pd.notna(raw) else "—", "Z-score": f"{z:+.2f}" if pd.notna(z) else "—", "Weight": f"{w}", "Contribution": f"{contrib:+.2f}"})
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with c2:
    st.markdown("**DE profile** (percentiles vs. league DEs)")
    fig = build_radar_figure(player, stat_labels, stat_methodology)
    if fig: st.plotly_chart(fig, use_container_width=True)
    else: st.caption("No radar data available.")

community_section(position_group=POSITION_GROUP, bundles=BUNDLES, bundle_weights=bundle_weights, advanced_mode=advanced_mode, page_url=PAGE_URL)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Data via [nflverse](https://github.com/nflverse) • 2024 regular season • Z-scored against 90 DEs with 200+ snaps • Fan project, not affiliated with the NFL or Detroit Lions.")

"""
Lions Punter Rater — 2024 season
"""
import json
from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from team_selector import get_team_and_season, filter_by_team_and_season, NFL_TEAMS
from lib_shared import apply_algo_weights, community_section, compute_effective_weights, get_algorithm_by_slug, inject_css, score_players

st.set_page_config(page_title="Lions Punter Rater", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")
inject_css()

# ── Team & Season selector ────────────────────────────────────
selected_team, selected_season = get_team_and_season()
team_name = NFL_TEAMS.get(selected_team, selected_team)

POSITION_GROUP = "punter"
PAGE_URL = "https://lions-rater.streamlit.app/Punter"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "league_p_all_seasons.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "punter_stat_metadata.json"

@st.cache_data
def load_punter_data(): return pl.read_parquet(DATA_PATH).to_pandas()
@st.cache_data
def load_punter_metadata():
    if not METADATA_PATH.exists(): return {}
    with open(METADATA_PATH) as f: return json.load(f)

RAW_COL_MAP = {
    "avg_distance_z": "avg_distance", "avg_net_z": "avg_net",
    "inside_20_rate_z": "inside_20_rate", "touchback_rate_z": "touchback_rate",
    "fair_catch_rate_z": "fair_catch_rate", "pin_rate_z": "pin_rate",
    "punt_epa_z": "punt_epa",
}

BUNDLES = {
    "distance": {"label": "📏 Distance", "description": "Raw leg power. Gross and net punt yardage.", "why": "Think a big leg is the most important thing a punter has? Crank this up.", "stats": {"avg_distance_z": 0.50, "avg_net_z": 0.50}},
    "placement": {"label": "📍 Placement", "description": "Pins opponents deep. Inside-20 rate, avoids touchbacks, forces fair catches.", "why": "Value punters who flip the field with precision, not just power? Slide right.", "stats": {"inside_20_rate_z": 0.30, "touchback_rate_z": 0.20, "fair_catch_rate_z": 0.20, "pin_rate_z": 0.30}},
    "impact": {"label": "💥 Impact", "description": "Overall field position value measured by EPA.", "why": "Care about the bottom line — how much field position does he actually create? Slide right.", "stats": {"punt_epa_z": 1.00}},
}
DEFAULT_BUNDLE_WEIGHTS = {"distance": 40, "placement": 60, "impact": 40}

RADAR_STATS = list(RAW_COL_MAP.keys())
RADAR_INVERT = set()
RADAR_LABEL_OVERRIDES = {"avg_distance_z": "Distance", "avg_net_z": "Net yards", "inside_20_rate_z": "Inside 20", "touchback_rate_z": "No touchbacks", "fair_catch_rate_z": "Fair catches", "pin_rate_z": "Pin rate", "punt_epa_z": "Punt EPA"}

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

TIER_LABELS = {1: "Counting stats", 2: "Rate stats", 3: "Modeled stats", 4: "Estimated stats"}
TIER_DESCRIPTIONS = {1: "Raw totals — sacks, tackles, yards, touchdowns.", 2: "Per-game and per-snap averages that adjust for playing time.", 3: "Stats adjusted for expected performance based on a model.", 4: "Inferred from patterns."}
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

SCORE_EXPLAINER = """
**What this number means.** Weighted average of z-scores — 0 is league-average punter, +1 is one SD above, −1 is one SD below.

**How to read it:** `+1.0` or higher → well above average • `+0.4` to `+1.0` → above average • `−0.4` to `+0.4` → roughly average • `−1.0` or lower → well below average

**Punter population:** 2024 regular season, z-scored against all punters league-wide with 20+ punt attempts.

**Touchback rate is inverted** — fewer touchbacks = better, so positive z = good.

**Pin rate** combines inside-20 and downed punts — the total "pinned deep" rate.
"""

if "punter_loaded_algo" not in st.session_state: st.session_state.punter_loaded_algo = None
if "upvoted_ids" not in st.session_state: st.session_state.upvoted_ids = set()
if "punter_tiers_enabled" not in st.session_state: st.session_state.punter_tiers_enabled = [1, 2]

st.title("🦁 Lions Punter Rater")
st.markdown("What makes a great player? **You decide.** Drag the sliders to weight what you value.")
st.caption(f"{selected_season} regular season • Z-scores vs all punters league-wide (20+ attempts)")

try: df = load_punter_data()
except FileNotFoundError: st.error(f"Couldn't find punter data at {DATA_PATH}."); st.stop()

# Filter to selected team and season
df = filter_by_team_and_season(df, selected_team, selected_season, team_col="recent_team", season_col="season_year")
if len(df) == 0:
    st.warning(f"No {team_name} punters found for {selected_season}.")
    st.stop()

meta = load_punter_metadata()
stat_tiers = meta.get("stat_tiers", {}); stat_labels = meta.get("stat_labels", {}); stat_methodology = meta.get("stat_methodology", {})

st.markdown("### Which stats should count?")
tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        checked = st.checkbox(f"{tier_badge(tier)} {TIER_LABELS[tier]}", value=(tier in st.session_state.punter_tiers_enabled), help=TIER_DESCRIPTIONS[tier], key=f"punter_tier_checkbox_{tier}")
        if checked: new_enabled.append(tier)
st.session_state.punter_tiers_enabled = new_enabled
if not new_enabled: st.warning("Enable at least one tier."); st.stop()
active_bundles = filter_bundles_by_tier(BUNDLES, stat_tiers, new_enabled)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

bundle_weights = {}; effective_weights = {}
if not active_bundles: st.info("No bundles in enabled tiers."); st.stop()
st.sidebar.markdown("Each slider controls how much a skill affects the final score. Slide right to prioritize, left to ignore.")
for bk, bundle in active_bundles.items():
    tier_summary = bundle_tier_summary(bundle["stats"], stat_tiers)
    st.sidebar.markdown(f"**{bundle['label']}**")
    st.sidebar.markdown(f"<div class='bundle-desc'>{bundle['description']}<br><small>{tier_summary}</small></div>", unsafe_allow_html=True)
    if f"punter_bundle_{bk}" not in st.session_state: st.session_state[f"punter_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
    bundle_weights[bk] = st.sidebar.slider(bundle["label"], 0, 100, step=5, key=f"punter_bundle_{bk}", label_visibility="collapsed", help=bundle.get("why", ""))
    st.sidebar.caption(f"_↑ {bundle.get('why', '')}_")
for bk in BUNDLES:
    if bk not in bundle_weights: bundle_weights[bk] = 0
effective_weights = compute_effective_weights(active_bundles, bundle_weights)

punters = df.copy()
if len(punters) == 0: st.warning("No punters found."); st.stop()
punters = score_players(punters, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0: st.info("All weights are zero — drag some sliders.")
punters = punters.sort_values("score", ascending=False).reset_index(drop=True)
punters.index = punters.index + 1

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Ranking")
ranked = punters.copy()

if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("player_name", "—"); top_score = top["score"]
    sign = "+" if top_score >= 0 else ""
    st.markdown(f"<div style='background:#0076B6;color:white;padding:14px 20px;border-radius:8px;margin-bottom:8px;font-size:1.1rem;'><span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span> &nbsp;·&nbsp; <strong>{top_name}</strong> &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{sign}{top_score:.2f}</span> <span style='opacity:0.85;'>({format_percentile(zscore_to_percentile(top_score))})</span></div>", unsafe_allow_html=True)

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "Player": ranked["player_name"],
    "Games": ranked.get("games", pd.Series([0]*len(ranked))).fillna(0).astype(int),
    "Punts": ranked.get("punt_att", pd.Series([0]*len(ranked))).fillna(0).astype(int),
    "Avg dist": ranked.get("avg_distance", pd.Series([0]*len(ranked))).apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—"),
    "Net": ranked.get("avg_net", pd.Series([0]*len(ranked))).apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—"),
    "In 20 %": ranked.get("inside_20_rate", pd.Series([0]*len(ranked))).apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—"),
    "Your score": ranked["score"].apply(format_score),
})
st.dataframe(display_df, use_container_width=True, hide_index=True)
with st.expander("ℹ️ How is the score calculated?"): st.markdown(SCORE_EXPLAINER)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")
selected = st.selectbox("Pick a punter", options=ranked["player_name"].tolist(), index=0)
player = ranked[ranked["player_name"] == selected].iloc[0]

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f"### {selected}")
    st.caption(f"{int(player.get('games') or 0)} games · {int(player.get('punt_att') or 0)} punts · Avg {player.get('avg_distance', 0):.1f} yds · Net {player.get('avg_net', 0):.1f} yds")
    st.markdown(f"**Your score:** {format_score(player['score'])}")
    st.markdown("---"); st.markdown("**How your score breaks down**")
    bundle_rows = []
    for bk, bundle in active_bundles.items():
        bw = bundle_weights.get(bk, 0)
        if bw == 0: continue
        contribution = sum(player.get(z, 0) * (bw * internal / total_weight) for z, internal in bundle["stats"].items() if pd.notna(player.get(z)) and total_weight > 0)
        bundle_rows.append({"Skill": bundle["label"], "Your weight": f"{bw}", "Points added": f"{contribution:+.2f}"})
    if bundle_rows: st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True, hide_index=True)
    with st.expander("🔬 See the underlying stats"):
        stat_rows = []; shown = set()
        for bundle in active_bundles.values(): shown.update(bundle["stats"].keys())
        for z_col in sorted(shown, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            raw_col = RAW_COL_MAP.get(z_col); z = player.get(z_col); raw = player.get(raw_col) if raw_col else None
            if raw_col in ("inside_20_rate", "touchback_rate", "fair_catch_rate", "pin_rate"):
                raw_fmt = f"{raw:.1%}" if pd.notna(raw) else "—"
            else:
                raw_fmt = f"{raw:.1f}" if pd.notna(raw) else "—"
            stat_rows.append({"Tier": tier_badge(stat_tiers.get(z_col, 2)), "Stat": stat_labels.get(z_col, z_col), "Raw": raw_fmt, "Z-score": f"{z:+.2f}" if pd.notna(z) else "—"})
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

with c2:
    st.markdown("**Punter profile** (percentiles vs. league punters)")
    fig = build_radar_figure(player, stat_labels, stat_methodology)
    if fig: st.plotly_chart(fig, use_container_width=True)

community_section(position_group=POSITION_GROUP, bundles=BUNDLES, bundle_weights=bundle_weights, advanced_mode=False, page_url=PAGE_URL)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Data via [nflverse](https://github.com/nflverse) • 2024 regular season • Z-scored against 34 punters with 20+ attempts • Fan project, not affiliated with the NFL or Detroit Lions.")

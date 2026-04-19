"""
Lions Kicker Rater — 2024 season
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

st.set_page_config(page_title="Lions Kicker Rater", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")
inject_css()

# ── Team & Season selector ────────────────────────────────────
selected_team, selected_season = get_team_and_season()
team_name = NFL_TEAMS.get(selected_team, selected_team)

POSITION_GROUP = "kicker"
PAGE_URL = "https://lions-rater.streamlit.app/Kicker"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "league_k_all_seasons.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "kicker_stat_metadata.json"

@st.cache_data
def load_kicker_data(): return pl.read_parquet(DATA_PATH).to_pandas()
@st.cache_data
def load_kicker_metadata():
    if not METADATA_PATH.exists(): return {}
    with open(METADATA_PATH) as f: return json.load(f)

RAW_COL_MAP = {
    "fg_pct_z": "fg_pct", "fg_40_pct_z": "fg_40_pct",
    "fg_over_expected_z": "fg_over_expected", "fg_epa_z": "fg_epa",
    "clutch_pct_z": "clutch_pct", "clutch_epa_z": "clutch_epa",
    "xp_pct_z": "xp_pct",
}

BUNDLES = {
    "accuracy": {"label": "🎯 Accuracy", "description": "Makes field goals at a high rate, especially from distance. Beats difficulty expectations.", "why": "Think making kicks is all that matters? Crank this up.", "stats": {"fg_pct_z": 0.30, "fg_40_pct_z": 0.30, "fg_over_expected_z": 0.40}},
    "clutch": {"label": "🧊 Clutch", "description": "Delivers in pressure situations — 4th quarter or OT, game within 8 points.", "why": "Value kickers who hit game-winners under pressure? Slide right.", "stats": {"clutch_pct_z": 0.50, "clutch_epa_z": 0.50}},
    "consistency": {"label": "✅ Consistency", "description": "Reliable on extra points and overall EPA contribution.", "why": "Want a kicker who never costs you points on easy kicks? Slide right.", "stats": {"xp_pct_z": 0.40, "fg_epa_z": 0.60}},
}
DEFAULT_BUNDLE_WEIGHTS = {"accuracy": 60, "clutch": 50, "consistency": 30}

RADAR_STATS = list(RAW_COL_MAP.keys())
RADAR_INVERT = set()
RADAR_LABEL_OVERRIDES = {"fg_pct_z": "FG %", "fg_40_pct_z": "40+ yd %", "fg_over_expected_z": "Over expected", "fg_epa_z": "FG EPA", "clutch_pct_z": "Clutch %", "clutch_epa_z": "Clutch EPA", "xp_pct_z": "XP %"}

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
**What this number means.** Weighted average of z-scores — 0 is league-average kicker, +1 is one SD above, −1 is one SD below.

**How to read it:** `+1.0` or higher → well above average • `+0.4` to `+1.0` → above average • `−0.4` to `+0.4` → roughly average • `−1.0` or lower → well below average

**Kicker population:** 2024 regular season, z-scored against all kickers league-wide with 10+ FG attempts.

**Clutch metric:** 4th quarter or OT, game within 8 points. Most kickers only get 5-10 clutch attempts per season — treat as directional.

**FG over expected:** Uses nflverse's pre-snap field goal probability model. Positive = making kicks harder than expected.
"""

if "kicker_loaded_algo" not in st.session_state: st.session_state.kicker_loaded_algo = None
if "upvoted_ids" not in st.session_state: st.session_state.upvoted_ids = set()
if "kicker_tiers_enabled" not in st.session_state: st.session_state.kicker_tiers_enabled = [1, 2, 3]

st.title("🦁 Lions Kicker Rater")
st.markdown("What makes a great player? **You decide.** Drag the sliders to weight what you value.")
st.caption(f"{selected_season} regular season • Z-scores vs all kickers league-wide (10+ FG attempts)")

try: df = load_kicker_data()
except FileNotFoundError: st.error(f"Couldn't find kicker data at {DATA_PATH}."); st.stop()

# Filter to selected team and season
df = filter_by_team_and_season(df, selected_team, selected_season, team_col="recent_team", season_col="season_year")
if len(df) == 0:
    st.warning(f"No {team_name} kickers found for {selected_season}.")
    st.stop()

meta = load_kicker_metadata()
stat_tiers = meta.get("stat_tiers", {}); stat_labels = meta.get("stat_labels", {}); stat_methodology = meta.get("stat_methodology", {})

st.markdown("### Which stats should count?")
tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        checked = st.checkbox(f"{tier_badge(tier)} {TIER_LABELS[tier]}", value=(tier in st.session_state.kicker_tiers_enabled), help=TIER_DESCRIPTIONS[tier], key=f"kicker_tier_checkbox_{tier}")
        if checked: new_enabled.append(tier)
st.session_state.kicker_tiers_enabled = new_enabled
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
    if f"kicker_bundle_{bk}" not in st.session_state: st.session_state[f"kicker_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
    bundle_weights[bk] = st.sidebar.slider(bundle["label"], 0, 100, step=5, key=f"kicker_bundle_{bk}", label_visibility="collapsed", help=bundle.get("why", ""))
    st.sidebar.caption(f"_↑ {bundle.get('why', '')}_")
for bk in BUNDLES:
    if bk not in bundle_weights: bundle_weights[bk] = 0
effective_weights = compute_effective_weights(active_bundles, bundle_weights)

kickers = df.copy()
if len(kickers) == 0: st.warning("No kickers found."); st.stop()
kickers = score_players(kickers, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0: st.info("All weights are zero — drag some sliders.")
kickers = kickers.sort_values("score", ascending=False).reset_index(drop=True)
kickers.index = kickers.index + 1

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Ranking")
ranked = kickers.copy()

if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("player_name", "—"); top_score = top["score"]
    sign = "+" if top_score >= 0 else ""
    st.markdown(f"<div style='background:#0076B6;color:white;padding:14px 20px;border-radius:8px;margin-bottom:8px;font-size:1.1rem;'><span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span> &nbsp;·&nbsp; <strong>{top_name}</strong> &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{sign}{top_score:.2f}</span> <span style='opacity:0.85;'>({format_percentile(zscore_to_percentile(top_score))})</span></div>", unsafe_allow_html=True)

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "Player": ranked["player_name"],
    "Games": ranked.get("games", pd.Series([0]*len(ranked))).fillna(0).astype(int),
    "FG": ranked.apply(lambda r: f"{int(r.get('fg_made', 0))}/{int(r.get('fg_att', 0))}", axis=1),
    "FG %": ranked.get("fg_pct", pd.Series([0]*len(ranked))).apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—"),
    "Clutch": ranked.apply(lambda r: f"{int(r.get('clutch_att', 0))} att, {r.get('clutch_pct', 0):.0%}" if pd.notna(r.get('clutch_pct')) else f"{int(r.get('clutch_att', 0))} att", axis=1),
    "Your score": ranked["score"].apply(format_score),
})
st.dataframe(display_df, use_container_width=True, hide_index=True)
with st.expander("ℹ️ How is the score calculated?"): st.markdown(SCORE_EXPLAINER)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")
selected = st.selectbox("Pick a kicker", options=ranked["player_name"].tolist(), index=0)
player = ranked[ranked["player_name"] == selected].iloc[0]

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f"### {selected}")
    st.caption(f"{int(player.get('games') or 0)} games · {int(player.get('fg_made') or 0)}/{int(player.get('fg_att') or 0)} FG · Long: {int(player.get('long_fg') or 0)} yds · {int(player.get('clutch_att') or 0)} clutch attempts")
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
            if raw_col in ("fg_pct", "fg_40_pct", "clutch_pct", "xp_pct"):
                raw_fmt = f"{raw:.1%}" if pd.notna(raw) else "—"
            else:
                raw_fmt = f"{raw:.3f}" if pd.notna(raw) else "—"
            stat_rows.append({"Tier": tier_badge(stat_tiers.get(z_col, 2)), "Stat": stat_labels.get(z_col, z_col), "Raw": raw_fmt, "Z-score": f"{z:+.2f}" if pd.notna(z) else "—"})
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

with c2:
    st.markdown("**Kicker profile** (percentiles vs. league kickers)")
    fig = build_radar_figure(player, stat_labels, stat_methodology)
    if fig: st.plotly_chart(fig, use_container_width=True)

community_section(position_group=POSITION_GROUP, bundles=BUNDLES, bundle_weights=bundle_weights, advanced_mode=False, page_url=PAGE_URL)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Data via [nflverse](https://github.com/nflverse) • 2024 regular season • Z-scored against 38 kickers with 10+ FG attempts • Fan project, not affiliated with the NFL or Detroit Lions.")

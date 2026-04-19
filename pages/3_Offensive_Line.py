"""
Lions Offensive Line Rater v2
===============================
2024 season data, z-scored against all 2024 starting OL league-wide.
Position-specific run attribution: tackles get outside/end-gap runs,
guards get interior gaps, center gets middle runs.
8 stats across 3 bundles.
"""

import json
from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from lib_shared import apply_algo_weights, community_section, compute_effective_weights, get_algorithm_by_slug, inject_css, score_players

st.set_page_config(page_title="Lions OL Rater", page_icon="🦁", layout="wide", initial_sidebar_state="expanded")
inject_css()

POSITION_GROUP = "ol"
PAGE_URL = "https://lions-rater.streamlit.app/Offensive_Line"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "master_lions_ol_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ol_stat_metadata.json"

@st.cache_data
def load_ol_data(): return pl.read_parquet(DATA_PATH).to_pandas()
@st.cache_data
def load_ol_metadata():
    if not METADATA_PATH.exists(): return {}
    with open(METADATA_PATH) as f: return json.load(f)

RAW_COL_MAP = {
    "pos_run_epa_z": "pos_run_epa",
    "pos_run_success_z": "pos_run_success",
    "pos_run_explosive_z": "pos_run_explosive",
    "team_sack_rate_z": "team_sack_rate",
    "team_pressure_rate_z": "team_pressure_rate",
    "penalty_rate_z": "penalty_rate",
    "penalty_epa_per_game_z": "penalty_epa_per_game",
    "snap_share_z": "snap_share",
    "sg_run_epa_z": "sg_run_epa",
    "sg_run_success_z": "sg_run_success",
    "sg_run_explosive_z": "sg_run_explosive",
    "uc_run_epa_z": "uc_run_epa",
    "uc_run_success_z": "uc_run_success",
    "uc_run_explosive_z": "uc_run_explosive",
}

BUNDLES = {
    "run_blocking": {
        "label": "🏃 Run blocking (overall)",
        "description": "Overall run game through this lineman's position-specific gaps. Scrambles excluded. Tackles on outside/end runs, guards on interior, center on middle.",
        "stats": {"pos_run_epa_z": 0.45, "pos_run_success_z": 0.35, "pos_run_explosive_z": 0.20},
    },
    "run_shotgun": {
        "label": "🔫 Run blocking — shotgun",
        "description": "Outside zone, spread runs, and stretch plays from shotgun formation. Rewards lateral athleticism and reach blocking.",
        "stats": {"sg_run_epa_z": 0.45, "sg_run_success_z": 0.35, "sg_run_explosive_z": 0.20},
    },
    "run_under_center": {
        "label": "🏋️ Run blocking — under center",
        "description": "Power, counter, and gap scheme runs from under center. Rewards drive blocking and combo blocks.",
        "stats": {"uc_run_epa_z": 0.45, "uc_run_success_z": 0.35, "uc_run_explosive_z": 0.20},
    },
    "pass_protection": {
        "label": "🛡️ Pass protection",
        "description": "Team sack rate and pressure rate. Lower = better. All 5 OL contribute, but elite linemen correlate with low team pressure.",
        "stats": {"team_sack_rate_z": 0.55, "team_pressure_rate_z": 0.45},
    },
    "discipline": {
        "label": "⚖️ Discipline & durability",
        "description": "Avoids penalties, minimizes penalty damage, and stays on the field.",
        "stats": {"penalty_rate_z": 0.30, "penalty_epa_per_game_z": 0.30, "snap_share_z": 0.40},
    },
}
DEFAULT_BUNDLE_WEIGHTS = {"run_blocking": 70, "run_shotgun": 0, "run_under_center": 0, "pass_protection": 50, "discipline": 30}

RADAR_STATS = ["pos_run_epa_z", "pos_run_explosive_z", "sg_run_epa_z", "sg_run_explosive_z", "uc_run_epa_z", "uc_run_explosive_z", "team_sack_rate_z", "team_pressure_rate_z", "penalty_rate_z", "snap_share_z"]
RADAR_INVERT = set()
RADAR_LABEL_OVERRIDES = {
    "pos_run_epa_z": "Run EPA", "pos_run_explosive_z": "Explosive runs",
    "sg_run_epa_z": "Shotgun EPA", "sg_run_explosive_z": "Shotgun explosive",
    "uc_run_epa_z": "Under center EPA", "uc_run_explosive_z": "UC explosive",
    "team_sack_rate_z": "Sack prevention", "team_pressure_rate_z": "Pressure prevention",
    "penalty_rate_z": "Discipline", "snap_share_z": "Durability",
}

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
TIER_DESCRIPTIONS = {1: "Raw totals — sacks, tackles, yards, touchdowns.", 2: "Per-game and per-snap averages that adjust for playing time.", 3: "Stats adjusted for expected performance based on a model.", 4: "Inferred from limited data — least reliable. Use with caution."}
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

def format_percentile(pct):
    if pct is None or pd.isna(pct): return "—"
    if pct >= 99: return "top 1%"
    if pct >= 50: return f"top {100 - int(pct)}%"
    return f"bottom {int(pct)}%"

def format_score(score):
    if pd.isna(score): return "—"
    sign = "+" if score >= 0 else ""
    pct = zscore_to_percentile(score)
    return f"{sign}{score:.2f} ({format_percentile(pct)})"

def sample_size_badge(snaps):
    if pd.isna(snaps): return ""
    if snaps < 400: return "🔴"
    if snaps < 800: return "🟡"
    return ""

def sample_size_caption(snaps):
    if pd.isna(snaps): return ""
    if snaps < 400: return f"⚠️ Only {int(snaps)} snaps. Small sample — treat as directional only."
    if snaps < 800: return f"⚠️ {int(snaps)} snaps. Moderate sample — scores may be noisy."
    return ""

POSITION_LABELS = {"LT": "Left Tackle", "LG": "Left Guard", "C": "Center", "RG": "Right Guard", "RT": "Right Tackle"}

SCORE_EXPLAINER = """
**What this number means.** Weighted average of z-scores — 0 is league-average starting OL, +1 is one SD above, −1 is one SD below.

**How to read it:** `+1.0` or higher → well above average • `+0.4` to `+1.0` → above average • `−0.4` to `+0.4` → roughly average • `−1.0` or lower → well below average

**Position-specific run blocking.** Unlike most OL tools, this page attributes runs to the correct lineman based on gap:
- **Tackles** are measured on outside/end-gap runs (sweeps, outside zone, tosses) — where their reach blocking and athleticism matter most.
- **Guards** are measured on interior guard + tackle gap runs — their combo blocks and short pulls.
- **Center** is measured on middle runs plus guard-gap runs from both sides.

**Formation splits.** Run blocking is further split by shotgun vs under center:
- **Shotgun** captures outside zone, stretch, and spread run schemes — rewards lateral athleticism and reach blocking.
- **Under center** captures power, counter, and gap schemes — rewards drive blocking and physicality.
This reveals linemen who excel in one scheme but struggle in another (e.g., a tackle who dominates power runs but struggles in outside zone).

**Data cleaning.** QB scrambles are excluded from all run stats — they inflate tackle numbers without reflecting blocking quality.

**Position-group z-scores.** Run stats are z-scored within position group (tackles vs tackles, guards vs guards, centers vs centers) so interior linemen aren't penalized for naturally lower explosive rates on inside runs.

**Pass protection limitation.** Sack rate and pressure rate are team-level — we can't isolate which lineman allowed the pressure from free data. But elite OL correlate with low team pressure rates.

**Inverted stats.** Sack rate, pressure rate, penalty rate, and penalty EPA are inverted so positive z always = good.
"""

if "ol_loaded_algo" not in st.session_state: st.session_state.ol_loaded_algo = None
if "upvoted_ids" not in st.session_state: st.session_state.upvoted_ids = set()
if "ol_tiers_enabled" not in st.session_state: st.session_state.ol_tiers_enabled = [1, 2]

st.title("🦁 Lions Offensive Line Rater")
st.markdown("What makes a great player? **You decide.** Drag the sliders to weight what you value, and watch the Lions starting five re-rank in real time. _No 'best lineman' — just **your** best lineman._")
st.caption("2024 regular season • Z-scores vs all 153 starting OL league-wide • Position-specific run gap attribution")

try: df = load_ol_data()
except FileNotFoundError: st.error(f"Couldn't find OL data at {DATA_PATH}."); st.stop()

meta = load_ol_metadata()
stat_tiers = meta.get("stat_tiers", {}); stat_labels = meta.get("stat_labels", {}); stat_methodology = meta.get("stat_methodology", {})

if "algo" in st.query_params and st.session_state.ol_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP: apply_algo_weights(linked, BUNDLES); st.rerun()

st.sidebar.header("What matters to you?")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle("🔬 Advanced mode", value=False, help="Show individual stat sliders.")
st.sidebar.markdown("Each slider controls how much a skill affects the final score. Slide right to prioritize, left to ignore.")

if st.session_state.ol_loaded_algo:
    la = st.session_state.ol_loaded_algo
    st.sidebar.info(f"Loaded: **{la['name']}** by {la['author']}\n\n_{la.get('description', '')}_")
    if st.sidebar.button("Clear loaded algorithm"): st.session_state.ol_loaded_algo = None

st.markdown("### Which stats should count?")
st.caption("Check more boxes to include more types of stats. More boxes = more data, but less certainty.")
tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        checked = st.checkbox(f"{tier_badge(tier)} {TIER_LABELS[tier]}", value=(tier in st.session_state.ol_tiers_enabled), help=TIER_DESCRIPTIONS[tier], key=f"ol_tier_checkbox_{tier}")
        if checked: new_enabled.append(tier)
st.session_state.ol_tiers_enabled = new_enabled
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
        if f"ol_bundle_{bk}" not in st.session_state: st.session_state[f"ol_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(bundle["label"], 0, 100, step=5, key=f"ol_bundle_{bk}", label_visibility="collapsed")
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
        w = st.sidebar.slider(f"{tier_badge(tier)} {label}", 0, 100, 50, 5, key=f"adv_ol_{z_col}", help="\n\n".join(help_parts) if help_parts else None)
        if w > 0: effective_weights[z_col] = w
    bundle_weights = {bk: 0 for bk in BUNDLES}


st.caption("Lions 2024 starting offensive linemen. Z-scores computed against all 153 qualified starting OL league-wide. Position-specific gap attribution for run blocking.")
ol = df.copy()
if len(ol) == 0: st.warning("No OL found."); st.stop()

ol = score_players(ol, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0: st.info("All weights are zero — drag some sliders.")
ol = ol.sort_values("score", ascending=False).reset_index(drop=True)
ol.index = ol.index + 1

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Ranking")
ranked = ol.copy()

if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("full_name", "—"); top_pos = POSITION_LABELS.get(top.get("depth_position", ""), "")
    top_score = top["score"]; top_snaps = top.get("off_snaps", 0)
    badge = sample_size_badge(top_snaps); sign = "+" if top_score >= 0 else ""
    pos_part = f" ({top_pos})" if top_pos else ""
    st.markdown(f"<div style='background:#0076B6;color:white;padding:14px 20px;border-radius:8px;margin-bottom:8px;font-size:1.1rem;'><span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span> &nbsp;·&nbsp; <strong>{top_name}</strong>{pos_part} {badge} &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{sign}{top_score:.2f}</span> <span style='opacity:0.85;'>({format_percentile(zscore_to_percentile(top_score))})</span></div>", unsafe_allow_html=True)
    warn = sample_size_caption(top_snaps)
    if warn: st.warning(warn)

st.caption("⚠️ Run blocking stats are position-specific: tackles measured on outside runs, guards on interior, center on middle. Pass protection is team-level.")

display_df = pd.DataFrame({
    "Rank": ranked.index, "": ranked.get("off_snaps", pd.Series([0]*len(ranked))).apply(sample_size_badge),
    "Player": ranked["full_name"],
    "Pos": ranked["depth_position"].map(POSITION_LABELS),
    "Games": ranked.get("games_played", pd.Series([0]*len(ranked))).fillna(0).astype(int),
    "Snaps": ranked.get("off_snaps", pd.Series([0]*len(ranked))).fillna(0).astype(int),
    "Penalties": ranked.get("penalties_total", pd.Series([0]*len(ranked))).fillna(0).astype(int),
    "Your score": ranked["score"].apply(format_score),
})
st.dataframe(display_df, use_container_width=True, hide_index=True)
with st.expander("ℹ️ How is the score calculated?"): st.markdown(SCORE_EXPLAINER)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")
selected = st.selectbox("Pick a lineman to see their breakdown", options=ranked["full_name"].tolist(), index=0)
player = ranked[ranked["full_name"] == selected].iloc[0]
warn = sample_size_caption(player.get("off_snaps", 0))
if warn: st.warning(warn)

c1, c2 = st.columns([1, 1])
with c1:
    pos = POSITION_LABELS.get(player.get("depth_position", ""), "")
    run_plays = int(player.get("pos_run_plays") or 0)
    gap_type = "outside/end-gap" if player.get("depth_position") in ("LT", "RT") else "interior gap" if player.get("depth_position") in ("LG", "RG") else "middle"
    st.markdown(f"### {selected}")
    st.caption(f"**{pos}** · {int(player.get('games_played') or 0)} games · {int(player.get('off_snaps') or 0)} snaps · {run_plays} {gap_type} runs · {int(player.get('penalties_total') or 0)} penalties")
    st.markdown(f"**Your score:** {format_score(player['score'])}")
    st.markdown("---"); st.markdown("**How your score breaks down**")
    if not advanced_mode:
        bundle_rows = []
        for bk, bundle in active_bundles.items():
            bw = bundle_weights.get(bk, 0)
            if bw == 0: continue
            contribution = sum(player.get(z, 0) * (bw * internal / total_weight) for z, internal in bundle["stats"].items() if pd.notna(player.get(z)) and total_weight > 0)
            bundle_rows.append({"Skill": bundle["label"], "Your weight": f"{bw}", "Points added": f"{contribution:+.2f}"})
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
            rows.append({"Tier": tier_badge(stat_tiers.get(z_col, 2)), "Stat": stat_labels.get(z_col, z_col), "Raw": f"{raw:.3f}" if pd.notna(raw) else "—", "Z-score": f"{z:+.2f}" if pd.notna(z) else "—", "Weight": f"{w}", "Points added": f"{contrib:+.2f}"})
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with c2:
    st.markdown("**OL profile** (percentiles vs. league starters)")
    fig = build_radar_figure(player, stat_labels, stat_methodology)
    if fig: st.plotly_chart(fig, use_container_width=True)
    else: st.caption("No radar data available.")
    st.caption("Each axis shows where this lineman ranks among all 153 qualified starting OL league-wide. 50 = median. Inverted stats (sacks, penalties) are flipped so higher = better on all axes.")

community_section(position_group=POSITION_GROUP, bundles=BUNDLES, bundle_weights=bundle_weights, advanced_mode=advanced_mode, page_url=PAGE_URL)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Data via [nflverse](https://github.com/nflverse) • Depth charts via ESPN • Play-by-play via nflfastR • Snap counts via PFR • 2024 regular season • Position-specific gap attribution • Fan project, not affiliated with the NFL or Detroit Lions.")

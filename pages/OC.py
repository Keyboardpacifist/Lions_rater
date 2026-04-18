"""
Lions OC Rater — Offensive Coordinators
Career default with 2024-only toggle. League-wide.
"""
import json
from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from lib_shared import apply_algo_weights, community_section, compute_effective_weights, get_algorithm_by_slug, inject_css, score_players

st.set_page_config(page_title="NFL OC Rater", page_icon="🦁", layout="wide", initial_sidebar_state="expanded")
inject_css()

POSITION_GROUP = "oc"
PAGE_URL = "https://lions-rater.streamlit.app/OC"
DATA_PATH_CAREER = Path(__file__).resolve().parent.parent / "data" / "master_ocs_with_z.parquet"
DATA_PATH_2024 = Path(__file__).resolve().parent.parent / "data" / "master_ocs_2024_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "oc_stat_metadata.json"

@st.cache_data
def load_oc_career(): return pl.read_parquet(DATA_PATH_CAREER).to_pandas()
@st.cache_data
def load_oc_2024(): return pl.read_parquet(DATA_PATH_2024).to_pandas()
@st.cache_data
def load_oc_metadata():
    if not METADATA_PATH.exists(): return {}
    with open(METADATA_PATH) as f: return json.load(f)

RAW_COL_MAP = {
    "epa_per_play_z": "epa_per_play", "pass_epa_per_play_z": "pass_epa_per_play",
    "rush_epa_per_play_z": "rush_epa_per_play", "success_rate_z": "success_rate",
    "explosive_pass_rate_z": "explosive_pass_rate", "explosive_rush_rate_z": "explosive_rush_rate",
    "third_down_rate_z": "third_down_rate", "red_zone_td_rate_z": "red_zone_td_rate",
    "win_pct_z": "win_pct",
    "off_cap_pct_z": "off_cap_pct", "off_draft_capital_z": "off_draft_capital",
}

BUNDLES = {
    "efficiency": {"label": "📊 Offensive efficiency", "description": "Overall EPA per play, passing and rushing EPA. The core measure of offensive production.", "stats": {"epa_per_play_z": 0.40, "pass_epa_per_play_z": 0.30, "rush_epa_per_play_z": 0.30}},
    "execution": {"label": "🎯 Situational execution", "description": "Third down conversions and red zone TD rate. Measures playcalling in critical moments.", "stats": {"third_down_rate_z": 0.50, "red_zone_td_rate_z": 0.50}},
    "explosiveness": {"label": "💥 Big play ability", "description": "Explosive pass plays (20+ yds) and explosive rush plays (10+ yds). Creates game-breaking moments.", "stats": {"explosive_pass_rate_z": 0.55, "explosive_rush_rate_z": 0.45}},
    "investment": {"label": "💰 Roster investment", "description": "Salary cap % and draft capital invested in offense. Higher = more resources given to this coordinator.", "stats": {"off_cap_pct_z": 0.50, "off_draft_capital_z": 0.50}},
    "winning": {"label": "🏆 Winning", "description": "Team win percentage during coordinator tenure. The ultimate measure, but the least isolatable.", "stats": {"win_pct_z": 1.00}},
}
DEFAULT_BUNDLE_WEIGHTS = {"efficiency": 60, "execution": 50, "explosiveness": 40, "investment": 0, "winning": 30}

RADAR_STATS = ["epa_per_play_z", "pass_epa_per_play_z", "rush_epa_per_play_z", "success_rate_z", "explosive_pass_rate_z", "explosive_rush_rate_z", "third_down_rate_z", "red_zone_td_rate_z", "win_pct_z", "off_cap_pct_z", "off_draft_capital_z"]
RADAR_INVERT = set()
RADAR_LABEL_OVERRIDES = {"epa_per_play_z": "Off EPA", "pass_epa_per_play_z": "Pass EPA", "rush_epa_per_play_z": "Rush EPA", "success_rate_z": "Success rate", "explosive_pass_rate_z": "Explosive pass", "explosive_rush_rate_z": "Explosive rush", "third_down_rate_z": "3rd down", "red_zone_td_rate_z": "Red zone", "win_pct_z": "Win %", "off_cap_pct_z": "Cap invested", "off_draft_capital_z": "Draft capital"}

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
TIER_DESCRIPTIONS = {1: "Pure recorded facts.", 2: "Counts divided by opportunity.", 3: "Compared against a modeled baseline.", 4: "Inferred from patterns."}
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

SCORE_EXPLAINER = """
**What this number means.** Weighted average of z-scores — 0 is league-average OC, +1 is one SD above, −1 is one SD below.

**How to read it:** `+1.0` or higher → well above average • `+0.4` to `+1.0` → above average • `−0.4` to `+0.4` → roughly average • `−1.0` or lower → well below average

**Talent caveat.** These stats measure the offense's output, not the coordinator in isolation. A great QB inflates OC numbers. A bad offensive line deflates them. The scores reflect the whole unit's performance under this coordinator's playcalling.

**Career vs 2024.** Career mode averages across all seasons (2016-2024). 2024-only shows single-season performance. Career is more stable; 2024 is more current.
"""

if "oc_loaded_algo" not in st.session_state: st.session_state.oc_loaded_algo = None
if "upvoted_ids" not in st.session_state: st.session_state.upvoted_ids = set()
if "oc_tiers_enabled" not in st.session_state: st.session_state.oc_tiers_enabled = [1, 2]

st.title("🦁 NFL Offensive Coordinator Rater")
st.markdown("**Build your own algorithm.** Drag the sliders to weight what you value, and watch NFL offensive coordinators re-rank in real time.")

# Career vs 2024 toggle
view_mode = st.radio("View mode", ["Career (2016-2024)", "2024 season only"], horizontal=True, index=0)
is_career = view_mode.startswith("Career")

if is_career:
    st.caption("Career averages 2016-2024 • Z-scores vs all OCs with 2+ seasons • League-wide")
else:
    st.caption("2024 regular season only • Z-scores vs all 2024 OCs • League-wide")

try:
    if is_career:
        df = load_oc_career()
    else:
        df = load_oc_2024()
except FileNotFoundError:
    st.error("Couldn't find OC data."); st.stop()

meta = load_oc_metadata()
stat_tiers = meta.get("stat_tiers", {}); stat_labels = meta.get("stat_labels", {}); stat_methodology = meta.get("stat_methodology", {})

st.sidebar.header("What do you value?")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle("🔬 Advanced mode", value=False)

st.markdown("### How speculative do you want to get?")
tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        checked = st.checkbox(f"{tier_badge(tier)} {TIER_LABELS[tier]}", value=(tier in st.session_state.oc_tiers_enabled), help=TIER_DESCRIPTIONS[tier], key=f"oc_tier_checkbox_{tier}")
        if checked: new_enabled.append(tier)
st.session_state.oc_tiers_enabled = new_enabled
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
        if f"oc_bundle_{bk}" not in st.session_state: st.session_state[f"oc_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(bundle["label"], 0, 100, step=5, key=f"oc_bundle_{bk}", label_visibility="collapsed")
    for bk in BUNDLES:
        if bk not in bundle_weights: bundle_weights[bk] = 0
    effective_weights = compute_effective_weights(active_bundles, bundle_weights)
else:
    st.sidebar.caption("Direct control over every stat.")
    all_enabled_stats = sorted([z for z, t in stat_tiers.items() if t in new_enabled], key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))
    for z_col in all_enabled_stats:
        tier = stat_tiers.get(z_col, 2); label = stat_labels.get(z_col, z_col); meth = stat_methodology.get(z_col, {}); help_parts = []
        if meth.get("what"): help_parts.append(f"What: {meth['what']}")
        if meth.get("limits"): help_parts.append(f"Limits: {meth['limits']}")
        w = st.sidebar.slider(f"{tier_badge(tier)} {label}", 0, 100, 50, 5, key=f"adv_oc_{z_col}", help="\n\n".join(help_parts) if help_parts else None)
        if w > 0: effective_weights[z_col] = w
    bundle_weights = {bk: 0 for bk in BUNDLES}

ocs = df.copy()
if len(ocs) == 0: st.warning("No OCs found."); st.stop()
ocs = score_players(ocs, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0: st.info("All weights are zero — drag some sliders.")
ocs = ocs.sort_values("score", ascending=False).reset_index(drop=True)
ocs.index = ocs.index + 1

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Ranking")
ranked = ocs.copy()

if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("coordinator", "—"); top_score = top["score"]
    top_teams = top.get("teams", top.get("team", ""))
    sign = "+" if top_score >= 0 else ""
    seasons_val = top.get("seasons", 1)
    badge = sample_size_badge(seasons_val) if is_career else ""
    st.markdown(f"<div style='background:#0076B6;color:white;padding:14px 20px;border-radius:8px;margin-bottom:8px;font-size:1.1rem;'><span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span> &nbsp;·&nbsp; <strong>{top_name}</strong> ({top_teams}) {badge} &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{sign}{top_score:.2f}</span> <span style='opacity:0.85;'>({score_label(top_score)})</span></div>", unsafe_allow_html=True)

if is_career:
    display_df = pd.DataFrame({
        "Rank": ranked.index,
        "": ranked.get("seasons", pd.Series([1]*len(ranked))).apply(sample_size_badge),
        "Coordinator": ranked.get("coordinator", ranked.get("player_name", "—")),
        "Teams": ranked.get("teams", ranked.get("team", "—")),
        "Seasons": ranked.get("seasons", pd.Series([1]*len(ranked))).fillna(1).astype(int),
        "W-L": ranked.apply(lambda r: f"{int(r.get('total_wins', 0))}-{int(r.get('total_losses', 0))}", axis=1),
        "EPA/play": ranked.get("epa_per_play", pd.Series([0]*len(ranked))).apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "—"),
        "Score": ranked["score"].apply(format_score),
    })
else:
    display_df = pd.DataFrame({
        "Rank": ranked.index,
        "Coordinator": ranked.get("coordinator", ranked.get("player_name", "—")),
        "Team": ranked.get("team", "—"),
        "EPA/play": ranked.get("epa_per_play", pd.Series([0]*len(ranked))).apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "—"),
        "3rd down": ranked.get("third_down_rate", pd.Series([0]*len(ranked))).apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—"),
        "Red zone": ranked.get("red_zone_td_rate", pd.Series([0]*len(ranked))).apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—"),
        "Score": ranked["score"].apply(format_score),
    })

st.dataframe(display_df, use_container_width=True, hide_index=True)
with st.expander("ℹ️ How is this score calculated?"): st.markdown(SCORE_EXPLAINER)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Coordinator detail")
coord_col = "coordinator" if "coordinator" in ranked.columns else "player_name"
selected = st.selectbox("Pick a coordinator", options=ranked[coord_col].tolist(), index=0)
player = ranked[ranked[coord_col] == selected].iloc[0]

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f"### {selected}")
    teams = player.get("teams", player.get("team", ""))
    if is_career:
        st.caption(f"**{teams}** · {int(player.get('seasons', 1))} seasons · {int(player.get('total_wins', 0))}-{int(player.get('total_losses', 0))} record")
    else:
        st.caption(f"**{teams}** · 2024 season")
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
                if raw_col in ("success_rate", "third_down_rate", "red_zone_td_rate", "win_pct", "explosive_pass_rate", "explosive_rush_rate"):
                    raw_fmt = f"{raw:.1%}" if pd.notna(raw) else "—"
                else:
                    raw_fmt = f"{raw:+.4f}" if pd.notna(raw) else "—"
                stat_rows.append({"Tier": tier_badge(stat_tiers.get(z_col, 2)), "Stat": stat_labels.get(z_col, z_col), "Raw": raw_fmt, "Z-score": f"{z:+.2f}" if pd.notna(z) else "—"})
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            raw_col = RAW_COL_MAP.get(z_col); z = player.get(z_col); raw = player.get(raw_col) if raw_col else None
            w = effective_weights.get(z_col, 0); contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            if raw_col in ("success_rate", "third_down_rate", "red_zone_td_rate", "win_pct", "explosive_pass_rate", "explosive_rush_rate"):
                raw_fmt = f"{raw:.1%}" if pd.notna(raw) else "—"
            else:
                raw_fmt = f"{raw:+.4f}" if pd.notna(raw) else "—"
            rows.append({"Tier": tier_badge(stat_tiers.get(z_col, 2)), "Stat": stat_labels.get(z_col, z_col), "Raw": raw_fmt, "Z-score": f"{z:+.2f}" if pd.notna(z) else "—", "Weight": f"{w}", "Contribution": f"{contrib:+.2f}"})
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with c2:
    st.markdown("**OC profile** (percentiles vs. league OCs)")
    fig = build_radar_figure(player, stat_labels, stat_methodology)
    if fig: st.plotly_chart(fig, use_container_width=True)

community_section(position_group=POSITION_GROUP, bundles=BUNDLES, bundle_weights=bundle_weights, advanced_mode=advanced_mode, page_url=PAGE_URL)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Data via [nflverse](https://github.com/nflverse) • 2016-2024 regular seasons • Coordinator tenures manually compiled • ⚠️ Stats reflect the entire offensive unit, not the coordinator in isolation • Fan project, not affiliated with the NFL.")

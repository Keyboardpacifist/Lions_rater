"""
Lions QB Rater — 2024 season
League-wide, 2024 single-season. Monkey-proofed UI.
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

st.set_page_config(page_title="QB Rater", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")
inject_css()

# ── Team & Season selector ────────────────────────────────────
selected_team, selected_season = get_team_and_season()
team_name = NFL_TEAMS.get(selected_team, selected_team)

POSITION_GROUP = "qb"
PAGE_URL = "https://lions-rater.streamlit.app/QB"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "league_qb_all_seasons.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "qb_stat_metadata.json"

@st.cache_data
def load_qb_data(): return pl.read_parquet(DATA_PATH).to_pandas()
@st.cache_data
def load_qb_metadata():
    if not METADATA_PATH.exists(): return {}
    with open(METADATA_PATH) as f: return json.load(f)

RAW_COL_MAP = {
    "pass_epa_per_play_z": "pass_epa_per_play", "yards_per_attempt_z": "yards_per_attempt",
    "td_rate_z": "td_rate", "int_rate_z": "int_rate",
    "completion_pct_z": "completion_pct", "passing_cpoe_z": "passing_cpoe",
    "sack_rate_z": "sack_rate", "first_down_rate_z": "first_down_rate",
    "air_yards_per_attempt_z": "air_yards_per_attempt", "yac_per_completion_z": "yac_per_completion",
    "turnover_rate_z": "turnover_rate",
    "rush_yards_per_game_z": "rush_yards_per_game", "rush_epa_per_carry_z": "rush_epa_per_carry",
    "passing_yards_per_game_z": "passing_yards_per_game", "passing_tds_per_game_z": "passing_tds_per_game",
}

BUNDLES = {
    "efficiency": {
        "label": "📊 Passing efficiency",
        "description": "How much value does he create per throw? EPA, yards per attempt, and TD rate.",
        "why": "Think pure passing efficiency is what separates elite QBs? Crank this up.",
        "stats": {"pass_epa_per_play_z": 0.45, "yards_per_attempt_z": 0.30, "td_rate_z": 0.25},
    },
    "accuracy": {
        "label": "🎯 Accuracy & precision",
        "description": "Does he put the ball where it needs to go? Completion %, CPOE, and first down rate.",
        "why": "Value QBs who are surgically accurate? Slide this right.",
        "stats": {"completion_pct_z": 0.25, "passing_cpoe_z": 0.40, "first_down_rate_z": 0.35},
    },
    "ball_security": {
        "label": "🛡️ Ball security",
        "description": "Does he protect the football? INT rate, sack rate, and overall turnover rate.",
        "why": "Think the best ability is availability of the football? Slide right.",
        "stats": {"int_rate_z": 0.35, "sack_rate_z": 0.30, "turnover_rate_z": 0.35},
    },
    "downfield": {
        "label": "🔥 Downfield passing",
        "description": "Can he push the ball vertically? Air yards and YAC per completion.",
        "why": "Want a QB who can stretch the field and create explosive plays? Slide right.",
        "stats": {"air_yards_per_attempt_z": 0.55, "yac_per_completion_z": 0.45},
    },
    "mobility": {
        "label": "🏃 Rushing & mobility",
        "description": "Is he a threat with his legs? Rush yards and rush EPA.",
        "why": "Value dual-threat QBs who can hurt you on the ground? Slide right.",
        "stats": {"rush_yards_per_game_z": 0.50, "rush_epa_per_carry_z": 0.50},
    },
}
DEFAULT_BUNDLE_WEIGHTS = {"efficiency": 60, "accuracy": 50, "ball_security": 40, "downfield": 30, "mobility": 20}

RADAR_STATS = [
    "pass_epa_per_play_z", "yards_per_attempt_z", "completion_pct_z",
    "passing_cpoe_z", "td_rate_z", "int_rate_z", "sack_rate_z",
    "air_yards_per_attempt_z", "rush_yards_per_game_z",
]
RADAR_INVERT = set()
RADAR_LABEL_OVERRIDES = {
    "pass_epa_per_play_z": "Pass EPA", "yards_per_attempt_z": "Yds/att",
    "completion_pct_z": "Comp %", "passing_cpoe_z": "CPOE",
    "td_rate_z": "TD rate", "int_rate_z": "Ball security",
    "sack_rate_z": "Sack avoidance", "air_yards_per_attempt_z": "Air yards",
    "rush_yards_per_game_z": "Rushing",
}

# ── Score formatting ──────────────────────────────────────────
def zscore_to_percentile(z):
    if pd.isna(z): return None
    return float(norm.cdf(z) * 100)

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

def sample_size_warning(att):
    if pd.isna(att): return ""
    if att < 200: return f"⚠️ Only {int(att)} attempts — small sample, treat with caution"
    if att < 350: return f"⚠️ {int(att)} attempts — moderate sample"
    return ""

# ── Tier system ───────────────────────────────────────────────
TIER_LABELS = {1: "Counting stats", 2: "Rate stats", 3: "Modeled stats", 4: "Estimated stats"}
TIER_DESCRIPTIONS = {
    1: "Yards per game, TDs per game — raw production totals.",
    2: "Per-attempt rates like EPA/play, completion %, TD rate.",
    3: "Stats adjusted for expected performance — like CPOE (completion over expected).",
    4: "Inferred from limited data — least reliable.",
}
def tier_badge(tier): return {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠"}.get(tier, "⚪")

def filter_bundles_by_tier(bundles, stat_tiers, enabled_tiers):
    filtered = {}
    for bk, bdef in bundles.items():
        kept = {z: w for z, w in bdef["stats"].items() if stat_tiers.get(z, 2) in enabled_tiers}
        if kept:
            filtered[bk] = {k: v for k, v in bdef.items()}
            filtered[bk]["stats"] = kept
    return filtered

def bundle_tier_summary(bundle_stats, stat_tiers):
    counts = {}
    for z in bundle_stats: t = stat_tiers.get(z, 2); counts[t] = counts.get(t, 0) + 1
    return " ".join(f"{tier_badge(t)}×{c}" for t, c in sorted(counts.items()))

# ── Radar chart ───────────────────────────────────────────────
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
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=axes + [axes[0]],
        customdata=descriptions + [descriptions[0]],
        fill="toself", fillcolor="rgba(31, 119, 180, 0.25)",
        line=dict(color="rgba(31, 119, 180, 0.9)", width=2),
        marker=dict(size=6, color="rgba(31, 119, 180, 1)"),
        hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<br><br><i>%{customdata}</i><extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[25, 50, 75, 100],
                            ticktext=["25th", "50th", "75th", "100th"],
                            tickfont=dict(size=9, color="#888"), gridcolor="#ddd"),
            angularaxis=dict(tickfont=dict(size=11), gridcolor="#ddd"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False, margin=dict(l=60, r=60, t=20, b=20),
        height=380, paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── Session state ─────────────────────────────────────────────
if "qb_loaded_algo" not in st.session_state: st.session_state.qb_loaded_algo = None
if "upvoted_ids" not in st.session_state: st.session_state.upvoted_ids = set()
if "qb_tiers_enabled" not in st.session_state: st.session_state.qb_tiers_enabled = [1, 2]

try: df = load_qb_data()
except FileNotFoundError: st.error(f"Couldn't find QB data at {DATA_PATH}."); st.stop()

# Filter to selected team and season
df = filter_by_team_and_season(df, selected_team, selected_season, team_col="recent_team", season_col="season_year")
if len(df) == 0:
    st.warning(f"No {team_name} quarterbacks found for {selected_season}.")
    st.stop()

meta = load_qb_metadata()
stat_tiers = meta.get("stat_tiers", {}); stat_labels = meta.get("stat_labels", {}); stat_methodology = meta.get("stat_methodology", {})

if "algo" in st.query_params and st.session_state.qb_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP:
        apply_algo_weights(linked, BUNDLES); st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════
st.title(f"🏈 {team_name} quarterbacks")
st.markdown("What makes a great QB? **You decide.** Use the sliders on the left to tell us what you value most, and the rankings update instantly.")
st.caption(f"{selected_season} regular season · Compared to all 39 QBs league-wide with 200+ pass attempts")

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.header("What matters to you?")
st.sidebar.markdown("Each slider controls how much a skill affects the final score. Slide right to prioritize it, or all the way left to ignore it.")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

if st.session_state.qb_loaded_algo:
    la = st.session_state.qb_loaded_algo
    st.sidebar.info(f"Loaded: **{la['name']}** by {la['author']}\n\n_{la.get('description', '')}_")
    if st.sidebar.button("Clear loaded algorithm"): st.session_state.qb_loaded_algo = None

# ══════════════════════════════════════════════════════════════
# STAT TYPE CHECKBOXES
# ══════════════════════════════════════════════════════════════
st.markdown("### Which stats should count?")
st.markdown("Check more boxes to include more types of stats. More boxes = more data, but less certainty.")
available_tiers = set(stat_tiers.values()) if stat_tiers else {1, 2}
tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        has_stats = tier in available_tiers
        if has_stats:
            checked = st.checkbox(
                f"{tier_badge(tier)} {TIER_LABELS[tier]}",
                value=(tier in st.session_state.qb_tiers_enabled),
                help=TIER_DESCRIPTIONS[tier],
                key=f"qb_tier_checkbox_{tier}",
            )
            if checked: new_enabled.append(tier)
        else:
            st.markdown(f"<span style='opacity:0.35'>{tier_badge(tier)} {TIER_LABELS[tier]}</span>", unsafe_allow_html=True)
            st.caption("No stats available")
st.session_state.qb_tiers_enabled = new_enabled
if not new_enabled: st.warning("Check at least one box above to include some stats."); st.stop()
active_bundles = filter_bundles_by_tier(BUNDLES, stat_tiers, new_enabled)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR SLIDERS
# ══════════════════════════════════════════════════════════════
advanced_mode = False
bundle_weights = {}
effective_weights = {}

if not active_bundles: st.info("No stat bundles available for the selected stat types."); st.stop()

for bk, bundle in active_bundles.items():
    st.sidebar.markdown(f"**{bundle['label']}**")
    st.sidebar.markdown(f"{bundle['description']}")
    if f"qb_bundle_{bk}" not in st.session_state:
        st.session_state[f"qb_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
    bundle_weights[bk] = st.sidebar.slider(
        bundle["label"], 0, 100, step=5,
        key=f"qb_bundle_{bk}", label_visibility="collapsed",
        help=bundle.get("why", ""),
    )
    st.sidebar.caption(f"_↑ {bundle.get('why', '')}_")

for bk in BUNDLES:
    if bk not in bundle_weights: bundle_weights[bk] = 0
effective_weights = compute_effective_weights(active_bundles, bundle_weights)

with st.sidebar.expander("Want more control? Adjust individual stats"):
    advanced_mode = st.checkbox("Enable individual stat control", value=False, key="qb_advanced_toggle")
    if advanced_mode:
        st.caption("Set the weight of each individual stat. This overrides the bundle sliders above.")
        effective_weights = {}
        all_enabled_stats = sorted([z for z, t in stat_tiers.items() if t in new_enabled], key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))
        for z_col in all_enabled_stats:
            label = stat_labels.get(z_col, z_col)
            meth = stat_methodology.get(z_col, {})
            help_text = meth.get("what", "")
            if meth.get("limits"): help_text += f"\n\nLimits: {meth['limits']}"
            w = st.slider(f"{tier_badge(stat_tiers.get(z_col, 2))} {label}", 0, 100, 50, 5, key=f"adv_qb_{z_col}", help=help_text if help_text else None)
            if w > 0: effective_weights[z_col] = w
        bundle_weights = {bk: 0 for bk in BUNDLES}

# ══════════════════════════════════════════════════════════════
# FILTER & SCORE
# ══════════════════════════════════════════════════════════════
min_attempts = st.slider("Minimum pass attempts", 0, 600, 200, step=25, help="Filter out QBs with too few attempts. 200 = roughly half a season of starts.")
qbs = df[df["attempts"].fillna(0) >= min_attempts].copy()

if len(qbs) == 0: st.warning("No QBs match the current filter."); st.stop()
qbs = score_players(qbs, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0: st.info("All sliders are at zero — slide at least one to the right to see rankings.")
qbs = qbs.sort_values("score", ascending=False).reset_index(drop=True)
qbs.index = qbs.index + 1

# ══════════════════════════════════════════════════════════════
# RANKING TABLE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
ranked = qbs.copy()

st.markdown("**How to read the score:** 0.00 = league average QB. The percentile shows where this QB ranks among all qualifying QBs.")

if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("player_display_name", "—"); top_score = top["score"]
    top_team = top.get("recent_team", "")
    top_pct = format_percentile(zscore_to_percentile(top_score))
    sign = "+" if top_score >= 0 else ""
    st.markdown(
        f"<div style='background:#0076B6;color:white;padding:14px 20px;border-radius:8px;"
        f"margin-bottom:8px;font-size:1.1rem;'>"
        f"<span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span>"
        f" &nbsp;·&nbsp; <strong>{top_name}</strong> ({top_team})"
        f" &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>"
        f"{sign}{top_score:.2f}</span>"
        f" <span style='opacity:0.85;'>({top_pct})</span></div>",
        unsafe_allow_html=True,
    )
    warn = sample_size_warning(top.get("attempts", 0))
    if warn: st.warning(warn)

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "Player": ranked["player_display_name"],
    "Team": ranked.get("recent_team", "—"),
    "Games": ranked.get("games", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "Att": ranked.get("attempts", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "Yds": ranked.get("passing_yards", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "TD": ranked.get("passing_tds", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "INT": ranked.get("passing_interceptions", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "EPA/play": ranked.get("pass_epa_per_play", pd.Series([0] * len(ranked))).apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "—"),
    "Your score": ranked["score"].apply(format_score),
})
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# PLAYER DETAIL
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
selected = st.selectbox("Pick a QB to see their full breakdown", options=ranked["player_display_name"].tolist(), index=0)
player = ranked[ranked["player_display_name"] == selected].iloc[0]

warn = sample_size_warning(player.get("attempts", 0))
if warn: st.warning(warn)

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f"### {selected}")
    team = player.get("recent_team", "")
    st.caption(
        f"**{team}** · {int(player.get('games') or 0)} games · "
        f"{int(player.get('attempts') or 0)} att · "
        f"{int(player.get('passing_yards') or 0)} yds · "
        f"{int(player.get('passing_tds') or 0)} TD / {int(player.get('passing_interceptions') or 0)} INT"
    )

    player_score = player["score"]
    player_pct = format_percentile(zscore_to_percentile(player_score))
    sign = "+" if player_score >= 0 else ""
    st.markdown(f"**Your score: {sign}{player_score:.2f} ({player_pct})**")
    st.markdown("_This score is based on your slider settings. Change the sliders and this number changes._")

    st.markdown("---")
    st.markdown("**Where the score comes from**")
    st.markdown("Each row shows how much one skill contributed to the total, based on your slider weights.")

    if not advanced_mode:
        bundle_rows = []
        for bk, bundle in active_bundles.items():
            bw = bundle_weights.get(bk, 0)
            if bw == 0: continue
            contribution = sum(
                player.get(z, 0) * (bw * internal / total_weight)
                for z, internal in bundle["stats"].items()
                if pd.notna(player.get(z)) and total_weight > 0
            )
            bundle_rows.append({"Skill": bundle["label"], "Your weight": f"{bw}", "Points added": f"{contribution:+.2f}"})
        if bundle_rows:
            st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True, hide_index=True)

        with st.expander("See the individual stats behind each skill"):
            stat_rows = []; shown = set()
            for bundle in active_bundles.values(): shown.update(bundle["stats"].keys())
            for z_col in sorted(shown, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                raw_col = RAW_COL_MAP.get(z_col)
                z = player.get(z_col); raw = player.get(raw_col) if raw_col else None
                pct = zscore_to_percentile(z) if pd.notna(z) else None
                if raw_col in ("completion_pct", "td_rate", "int_rate", "sack_rate", "first_down_rate", "turnover_rate"):
                    raw_fmt = f"{raw:.1%}" if pd.notna(raw) else "—"
                elif raw_col in ("passing_cpoe",):
                    raw_fmt = f"{raw:+.2f}" if pd.notna(raw) else "—"
                else:
                    raw_fmt = f"{raw:.2f}" if pd.notna(raw) else "—"
                stat_rows.append({"Stat": stat_labels.get(z_col, z_col), "Value": raw_fmt, "Percentile": f"{int(pct)}th" if pct is not None else "—"})
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            raw_col = RAW_COL_MAP.get(z_col)
            z = player.get(z_col); raw = player.get(raw_col) if raw_col else None
            w = effective_weights.get(z_col, 0)
            contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            pct = zscore_to_percentile(z) if pd.notna(z) else None
            if raw_col in ("completion_pct", "td_rate", "int_rate", "sack_rate", "first_down_rate", "turnover_rate"):
                raw_fmt = f"{raw:.1%}" if pd.notna(raw) else "—"
            elif raw_col in ("passing_cpoe",):
                raw_fmt = f"{raw:+.2f}" if pd.notna(raw) else "—"
            else:
                raw_fmt = f"{raw:.2f}" if pd.notna(raw) else "—"
            rows.append({"Stat": stat_labels.get(z_col, z_col), "Value": raw_fmt, "Percentile": f"{int(pct)}th" if pct is not None else "—", "Weight": f"{w}", "Points added": f"{contrib:+.2f}"})
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with c2:
    st.markdown("**Percentile profile vs. all league QBs**")
    st.caption("50th = league average. Higher = better. INT rate and sack rate are inverted (higher = fewer turnovers/sacks).")
    fig = build_radar_figure(player, stat_labels, stat_methodology)
    if fig: st.plotly_chart(fig, use_container_width=True)

community_section(position_group=POSITION_GROUP, bundles=BUNDLES, bundle_weights=bundle_weights, advanced_mode=advanced_mode, page_url=PAGE_URL)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    "Data via [nflverse](https://github.com/nflverse) · 2024 regular season · "
    "Z-scored against 39 QBs with 200+ pass attempts · "
    "Fan project, not affiliated with the NFL or Detroit Lions."
)

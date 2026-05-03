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
from team_selector import get_team_and_season, filter_by_team_and_season, NFL_TEAMS, display_abbr
from career_arc import career_arc_section
from lib_shared import apply_algo_weights, community_section, compute_effective_weights, get_algorithm_by_slug, inject_css, metric_picker, radar_season_row, render_combine_chart, render_master_detail_leaderboard, render_player_card, render_player_year_picker, score_players
from lib_top_nav import render_home_button
import lib_gas_panels as gp

st.set_page_config(page_title="QB Rater", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")
inject_css()

render_home_button()  # ← back to landing
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

# Per-stat raw value formatting for the radar benchmark hover.
_RADAR_RAW_FORMATTERS = {
    "pass_epa_per_play_z": ("EPA/play", lambda v: f"{v:+.2f}"),
    "yards_per_attempt_z": ("yds/att", lambda v: f"{v:.2f}"),
    "completion_pct_z": ("comp %", lambda v: f"{v*100:.1f}%"),
    "passing_cpoe_z": ("CPOE", lambda v: f"{v:+.1f}"),
    "td_rate_z": ("TD rate", lambda v: f"{v*100:.1f}%"),
    "int_rate_z": ("INT rate", lambda v: f"{v*100:.2f}%"),
    "sack_rate_z": ("sack rate", lambda v: f"{v*100:.1f}%"),
    "air_yards_per_attempt_z": ("air yds/att", lambda v: f"{v:.2f}"),
    "rush_yards_per_game_z": ("rush yds/g", lambda v: f"{v:.1f}"),
}

def _format_radar_raw(z_col, raw_value):
    if raw_value is None or pd.isna(raw_value):
        return ""
    spec = _RADAR_RAW_FORMATTERS.get(z_col)
    if spec is None:
        return f"{raw_value:.2f}"
    label, fmt = spec
    return f"{label}: {fmt(raw_value)}"


# ── Radar chart ───────────────────────────────────────────────
def build_radar_figure(player, stat_labels, stat_methodology,
                        benchmark=None, benchmark_raw=None,
                        benchmark_label="Top 32 starter avg"):
    axes, values, descriptions, bench_values, bench_raw_strs = [], [], [], [], []
    for z_col in RADAR_STATS:
        if z_col not in player.index: continue
        z = player.get(z_col)
        if pd.isna(z): continue
        pct = zscore_to_percentile(z)
        label = RADAR_LABEL_OVERRIDES.get(z_col, stat_labels.get(z_col, z_col))
        desc = stat_methodology.get(z_col, {}).get("what", "")
        axes.append(label); values.append(pct); descriptions.append(desc)
        if benchmark is not None:
            bz = benchmark.get(z_col)
            bench_values.append(zscore_to_percentile(bz) if bz is not None and pd.notna(bz) else None)
            raw_v = benchmark_raw.get(z_col) if benchmark_raw else None
            bench_raw_strs.append(_format_radar_raw(z_col, raw_v))
    if not axes: return None
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=axes + [axes[0]],
        customdata=descriptions + [descriptions[0]],
        fill="toself", fillcolor="rgba(31, 119, 180, 0.25)",
        line=dict(color="rgba(31, 119, 180, 0.9)", width=2),
        marker=dict(size=6, color="rgba(31, 119, 180, 1)"),
        name="This player",
        hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<br><br><i>%{customdata}</i><extra></extra>",
    ))
    if benchmark is not None and any(v is not None for v in bench_values):
        bv_clean = [v if v is not None else 50 for v in bench_values]
        bench_hover = []
        for ax, raw_str, pct in zip(axes, bench_raw_strs, bv_clean):
            extra = f"{raw_str} · " if raw_str else ""
            bench_hover.append(f"<b>{ax}</b><br>{benchmark_label}<br>{extra}{pct:.0f}th percentile")
        bench_hover.append(bench_hover[0])
        fig.add_trace(go.Scatterpolar(
            r=bv_clean + [bv_clean[0]], theta=axes + [axes[0]],
            mode="lines+markers",
            line=dict(color="rgba(102, 102, 102, 0.9)", width=2, dash="dot"),
            marker=dict(size=10, color="rgba(102, 102, 102, 0.95)",
                        symbol="diamond", line=dict(width=2, color="white")),
            name=benchmark_label,
            hovertext=bench_hover, hoverinfo="text",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickvals=[25, 50, 75, 100],
                            ticktext=["25th", "50th", "75th", "100th"],
                            tickfont=dict(size=9, color="#888"), gridcolor="#ddd"),
            angularaxis=dict(tickfont=dict(size=11), gridcolor="#ddd"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=(benchmark is not None),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.7)", bordercolor="#ccc", borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=60, r=60, t=20, b=20),
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
# HIDDEN 2026-05-03 — visible page header
# (referenced sliders that are now hidden).
if False:
    st.subheader(f"{team_name} quarterbacks")
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
# HIDDEN 2026-05-03 — tier-checkbox UI; defaults
# applied via session_state read below.
if False:
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
new_enabled = list(
    st.session_state.get(
        "qb_tiers_enabled", [1, 2])
) or [1, 2]
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
min_attempts = st.slider("Minimum pass attempts", 0, 600, 100, step=25, help="Filter out QBs with too few attempts. 100 ≈ 4-5 games of starting (matches the 100-snap floor used elsewhere).")
qbs = df[df["attempts"].fillna(0) >= min_attempts].copy()

if len(qbs) == 0: st.warning("No QBs match the current filter."); st.stop()
qbs = score_players(qbs, effective_weights)

# Metric picker — sort leaderboard by any nerd metric
QB_METRICS = {
    "Passing yards": ("passing_yards", False),
    "Passing TDs": ("passing_tds", False),
    "Attempts": ("attempts", False),
    "Completions": ("completions", False),
    "Completion %": ("completion_pct", False),
    "Yards per attempt": ("yards_per_attempt", False),
    "TD rate": ("td_rate", False),
    "INT rate (lower better)": ("int_rate", True),
    "Sack rate (lower better)": ("sack_rate", True),
    "Turnover rate (lower better)": ("turnover_rate", True),
    "EPA per play": ("pass_epa_per_play", False),
    "CPOE": ("passing_cpoe", False),
    "Pass success rate": ("pass_success_rate", False),
    "First-down rate": ("first_down_rate", False),
    "Air yards per attempt": ("air_yards_per_attempt", False),
    "Passing yds per game": ("passing_yards_per_game", False),
    "Passing TDs per game": ("passing_tds_per_game", False),
    "Rushing yds per game": ("rush_yards_per_game", False),
    "Rush EPA per carry": ("rush_epa_per_carry", False),
}
sort_label, sort_col, sort_ascending = metric_picker(QB_METRICS, key="qb_metric_picker")
total_weight = sum(effective_weights.values())
if total_weight == 0: st.info("All sliders are at zero — slide at least one to the right to see rankings.")
if sort_col in qbs.columns:
    qbs = qbs.sort_values(sort_col, ascending=sort_ascending, na_position="last").reset_index(drop=True)
else:
    qbs = qbs.sort_values("score", ascending=False).reset_index(drop=True)
qbs.index = qbs.index + 1

# ══════════════════════════════════════════════════════════════
# RANKING TABLE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
ranked = qbs.copy()

# ── Master/detail click-to-detail leaderboard ──────────────────
st.markdown("**How to read the score:** 0.00 = avg starting QB (z-scores baselined on top-32 by snaps). The percentile shows where this QB ranks among all qualifying QBs (100+ snaps).")

# Top scorer banner (browse-only)
_top_html = None
_top_warn = None
if len(ranked) > 0:
    _top = ranked.iloc[0]
    _top_name = _top.get("player_display_name", "—")
    _top_team = _top.get("recent_team", "")
    _top_score = _top["score"]
    _top_pct = format_percentile(zscore_to_percentile(_top_score))
    _sign = "+" if _top_score >= 0 else ""
    _top_html = (
        f"<div style='background:#0076B6;color:white;padding:14px 20px;border-radius:8px;"
        f"margin-bottom:8px;font-size:1.1rem;'>"
        f"<span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span>"
        f" &nbsp;·&nbsp; <strong>{_top_name}</strong> ({_top_team})"
        f" &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>"
        f"{_sign}{_top_score:.2f}</span>"
        f" <span style='opacity:0.85;'>({_top_pct})</span></div>"
    )
    _top_warn = sample_size_warning(_top.get("attempts", 0))

def _fmt_int(v): return f"{int(v)}" if pd.notna(v) else "—"
def _fmt_signed(v, places=2): return f"{v:+.{places}f}" if pd.notna(v) else "—"
def _fmt_pct(v): return f"{v*100:.1f}%" if pd.notna(v) else "—"

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "Player": ranked["player_display_name"],
    "Att": ranked.get("attempts", pd.Series([float("nan")] * len(ranked))).apply(_fmt_int),
    "Yds": ranked.get("passing_yards", pd.Series([float("nan")] * len(ranked))).apply(_fmt_int),
    "TDs": ranked.get("passing_tds", pd.Series([float("nan")] * len(ranked))).apply(_fmt_int),
    "INT": ranked.get("passing_interceptions", pd.Series([float("nan")] * len(ranked))).apply(_fmt_int),
    "EPA/play": ranked.get("pass_epa_per_play", pd.Series([float("nan")] * len(ranked))).apply(lambda v: _fmt_signed(v, 2)),
    "CPOE": ranked.get("passing_cpoe", pd.Series([float("nan")] * len(ranked))).apply(lambda v: _fmt_signed(v, 1)),
    "Success%": ranked.get("pass_success_rate", pd.Series([float("nan")] * len(ranked))).apply(_fmt_pct),
    "Your score": ranked["score"].apply(format_score),
})

selected = render_master_detail_leaderboard(
    display_df=display_df,
    name_col="Player",
    key_prefix="qb",
    team=selected_team,
    season=selected_season,
    top_banner_html=_top_html,
    top_banner_warn=_top_warn,
    leaderboard_caption=(
        "**EPA/play** = Expected Points Added per dropback (modern efficiency stat) · "
        "**CPOE** = Completion % over expected (NGS-derived) · "
        "**Success%** = % of dropbacks producing positive EPA. "
        "**Click any player name above** to view their profile."
    ),
)
if selected is None:
    st.stop()

player = ranked[ranked["player_display_name"] == selected].iloc[0]
warn = sample_size_warning(player.get("attempts", 0))
if warn: st.warning(warn)

# ── Split-season panel: surface other stints if traded mid-season ──
all_qbs_full = load_qb_data()
season_stints = all_qbs_full[
    (all_qbs_full["player_id"] == player.get("player_id"))
    & (all_qbs_full["season_year"] == selected_season)
].copy()
if len(season_stints) > 1:
    n = len(season_stints)
    st.info(f"**Split season** — {selected} played for {n} teams in {selected_season}.")
    season_stints = season_stints.sort_values("first_week" if "first_week" in season_stints.columns else "attempts", ascending=True)
    split_rows = []
    for _, stint in season_stints.iterrows():
        team_disp = display_abbr(stint["recent_team"])
        is_current = stint["recent_team"] == player["recent_team"]
        split_rows.append({
            "Team": f"⮕ {team_disp}" if is_current else team_disp,
            "Games": _fmt_int(stint.get("games")),
            "Att": _fmt_int(stint.get("attempts")),
            "Yds": _fmt_int(stint.get("passing_yards")),
            "TDs": _fmt_int(stint.get("passing_tds")),
            "INT": _fmt_int(stint.get("passing_interceptions")),
            "EPA/play": _fmt_signed(stint.get("pass_epa_per_play"), 2),
            "CPOE": _fmt_signed(stint.get("passing_cpoe"), 1),
            "Success%": _fmt_pct(stint.get("pass_success_rate")),
        })
    # Total row — weighted aggregates
    def _safe_sum(col):
        return season_stints[col].fillna(0).sum() if col in season_stints.columns else float("nan")
    def _weighted_mean(value_col, weight_col):
        if value_col not in season_stints.columns or weight_col not in season_stints.columns:
            return float("nan")
        v = season_stints[value_col]; w = season_stints[weight_col]
        mask = v.notna() & w.notna() & (w > 0)
        if not mask.any(): return float("nan")
        return (v[mask] * w[mask]).sum() / w[mask].sum()
    total_games = _safe_sum("games")
    total_att = _safe_sum("attempts")
    total_yds = _safe_sum("passing_yards")
    total_tds = _safe_sum("passing_tds")
    total_int = _safe_sum("passing_interceptions")
    season_epa = _weighted_mean("pass_epa_per_play", "attempts")
    season_cpoe = _weighted_mean("passing_cpoe", "attempts")
    season_success = _weighted_mean("pass_success_rate", "attempts")
    split_rows.append({
        "Team": f"**Total ({selected_season})**",
        "Games": _fmt_int(total_games),
        "Att": _fmt_int(total_att),
        "Yds": _fmt_int(total_yds),
        "TDs": _fmt_int(total_tds),
        "INT": _fmt_int(total_int),
        "EPA/play": _fmt_signed(season_epa, 2),
        "CPOE": _fmt_signed(season_cpoe, 1),
        "Success%": _fmt_pct(season_success),
    })
    st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)
    st.caption(f"⮕ marks the stint shown on this page ({display_abbr(player['recent_team'])}). Stints chronological. Total uses weighted aggregates (rate stats weighted by attempts).")

# ── Unified Season picker — drives stat bar + bundle table + radar ──
player_career = all_qbs_full[all_qbs_full["player_id"] == player.get("player_id")]

_yr = render_player_year_picker(
    career_df=player_career,
    default_season=selected_season,
    season_col="season_year",
    team_col="recent_team",
    key_prefix=f"qb_{player.get('player_id') or selected}",
)
view_row = _yr["view_row"] if _yr["view_row"] is not None else player
year_choice = _yr["year_choice"]

if total_weight > 0:
    _view_score = sum(view_row.get(z, 0) * (w / total_weight)
                       for z, w in effective_weights.items()
                       if pd.notna(view_row.get(z)))
else:
    _view_score = float("nan")

from lib_shared import render_nfl_player_banner
render_nfl_player_banner(
    position="qb", player_name=selected, view_row=view_row,
    score=_view_score,
    season_str=_yr.get("season_str") or f"Season {selected_season}",
    player_career=player_career,
    is_career_view=_yr["is_career_view"],
)

from lib_movement_panel import (
    render_movement_panel, render_advanced_tracking,
)
_yr_for_panels = int(view_row.get("season_year", selected_season))
render_advanced_tracking(selected, "qb", season=_yr_for_panels)
render_movement_panel(selected, "qb", season=_yr_for_panels)

QB_STAT_SPECS = [
    ("passing_yards", "{:.0f}", "Pass Yds"),
    ("passing_tds", "{:.0f}", "TD"),
    ("passing_interceptions", "{:.0f}", "INT"),
    ("yards_per_attempt", "{:.1f}", "Y/Att"),
    ("pass_epa_per_play", "{:+.2f}", "EPA/Play"),
    ("passing_cpoe", "{:+.1f}", "CPOE"),
]
NFL_SUM_COLS = {"off_snaps", "def_snaps", "snaps", "games", "targets",
                "receptions", "rec_yards", "rec_tds",
                "attempts", "completions", "passing_yards", "passing_tds",
                "passing_interceptions", "rushing_yards", "rushing_tds",
                "carries", "rushing_attempts", "tackles", "def_tackles",
                "sacks", "tfls", "tackles_for_loss",
                "interceptions", "def_interceptions", "passes_defensed",
                "passes_defended", "qb_hits", "fg_made", "fg_attempts",
                "fg_att", "xp_made", "punts", "punt_yards", "total_yards"}
# ── Trading-card visual ────────────────────────────────────────
_team_abbr = _yr["team_str"] if _yr["team_str"] else (player.get("recent_team") or "")
# In-page banner removed — the trading card below is now the page hero.

# ── Trading-card export ──────────────────────────────────────────
def _safe_fmt(v, fmt="{:.0f}"):
    if v is None or (isinstance(v, float) and pd.isna(v)): return "—"
    try: return fmt.format(v)
    except: return str(v)

from lib_player_blurb import make_card_narrative
_card_narrative = make_card_narrative(view_row, all_qbs_full, "qb")

_card_stats = [
    ("Pass yds", _safe_fmt(view_row.get("passing_yards")),
                 _safe_fmt(view_row.get("passing_tds"), "{:.0f} TD")),
    ("INT",      _safe_fmt(view_row.get("passing_interceptions")), ""),
    ("Y/Att",    _safe_fmt(view_row.get("yards_per_attempt"), "{:.1f}"), ""),
    ("EPA/Play", _safe_fmt(view_row.get("pass_epa_per_play"), "{:+.2f}"), ""),
]

from lib_shared import team_theme as _theme
from lib_trading_card import render_card_download_button as _render_card
_render_card(
    player_name=selected,
    position_label=(player.get("position") or "QB"),
    season_str=_yr["season_str"] or f"Season {selected_season}",
    score=_view_score,
    narrative=_card_narrative,
    key_stats=_card_stats,
    player_id=player.get("player_id") or selected,
    team_abbr=_team_abbr,
    theme=_theme(_team_abbr),
    preset_name=(st.session_state.qb_loaded_algo.get("name")
                  if st.session_state.get("qb_loaded_algo") else None),
    key_prefix=f"qb_{player.get('player_id') or selected}",
    position_group="qb",
    bundle_weights=bundle_weights,
    season=(None if _yr["is_career_view"] else selected_season),
)

# ════════════════════════════════════════════════════════════════
# TABBED PLAYER DETAIL — Profile / Game-Context / Compare /
# Career / Game splits
# Trading card hero stays sticky above the tabs.
# ════════════════════════════════════════════════════════════════

# Compute the radar benchmark once — used by Profile + Compare tabs
_radar_row = view_row if view_row is not None else player
_season_pool_qb = all_qbs_full[all_qbs_full["season_year"] == selected_season]
_top32_qb = _season_pool_qb.sort_values("attempts", ascending=False).head(32)
_radar_bench = {z: _top32_qb[z].mean() for z in RADAR_STATS
                  if z in _top32_qb.columns and _top32_qb[z].notna().any()}
_radar_bench_raw = {}
for z in RADAR_STATS:
    raw_col = RAW_COL_MAP.get(z)
    if raw_col and raw_col in _top32_qb.columns and _top32_qb[raw_col].notna().any():
        _radar_bench_raw[z] = _top32_qb[raw_col].mean()


def _qb_score_of(row):
    if row is None or total_weight <= 0:
        return float("nan")
    return sum(
        row.get(z, 0) * (w / total_weight)
        for z, w in effective_weights.items()
        if pd.notna(row.get(z))
    )


tab_profile, tab_panel, tab_compare, tab_career, tab_splits = st.tabs([
    "📊 Score & Profile",
    "🎯 Game-Context Analysis",
    "⚔️ Compare",
    "📈 Career & Combine",
    "📅 Game-by-game",
])


# ─── 📊 SCORE & PROFILE ─────────────────────────────────
with tab_profile:
    c1, c2 = st.columns([1, 1])
    with c1:
        _sign = "+" if pd.notna(_view_score) and _view_score >= 0 else ""
        _pct = format_percentile(zscore_to_percentile(_view_score)) if pd.notna(_view_score) else "—"
        _score_str = f"{_sign}{_view_score:.2f}" if pd.notna(_view_score) else "—"
        st.markdown(f"**Your score: {_score_str} ({_pct})**")
        st.markdown("_This score is based on your slider settings. Change the sliders and this number changes._")
        st.markdown("---")
        st.markdown("**Where the score comes from**")
        st.markdown("Each row shows how much one skill contributed to the total, based on your slider weights.")

        if not advanced_mode:
            stat_rows = []; shown = set()
            for bundle in active_bundles.values(): shown.update(bundle["stats"].keys())
            for z_col in sorted(shown, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                raw_col = RAW_COL_MAP.get(z_col)
                z = view_row.get(z_col); raw = view_row.get(raw_col) if raw_col else None
                pct = zscore_to_percentile(z) if pd.notna(z) else None
                if raw_col in ("completion_pct", "td_rate", "int_rate", "sack_rate", "first_down_rate", "turnover_rate"):
                    raw_fmt = f"{raw:.1%}" if pd.notna(raw) else "—"
                elif raw_col in ("passing_cpoe",):
                    raw_fmt = f"{raw:+.2f}" if pd.notna(raw) else "—"
                else:
                    raw_fmt = f"{raw:.2f}" if pd.notna(raw) else "—"
                stat_rows.append({"Stat": stat_labels.get(z_col, z_col), "Value": raw_fmt, "Percentile": f"{int(pct)}th" if pct is not None else "—"})
            if stat_rows:
                st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

            with st.expander("⚙️  How your slider preset weights this player"):
                bundle_rows = []
                for bk, bundle in active_bundles.items():
                    bw = bundle_weights.get(bk, 0)
                    if bw == 0: continue
                    contribution = sum(
                        view_row.get(z, 0) * (bw * internal / total_weight)
                        for z, internal in bundle["stats"].items()
                        if pd.notna(view_row.get(z)) and total_weight > 0
                    )
                    bundle_rows.append({"Skill": bundle["label"], "Your weight": f"{bw}", "Points added": f"{contribution:+.2f}"})
                if bundle_rows:
                    st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No bundles weighted — drag some sliders.")
        else:
            rows = []
            for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                raw_col = RAW_COL_MAP.get(z_col)
                z = view_row.get(z_col); raw = view_row.get(raw_col) if raw_col else None
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
        st.caption("Solid blue = this player. Dashed gray = top-32 starter average. INT rate and sack rate are inverted (higher = fewer turnovers/sacks).")
        fig = build_radar_figure(_radar_row, stat_labels, stat_methodology,
                                  benchmark=_radar_bench, benchmark_raw=_radar_bench_raw)
        if fig: st.plotly_chart(fig, use_container_width=True)


# ─── 🎯 GAME-CONTEXT ANALYSIS (the QB panel) ────────────
with tab_panel:
    from lib_qb_panel import (
        get_qb_peers as _get_qb_peers,
        render_pressure_split as _render_pressure_split,
        render_competition_split as _render_competition_split,
        render_throw_map as _render_throw_map,
        render_situational_split as _render_situational_split,
        render_presnap_split as _render_presnap_split,
        render_processing_split as _render_processing_split,
    )
    _qb_panel_pid = player.get("player_id")
    if not _qb_panel_pid:
        st.info("No player_id available — game-context analysis isn't available.")
    else:
        _qb_panel_season = None if _yr["is_career_view"] else selected_season

        # Comparison picker — drives every panel below in side-by-side mode
        _qb_peer_options = _get_qb_peers(
            season=_qb_panel_season,
            exclude_player_id=_qb_panel_pid,
        )
        _comp_labels = ["None"] + [opt["label"] for opt in _qb_peer_options]
        _comp_pick = st.selectbox(
            "Compare to another QB:",
            options=_comp_labels,
            index=0,
            key=f"qb_compare_{_qb_panel_pid}",
            help="Pick another QB-season to render every panel below in "
                 "side-by-side comparison mode.",
        )
        if _comp_pick != "None":
            _comp_idx = _comp_labels.index(_comp_pick) - 1
            _comp = _qb_peer_options[_comp_idx]
            _comp_pid = _comp["player_id"]
            _comp_name = _comp["label"].split(" — ")[0]
            _comp_season = _comp["season"]
        else:
            _comp_pid = _comp_name = _comp_season = None

        # HERO panel: throw map (always visible at top of tab)
        st.markdown("### 🎯 Throw map — where does he hit?")
        _render_throw_map(
            player_id=_qb_panel_pid,
            player_name=selected,
            season=_qb_panel_season,
            theme=_theme(_team_abbr),
            key_prefix=f"qb_{_qb_panel_pid}",
            comparison_player_id=_comp_pid,
            comparison_player_name=_comp_name,
            comparison_season=_comp_season,
        )

        _comp_kwargs = dict(
            comparison_player_id=_comp_pid,
            comparison_player_name=_comp_name,
            comparison_season=_comp_season,
        )

        with st.expander("📋  Pre-snap — formation, tempo, down splits", expanded=False):
            _render_presnap_split(
                player_id=_qb_panel_pid, player_name=selected,
                season=_qb_panel_season, theme=_theme(_team_abbr), **_comp_kwargs,
            )
        with st.expander("🧠  Processing — time to throw, aggressiveness, depth", expanded=False):
            _render_processing_split(
                player_id=_qb_panel_pid, player_name=selected,
                season=_qb_panel_season, theme=_theme(_team_abbr), **_comp_kwargs,
            )
        with st.expander("🥊  Under pressure — clean pocket vs. pressured", expanded=False):
            _render_pressure_split(
                player_id=_qb_panel_pid, player_name=selected,
                season=_qb_panel_season, theme=_theme(_team_abbr), **_comp_kwargs,
            )
        with st.expander("⏱️  Situational — 3rd down, red zone, 4th quarter, 2-min drill", expanded=False):
            _render_situational_split(
                player_id=_qb_panel_pid, player_name=selected,
                season=_qb_panel_season, theme=_theme(_team_abbr), **_comp_kwargs,
            )
        with st.expander("🏆  Elite vs. weak competition — does he rise to elite D or feast on bad ones?", expanded=False):
            _render_competition_split(
                player_id=_qb_panel_pid, player_name=selected,
                season=_qb_panel_season, theme=_theme(_team_abbr), **_comp_kwargs,
            )


# ─── ⚔️ COMPARE ─────────────────────────────────────────
with tab_compare:
    from lib_shared import render_player_comparison
    render_player_comparison(
        player_row=view_row,
        player_name=selected,
        league_df=all_qbs_full,
        name_col="player_display_name",
        year_choice=year_choice,
        primary_score=_view_score,
        compute_comparison_score=_qb_score_of,
        radar_builder=build_radar_figure,
        benchmark=_radar_bench,
        benchmark_raw=_radar_bench_raw,
        stat_labels=stat_labels,
        stat_methodology=stat_methodology,
        key_prefix=f"qb_cmp_{player.get('player_id', selected)}",
        position_label="quarterback",
        theme=_theme(player.get("recent_team") or ""),
    )


# ─── 📈 CAREER & COMBINE ────────────────────────────────
with tab_career:
    _WORKOUTS_PATH = Path(__file__).resolve().parent.parent / "data" / "college" / "nfl_all_workouts.parquet"
    render_combine_chart(
        player_name=selected,
        position="QB",
        workouts_path=_WORKOUTS_PATH,
        key=f"qb_combine_chart_{player.get('player_id', selected)}",
    )
    career_arc_section(
        player=player,
        league_parquet_path=DATA_PATH,
        z_score_cols=list(RAW_COL_MAP.keys()),
        stat_labels=stat_labels,
        id_col="player_id",
        name_col="player_display_name",
        position_label="quarterbacks",
    )


# ─── 📅 GAME-BY-GAME SPLITS ─────────────────────────────
with tab_splits:
    from lib_splits import render_splits_section as _render_splits_section
    _render_splits_section(
        player_name=selected,
        season=selected_season,
        position_group="QB",
        key_prefix=f"qb_{player.get('player_id') or selected}",
        is_career_view=_yr["is_career_view"],
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    "Data via [nflverse](https://github.com/nflverse) · 2024 regular season · "
    "Z-scored against 39 QBs with 200+ pass attempts · "
    "Fan project, not affiliated with the NFL or Detroit Lions."
)

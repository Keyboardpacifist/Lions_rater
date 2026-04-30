"""
Lions DE Rater — 2024 season
Monkey-proofed UI: every control explains WHAT it does and WHY you'd use it.
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

st.set_page_config(page_title="DE Rater", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")
inject_css()

# ── Team & Season selector ────────────────────────────────────
selected_team, selected_season = get_team_and_season()
team_name = NFL_TEAMS.get(selected_team, selected_team)

POSITION_GROUP = "de"
PAGE_URL = "https://lions-rater.streamlit.app/DE"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "league_de_all_seasons.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "de_stat_metadata.json"

@st.cache_data
def load_de_data(): return pl.read_parquet(DATA_PATH).to_pandas()
@st.cache_data
def load_de_metadata():
    if not METADATA_PATH.exists(): return {}
    with open(METADATA_PATH) as f: return json.load(f)

# ── Stat definitions ──────────────────────────────────────────
RAW_COL_MAP = {
    "sacks_per_game_z": "sacks_per_game", "qb_hits_per_game_z": "qb_hits_per_game",
    "pressure_rate_z": "pressure_rate",
    "pressures_per_game_z": "pressures_per_game",
    "hurries_per_game_z": "hurries_per_game",
    "qb_knockdowns_per_game_z": "qb_knockdowns_per_game",
    "missed_tackle_pct_z": "missed_tackle_pct",
    "tfl_per_game_z": "tfl_per_game",
    "solo_tackle_rate_z": "solo_tackle_rate", "tackles_per_snap_z": "tackles_per_snap",
    "forced_fumbles_per_game_z": "forced_fumbles_per_game",
    "passes_defended_per_game_z": "passes_defended_per_game",
    "interceptions_per_game_z": "interceptions_per_game",
}

BUNDLES = {
    "pass_rush": {
        "label": "🔥 Pass rush",
        "description": "Can he get to the QB? Sacks, hits, hurries, and total pressures.",
        "why": "Think rushing the passer is the most important thing a DE does? Crank this up.",
        "stats": {
            "sacks_per_game_z": 0.25, "qb_hits_per_game_z": 0.15,
            "pressures_per_game_z": 0.25, "hurries_per_game_z": 0.15,
            "qb_knockdowns_per_game_z": 0.10, "pressure_rate_z": 0.10,
        },
    },
    "run_defense": {
        "label": "🛡️ Run defense",
        "description": "Can he stop the run? TFLs, solo tackles, tackle reliability.",
        "why": "Value edge-setters who stuff the run? Slide this right.",
        "stats": {
            "tfl_per_game_z": 0.35, "solo_tackle_rate_z": 0.20,
            "tackles_per_snap_z": 0.25, "missed_tackle_pct_z": 0.20,
        },
    },
    "playmaking": {
        "label": "💥 Playmaking",
        "description": "Does he create turnovers? Forced fumbles, batted passes, interceptions.",
        "why": "Want DEs who change games with forced turnovers? Slide this right.",
        "stats": {"forced_fumbles_per_game_z": 0.40, "passes_defended_per_game_z": 0.35, "interceptions_per_game_z": 0.25},
    },
}
DEFAULT_BUNDLE_WEIGHTS = {"pass_rush": 60, "run_defense": 40, "playmaking": 30}

RADAR_STATS = list(RAW_COL_MAP.keys())
RADAR_INVERT = set()
RADAR_LABEL_OVERRIDES = {
    "sacks_per_game_z": "Sacks", "qb_hits_per_game_z": "QB hits",
    "pressure_rate_z": "Pressure rate", "tfl_per_game_z": "TFLs",
    "pressures_per_game_z": "Pressures",
    "hurries_per_game_z": "Hurries",
    "qb_knockdowns_per_game_z": "Knockdowns",
    "missed_tackle_pct_z": "Tackle reliability",
    "solo_tackle_rate_z": "Solo tackle %", "tackles_per_snap_z": "Tackles/snap",
    "forced_fumbles_per_game_z": "Forced fumbles",
    "passes_defended_per_game_z": "Pass defense",
    "interceptions_per_game_z": "Interceptions",
}

# ── Tier system ───────────────────────────────────────────────
TIER_LABELS = {
    1: "Counting stats",
    2: "Rate stats",
    3: "Modeled stats",
    4: "Estimated stats",
}
TIER_DESCRIPTIONS = {
    1: "Sacks, tackles, forced fumbles — raw totals per game.",
    2: "Per-game and per-snap averages that adjust for playing time.",
    3: "Stats adjusted for expected performance based on a model.",
    4: "Inferred from limited data — least reliable. Use with caution.",
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
    for z in bundle_stats:
        t = stat_tiers.get(z, 2)
        counts[t] = counts.get(t, 0) + 1
    return " ".join(f"{tier_badge(t)}×{c}" for t, c in sorted(counts.items()))

# ── Score formatting ──────────────────────────────────────────
def zscore_to_percentile(z):
    if pd.isna(z): return None
    return float(norm.cdf(z) * 100)

def format_percentile(pct):
    """Format percentile as human-readable rank."""
    if pct is None or pd.isna(pct): return "—"
    if pct >= 99: return "top 1%"
    if pct >= 95: return f"top {100 - int(pct)}%"
    if pct >= 50: return f"top {100 - int(pct)}%"
    return f"bottom {int(pct)}%"

def format_score(score):
    """Format score as +1.89 (top 3%)."""
    if pd.isna(score): return "—"
    sign = "+" if score >= 0 else ""
    pct = zscore_to_percentile(score)
    pct_label = format_percentile(pct)
    return f"{sign}{score:.2f} ({pct_label})"

def format_score_short(score):
    """Just the percentile for table display."""
    if pd.isna(score): return "—"
    pct = zscore_to_percentile(score)
    return format_percentile(pct)

def sample_size_warning(snaps):
    if pd.isna(snaps): return ""
    if snaps < 300: return f"⚠️ Only {int(snaps)} snaps — small sample, treat with caution"
    if snaps < 500: return f"⚠️ {int(snaps)} snaps — moderate sample"
    return ""

# Per-stat raw value formatting for the radar benchmark hover.
_RADAR_RAW_FORMATTERS = {
    "sacks_per_game_z": ("sacks/g", lambda v: f"{v:.2f}"),
    "qb_hits_per_game_z": ("QB hits/g", lambda v: f"{v:.2f}"),
    "pressure_rate_z": ("press rate", lambda v: f"{v*100:.1f}%"),
    "tfl_per_game_z": ("TFL/g", lambda v: f"{v:.2f}"),
    "solo_tackle_rate_z": ("solo tkl rate", lambda v: f"{v*100:.1f}%"),
    "tackles_per_snap_z": ("tkl/snap", lambda v: f"{v:.3f}"),
    "forced_fumbles_per_game_z": ("FF/g", lambda v: f"{v:.2f}"),
    "passes_defended_per_game_z": ("PD/g", lambda v: f"{v:.2f}"),
    "interceptions_per_game_z": ("INT/g", lambda v: f"{v:.2f}"),
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
def build_radar_figure(player, stat_labels, stat_methodology, benchmark=None, benchmark_raw=None, benchmark_label="Top 32 starter avg"):
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
        bench_hover = [f"<b>{ax}</b><br>{benchmark_label}<br>{(rs + ' · ') if rs else ''}{p:.0f}th percentile" for ax, rs, p in zip(axes, bench_raw_strs, bv_clean)]
        bench_hover.append(bench_hover[0])
        fig.add_trace(go.Scatterpolar(
            r=bv_clean + [bv_clean[0]], theta=axes + [axes[0]],
            mode="lines+markers",
            line=dict(color="rgba(102, 102, 102, 0.9)", width=2, dash="dot"),
            marker=dict(size=10, color="rgba(102, 102, 102, 0.95)", symbol="diamond", line=dict(width=2, color="white")),
            name=benchmark_label, hovertext=bench_hover, hoverinfo="text",
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
                    bgcolor="rgba(255,255,255,0.7)", bordercolor="#ccc", borderwidth=1, font=dict(size=10)),
        margin=dict(l=60, r=60, t=20, b=20),
        height=380, paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── Session state ─────────────────────────────────────────────
if "de_loaded_algo" not in st.session_state: st.session_state.de_loaded_algo = None
if "upvoted_ids" not in st.session_state: st.session_state.upvoted_ids = set()
if "de_tiers_enabled" not in st.session_state: st.session_state.de_tiers_enabled = [1, 2]

# ── Load data ─────────────────────────────────────────────────
try:
    df = load_de_data()
except FileNotFoundError:
    st.error(f"Couldn't find DE data at {DATA_PATH}.")
    st.stop()

# Filter to selected team and season
df = filter_by_team_and_season(df, selected_team, selected_season, team_col="recent_team", season_col="season_year")
if len(df) == 0:
    st.warning(f"No {team_name} defensive ends found for {selected_season}.")
    st.stop()

meta = load_de_metadata()
stat_tiers = meta.get("stat_tiers", {})
stat_labels = meta.get("stat_labels", {})
stat_methodology = meta.get("stat_methodology", {})

if "algo" in st.query_params and st.session_state.de_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP:
        apply_algo_weights(linked, BUNDLES)
        st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════
st.subheader(f"{team_name} defensive ends")
st.markdown("What makes a great DE? **You decide.** Use the sliders on the left to tell us what you value most, and the rankings update instantly.")
st.caption(f"{selected_season} regular season · Compared to all 90 DEs league-wide with 200+ snaps")

# ══════════════════════════════════════════════════════════════
# SIDEBAR — "What matters to you?"
# ══════════════════════════════════════════════════════════════
st.sidebar.header("What matters to you?")
st.sidebar.markdown("Each slider controls how much a skill affects the final score. Slide right to prioritize it, or all the way left to ignore it.")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

if st.session_state.de_loaded_algo:
    la = st.session_state.de_loaded_algo
    st.sidebar.info(f"Loaded: **{la['name']}** by {la['author']}\n\n_{la.get('description', '')}_")
    if st.sidebar.button("Clear loaded algorithm"):
        st.session_state.de_loaded_algo = None

# ══════════════════════════════════════════════════════════════
# STAT TYPE CHECKBOXES (tiers)
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
                value=(tier in st.session_state.de_tiers_enabled),
                help=TIER_DESCRIPTIONS[tier],
                key=f"de_tier_checkbox_{tier}",
            )
            if checked:
                new_enabled.append(tier)
        else:
            st.markdown(f"<span style='opacity:0.35'>{tier_badge(tier)} {TIER_LABELS[tier]}</span>", unsafe_allow_html=True)
            st.caption("No stats available")
st.session_state.de_tiers_enabled = new_enabled
if not new_enabled:
    st.warning("Check at least one box above to include some stats.")
    st.stop()
active_bundles = filter_bundles_by_tier(BUNDLES, stat_tiers, new_enabled)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR SLIDERS (bundles or advanced)
# ══════════════════════════════════════════════════════════════
advanced_mode = False
bundle_weights = {}
effective_weights = {}

if not active_bundles:
    st.info("No stat bundles available for the selected stat types.")
    st.stop()

for bk, bundle in active_bundles.items():
    tier_summary = bundle_tier_summary(bundle["stats"], stat_tiers)
    st.sidebar.markdown(f"**{bundle['label']}**")
    st.sidebar.markdown(f"{bundle['description']}")
    if f"de_bundle_{bk}" not in st.session_state:
        st.session_state[f"de_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
    bundle_weights[bk] = st.sidebar.slider(
        bundle["label"], 0, 100, step=5,
        key=f"de_bundle_{bk}", label_visibility="collapsed",
        help=bundle.get("why", ""),
    )
    st.sidebar.caption(f"_↑ {bundle.get('why', '')}_")

for bk in BUNDLES:
    if bk not in bundle_weights:
        bundle_weights[bk] = 0
effective_weights = compute_effective_weights(active_bundles, bundle_weights)

# Advanced mode — hidden behind expander
with st.sidebar.expander("Want more control? Adjust individual stats"):
    advanced_mode = st.checkbox("Enable individual stat control", value=False, key="de_advanced_toggle")
    if advanced_mode:
        st.caption("Set the weight of each individual stat. This overrides the bundle sliders above.")
        effective_weights = {}
        all_enabled_stats = sorted(
            [z for z, t in stat_tiers.items() if t in new_enabled],
            key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)),
        )
        for z_col in all_enabled_stats:
            label = stat_labels.get(z_col, z_col)
            meth = stat_methodology.get(z_col, {})
            help_text = meth.get("what", "")
            if meth.get("limits"):
                help_text += f"\n\nLimits: {meth['limits']}"
            w = st.slider(
                f"{tier_badge(stat_tiers.get(z_col, 2))} {label}",
                0, 100, 50, 5,
                key=f"adv_de_{z_col}",
                help=help_text if help_text else None,
            )
            if w > 0:
                effective_weights[z_col] = w
        bundle_weights = {bk: 0 for bk in BUNDLES}

# ══════════════════════════════════════════════════════════════
# COMPUTE SCORES & RANK
# ══════════════════════════════════════════════════════════════
des = df.copy()
if len(des) == 0:
    st.warning("No DEs found.")
    st.stop()
des = score_players(des, effective_weights)

# Metric picker — sort by any nerd metric
DE_METRICS = {
    "Sacks": ("def_sacks", False),
    "QB hits": ("def_qb_hits", False),
    "Tackles for loss": ("def_tackles_for_loss", False),
    "Pressures (PFR)": ("pfr_pressures", False),
    "Pressure rate": ("pressure_rate", False),
    "Sacks per game": ("sacks_per_game", False),
    "QB hits per game": ("qb_hits_per_game", False),
    "TFL per game": ("tfl_per_game", False),
    "Tackles per snap": ("tackles_per_snap", False),
    "Forced fumbles per game": ("forced_fumbles_per_game", False),
    "Passes defended per game": ("passes_defended_per_game", False),
    "Solo tackle rate": ("solo_tackle_rate", False),
    "Missed tackle % (lower better)": ("pfr_missed_tackle_pct", True),
}
sort_label, sort_col, sort_ascending = metric_picker(DE_METRICS, key="de_metric_picker")
total_weight = sum(effective_weights.values())
if total_weight == 0:
    st.info("All sliders are at zero — slide at least one to the right to see rankings.")
if sort_col in des.columns:
    des = des.sort_values(sort_col, ascending=sort_ascending, na_position="last").reset_index(drop=True)
else:
    des = des.sort_values("score", ascending=False).reset_index(drop=True)
des.index = des.index + 1

# ══════════════════════════════════════════════════════════════
# RANKING TABLE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
ranked = des.copy()

# ── Master/detail click-to-detail leaderboard ──────────────────
# Score guide — visible by default
st.markdown("""
**How to read the score:** 0.00 = avg starter · Positive = above avg starter · Negative = below.
The percentile shows where this player ranks among all qualifying DEs league-wide (100+ defensive snaps).
""")

# Top scorer banner (browse-only)
_top_html = None
_top_warn = None
if len(ranked) > 0:
    _top = ranked.iloc[0]
    _top_name = _top.get("player_name", "—")
    _top_score = _top["score"]
    _top_pct = format_percentile(zscore_to_percentile(_top_score))
    _sign = "+" if _top_score >= 0 else ""
    _top_html = (
        f"<div style='background:#0076B6;color:white;padding:14px 20px;border-radius:8px;"
        f"margin-bottom:8px;font-size:1.1rem;'>"
        f"<span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span>"
        f" &nbsp;·&nbsp; <strong>{_top_name}</strong>"
        f" &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>"
        f"{_sign}{_top_score:.2f}</span>"
        f" <span style='opacity:0.85;'>({_top_pct})</span></div>"
    )
    _top_warn = sample_size_warning(_top.get("def_snaps", 0))

def _fmt_int(v): return f"{int(v)}" if pd.notna(v) else "—"
def _fmt_pct(v): return f"{v*100:.1f}%" if pd.notna(v) else "—"
def _fmt_signed(v, places=2): return f"{v:+.{places}f}" if pd.notna(v) else "—"
def _fmt_float(v, places=2): return f"{v:.{places}f}" if pd.notna(v) else "—"

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "Player": ranked["player_name"],
    "Snaps": ranked.get("def_snaps", pd.Series([0] * len(ranked))).apply(
        lambda s: f"{int(s)} ⚠️" if pd.notna(s) and s < 300 else _fmt_int(s)
    ),
    "Sacks": ranked.get("def_sacks", pd.Series([float("nan")] * len(ranked))).apply(lambda v: _fmt_float(v, 1)),
    "QB hits": ranked.get("def_qb_hits", pd.Series([float("nan")] * len(ranked))).apply(_fmt_int),
    "TFL": ranked.get("def_tackles_for_loss", pd.Series([float("nan")] * len(ranked))).apply(_fmt_int),
    "Pressures": ranked.get("pfr_pressures", pd.Series([float("nan")] * len(ranked))).apply(_fmt_int),
    "Press rate": ranked.get("pressure_rate", pd.Series([float("nan")] * len(ranked))).apply(_fmt_pct),
    "Your score": ranked["score"].apply(format_score),
})

selected = render_master_detail_leaderboard(
    display_df=display_df,
    name_col="Player",
    key_prefix="de",
    team=selected_team,
    season=selected_season,
    top_banner_html=_top_html,
    top_banner_warn=_top_warn,
    leaderboard_caption=(
        "⚠️ = under 300 snaps. **Pressures** = PFR pressures (sacks + hurries + QB hits) · "
        "**Press rate** = pressures / estimated pass plays defended while on field. "
        "**Click any player name above** to view their profile."
    ),
)
if selected is None:
    st.stop()

player = ranked[ranked["player_name"] == selected].iloc[0]
warn = sample_size_warning(player.get("def_snaps", 0))
if warn:
    st.warning(warn)

# ── Split-season panel for traded DEs ──
all_des_full = load_de_data()
season_stints = all_des_full[
    (all_des_full["player_id"] == player.get("player_id"))
    & (all_des_full["season_year"] == selected_season)
].copy() if "player_id" in all_des_full.columns and "season_year" in all_des_full.columns else pd.DataFrame()
if len(season_stints) > 1:
    n = len(season_stints)
    st.info(f"**Split season** — {selected} played for {n} teams in {selected_season}.")
    season_stints = season_stints.sort_values("first_week" if "first_week" in season_stints.columns else "def_snaps", ascending=True)
    split_rows = []
    for _, stint in season_stints.iterrows():
        team_disp = display_abbr(stint["recent_team"])
        is_current = stint["recent_team"] == player["recent_team"]
        split_rows.append({
            "Team": f"⮕ {team_disp}" if is_current else team_disp,
            "Games": _fmt_int(stint.get("games")),
            "Snaps": _fmt_int(stint.get("def_snaps")),
            "Sacks": _fmt_float(stint.get("def_sacks"), 1),
            "QB hits": _fmt_int(stint.get("def_qb_hits")),
            "TFL": _fmt_int(stint.get("def_tackles_for_loss")),
            "Pressures": _fmt_int(stint.get("pfr_pressures")),
            "Press rate": _fmt_pct(stint.get("pressure_rate")),
        })
    # Total row — sum counts; press rate = total pressures / total exposure
    def _safe_sum(col):
        return season_stints[col].fillna(0).sum() if col in season_stints.columns else float("nan")
    total_press = _safe_sum("pfr_pressures")
    total_exposure = _safe_sum("pass_plays_exposure") if "pass_plays_exposure" in season_stints.columns else float("nan")
    season_press_rate = (total_press / total_exposure) if total_exposure > 0 else float("nan")
    split_rows.append({
        "Team": f"**Total ({selected_season})**",
        "Games": _fmt_int(_safe_sum("games")),
        "Snaps": _fmt_int(_safe_sum("def_snaps")),
        "Sacks": _fmt_float(_safe_sum("def_sacks"), 1),
        "QB hits": _fmt_int(_safe_sum("def_qb_hits")),
        "TFL": _fmt_int(_safe_sum("def_tackles_for_loss")),
        "Pressures": _fmt_int(total_press),
        "Press rate": _fmt_pct(season_press_rate),
    })
    st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)
    st.caption(f"⮕ marks the stint shown on this page ({display_abbr(player['recent_team'])}). Stints chronological. Total combines per-stint counts; press rate = total pressures / total exposure.")

# ── Unified Season picker — drives stat bar + bundle table + radar ──
player_career = all_des_full[all_des_full["player_id"] == player.get("player_id")] if "player_id" in all_des_full.columns else all_des_full[0:0]

st.markdown(f"### {selected}")

_yr = render_player_year_picker(
    career_df=player_career,
    default_season=selected_season,
    season_col="season_year",
    team_col="recent_team",
    key_prefix=f"de_{player.get('player_id') or selected}",
)
view_row = _yr["view_row"] if _yr["view_row"] is not None else player
year_choice = _yr["year_choice"]

if total_weight > 0:
    _view_score = sum(view_row.get(z, 0) * (w / total_weight)
                       for z, w in effective_weights.items()
                       if pd.notna(view_row.get(z)))
else:
    _view_score = float("nan")

DE_STAT_SPECS = [
    ("def_tackles", "{:.0f}", "Tkl"),
    ("def_sacks", "{:.1f}", "Sacks"),
    ("def_tackles_for_loss", "{:.1f}", "TFL"),
    ("def_qb_hits", "{:.0f}", "Hits"),
    ("pressure_rate", "{:.1%}", "Press%"),
    ("def_snaps", "{:.0f}", "Snaps"),
]
NFL_SUM_COLS = {"off_snaps", "def_snaps", "snaps", "games", "targets",
                "receptions", "rec_yards", "rec_tds",
                "attempts", "completions", "passing_yards", "passing_tds",
                "passing_interceptions", "rushing_yards", "rushing_tds",
                "carries", "rushing_attempts", "tackles", "def_tackles",
                "def_sacks", "def_qb_hits", "def_tackles_for_loss",
                "pfr_pressures",
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
_card_narrative = make_card_narrative(view_row, all_des_full, "de")

_card_stats = [
    ("Sacks",  _safe_fmt(view_row.get("def_sacks"), "{:.1f}"),
               _safe_fmt(view_row.get("def_qb_hits"), "{:.0f} hits")),
    ("TFL",    _safe_fmt(view_row.get("def_tackles_for_loss"), "{:.1f}"), ""),
    ("Press%", _safe_fmt(view_row.get("pressure_rate"), "{:.1%}"), ""),
    ("Snaps",  _safe_fmt(view_row.get("def_snaps")), ""),
]

from lib_shared import team_theme as _theme
from lib_trading_card import render_card_download_button as _render_card
_render_card(
    player_name=selected,
    position_label=(player.get("position") or "DE"),
    season_str=_yr["season_str"] or f"Season {selected_season}",
    score=_view_score,
    narrative=_card_narrative,
    key_stats=_card_stats,
    player_id=player.get("player_id") or selected,
    team_abbr=_team_abbr,
    theme=_theme(_team_abbr),
    preset_name=(st.session_state.de_loaded_algo.get("name")
                  if st.session_state.get("de_loaded_algo") else None),
    key_prefix=f"de_{player.get('player_id') or selected}",
    position_group="de",
    bundle_weights=bundle_weights,
    season=(None if _yr["is_career_view"] else selected_season),
)

# ════════════════════════════════════════════════════════════════
# TABBED PLAYER DETAIL — Profile / Compare / Career & Combine / Splits
# Trading card hero stays sticky above the tabs.
# ════════════════════════════════════════════════════════════════

_radar_row = view_row if view_row is not None else player
_season_pool_de = all_des_full[all_des_full["season_year"] == selected_season] if "season_year" in all_des_full.columns else all_des_full
_snap_col_de = "def_snaps" if "def_snaps" in _season_pool_de.columns else "off_snaps"
_top32_de = _season_pool_de.sort_values(_snap_col_de, ascending=False).head(32)
_radar_bench = {z: _top32_de[z].mean() for z in RADAR_STATS if z in _top32_de.columns and _top32_de[z].notna().any()}
_radar_bench_raw = {}
for z in RADAR_STATS:
    raw_col = RAW_COL_MAP.get(z)
    if raw_col and raw_col in _top32_de.columns and _top32_de[raw_col].notna().any():
        _radar_bench_raw[z] = _top32_de[raw_col].mean()


def _de_score_of(row):
    if row is None or total_weight <= 0:
        return float("nan")
    return sum(
        row.get(z, 0) * (w / total_weight)
        for z, w in effective_weights.items()
        if pd.notna(row.get(z))
    )


tab_profile, tab_compare, tab_career, tab_splits = st.tabs([
    "📊 Score & Profile",
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
            stat_rows = []
            shown = set()
            for bundle in active_bundles.values():
                shown.update(bundle["stats"].keys())
            for z_col in sorted(shown, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                raw_col = RAW_COL_MAP.get(z_col)
                z = view_row.get(z_col)
                raw = view_row.get(raw_col) if raw_col else None
                pct = zscore_to_percentile(z) if pd.notna(z) else None
                stat_rows.append({
                    "Stat": stat_labels.get(z_col, z_col),
                    "Value": f"{raw:.3f}" if pd.notna(raw) else "—",
                    "Percentile": f"{int(pct)}th" if pct is not None else "—",
                })
            if stat_rows:
                st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

            with st.expander("⚙️  How your slider preset weights this player"):
                bundle_rows = []
                for bk, bundle in active_bundles.items():
                    bw = bundle_weights.get(bk, 0)
                    if bw == 0:
                        continue
                    contribution = sum(
                        view_row.get(z, 0) * (bw * internal / total_weight)
                        for z, internal in bundle["stats"].items()
                        if pd.notna(view_row.get(z)) and total_weight > 0
                    )
                    bundle_rows.append({
                        "Skill": bundle["label"],
                        "Your weight": f"{bw}",
                        "Points added": f"{contribution:+.2f}",
                    })
                if bundle_rows:
                    st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No bundles weighted — drag some sliders.")
        else:
            rows = []
            for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                raw_col = RAW_COL_MAP.get(z_col)
                z = view_row.get(z_col)
                raw = view_row.get(raw_col) if raw_col else None
                w = effective_weights.get(z_col, 0)
                contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
                pct = zscore_to_percentile(z) if pd.notna(z) else None
                rows.append({
                    "Stat": stat_labels.get(z_col, z_col),
                    "Value": f"{raw:.3f}" if pd.notna(raw) else "—",
                    "Percentile": f"{int(pct)}th" if pct is not None else "—",
                    "Weight": f"{w}",
                    "Points added": f"{contrib:+.2f}",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**Percentile profile vs. all league DEs**")
        st.caption("Solid blue = this player. Dashed gray = top-32 starter average.")
        fig = build_radar_figure(_radar_row, stat_labels, stat_methodology, benchmark=_radar_bench, benchmark_raw=_radar_bench_raw)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No radar data available.")


# ─── ⚔️ COMPARE ─────────────────────────────────────────
with tab_compare:
    from lib_shared import render_player_comparison, team_theme as _theme
    render_player_comparison(
        player_row=view_row,
        player_name=selected,
        league_df=all_des_full,
        name_col="player_display_name",
        year_choice=year_choice,
        primary_score=_view_score,
        compute_comparison_score=_de_score_of,
        radar_builder=build_radar_figure,
        benchmark=_radar_bench,
        benchmark_raw=_radar_bench_raw,
        stat_labels=stat_labels,
        stat_methodology=stat_methodology,
        key_prefix=f"de_cmp_{player.get('player_id', selected)}",
        position_label="defensive end",
        theme=_theme(player.get("recent_team") or ""),
    )


# ─── 📈 CAREER & COMBINE ────────────────────────────────
with tab_career:
    _WORKOUTS_PATH = Path(__file__).resolve().parent.parent / "data" / "college" / "nfl_all_workouts.parquet"
    render_combine_chart(
        player_name=selected,
        position="DE",
        pool_positions=["DE", "EDGE"],
        workouts_path=_WORKOUTS_PATH,
        key=f"de_combine_chart_{player.get('player_id', selected)}",
    )
    career_arc_section(
        player=player,
        league_parquet_path=DATA_PATH,
        z_score_cols=list(RAW_COL_MAP.keys()),
        stat_labels=stat_labels,
        id_col="player_id",
        name_col="player_display_name",
        position_label="defensive ends",
    )


# ─── 📅 GAME-BY-GAME SPLITS ─────────────────────────────
with tab_splits:
    from lib_splits import render_splits_section as _render_splits_section
    _render_splits_section(
        player_name=selected,
        season=selected_season,
        position_group="DE",
        key_prefix=f"de_{player.get('player_id') or selected}",
        is_career_view=_yr["is_career_view"],
    )

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    "Data via [nflverse](https://github.com/nflverse) · 2024 regular season · "
    "Compared against 90 DEs with 200+ snaps · "
    "Fan project, not affiliated with the NFL or Detroit Lions."
)

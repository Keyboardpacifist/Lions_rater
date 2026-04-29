"""
Lions Safety Rater — 2024 season
Monkey-proofed UI: every control explains WHAT it does and WHY you'd use it.
"""
import json
from pathlib import Path
import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from team_selector import get_team_and_season, filter_by_team_and_season, NFL_TEAMS
from career_arc import career_arc_section
from lib_shared import apply_algo_weights, community_section, compute_effective_weights, get_algorithm_by_slug, inject_css, render_combine_chart, render_master_detail_leaderboard, render_player_card, render_player_year_picker, score_players

st.set_page_config(page_title="Lions Safety Rater", page_icon="🏈", layout="wide", initial_sidebar_state="expanded")
inject_css()

# ── Team & Season selector ────────────────────────────────────
selected_team, selected_season = get_team_and_season()
team_name = NFL_TEAMS.get(selected_team, selected_team)

POSITION_GROUP = "safety"
PAGE_URL = "https://lions-rater.streamlit.app/Safety"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "league_s_all_seasons.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "safety_stat_metadata.json"

@st.cache_data
def load_safety_data(): return pl.read_parquet(DATA_PATH).to_pandas()
@st.cache_data
def load_safety_metadata():
    if not METADATA_PATH.exists(): return {}
    with open(METADATA_PATH) as f: return json.load(f)

RAW_COL_MAP = {
    "solo_tackle_rate_z": "solo_tackle_rate", "tackles_per_snap_z": "tackles_per_snap",
    "tfl_per_game_z": "tfl_per_game", "sacks_per_game_z": "sacks_per_game",
    "forced_fumbles_per_game_z": "forced_fumbles_per_game",
    "passes_defended_per_game_z": "passes_defended_per_game",
    "interceptions_per_game_z": "interceptions_per_game",
    # Coverage quality (PFR — added Phase 2.5)
    "targets_per_game_z": "targets_per_game",
    "completion_pct_allowed_z": "completion_pct_allowed",
    "yards_per_target_allowed_z": "yards_per_target_allowed",
    "passer_rating_allowed_z": "passer_rating_allowed",
    "avg_depth_of_target_z": "avg_depth_of_target",
    "missed_tackle_pct_z": "missed_tackle_pct",
}

BUNDLES = {
    "coverage_quality": {
        "label": "🛡️ Coverage quality",
        "description": "When targeted, what does the QB get? Catch%, Y/Tgt, passer rating allowed.",
        "why": "Modern free safeties live or die on suppression — slide this right if that's how you evaluate.",
        "stats": {
            "passer_rating_allowed_z": 0.35,
            "completion_pct_allowed_z": 0.25,
            "yards_per_target_allowed_z": 0.20,
            "passes_defended_per_game_z": 0.10,
            "interceptions_per_game_z": 0.10,
        },
    },
    "ball_skills": {
        "label": "🦅 Ball skills",
        "description": "Takeaways. INTs and PDs in volume.",
        "why": "Want safeties who turn balls into possessions? Slide right.",
        "stats": {"interceptions_per_game_z": 0.55, "passes_defended_per_game_z": 0.45},
    },
    "tackling": {
        "label": "🎯 Run support & tackling",
        "description": "Reliable tackler. Solo tackles, TFLs, tackle reliability.",
        "why": "Value safeties who play downhill and stop the run? Slide right.",
        "stats": {
            "solo_tackle_rate_z": 0.25, "tackles_per_snap_z": 0.25,
            "missed_tackle_pct_z": 0.30, "tfl_per_game_z": 0.20,
        },
    },
    "versatility": {
        "label": "💥 Versatile playmaker",
        "description": "Does he do everything? Blitzes, forces fumbles, plays all over.",
        "why": "Want safeties who can blitz, cover, and force turnovers? Slide right.",
        "stats": {"sacks_per_game_z": 0.35, "forced_fumbles_per_game_z": 0.35, "tfl_per_game_z": 0.30},
    },
}
DEFAULT_BUNDLE_WEIGHTS = {"coverage_quality": 60, "ball_skills": 40, "tackling": 40, "versatility": 30}

RADAR_STATS = list(RAW_COL_MAP.keys())
RADAR_INVERT = set()
RADAR_LABEL_OVERRIDES = {
    "solo_tackle_rate_z": "Solo tackle %", "tackles_per_snap_z": "Tackles/snap",
    "tfl_per_game_z": "TFLs", "sacks_per_game_z": "Sacks",
    "forced_fumbles_per_game_z": "Forced fumbles",
    "passes_defended_per_game_z": "Pass defense",
    "interceptions_per_game_z": "Interceptions",
    "targets_per_game_z": "Targets",
    "completion_pct_allowed_z": "Catch% allowed",
    "yards_per_target_allowed_z": "Y/Tgt allowed",
    "passer_rating_allowed_z": "Passer rating allowed",
    "avg_depth_of_target_z": "ADOT",
    "missed_tackle_pct_z": "Tackle reliability",
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
    pct_label = format_percentile(pct)
    return f"{sign}{score:.2f} ({pct_label})"

def sample_size_warning(snaps):
    if pd.isna(snaps): return ""
    if snaps < 300: return f"⚠️ Only {int(snaps)} snaps — small sample, treat with caution"
    if snaps < 500: return f"⚠️ {int(snaps)} snaps — moderate sample"
    return ""





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



def build_radar_figure(player, stat_labels, stat_methodology):
    axes, values, descriptions = [], [], []
    for z_col in RADAR_STATS:
        if z_col not in player.index: continue
        z = player.get(z_col)
        if pd.isna(z): continue
        if z_col in RADAR_INVERT: z = -z
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


if "safety_loaded_algo" not in st.session_state: st.session_state.safety_loaded_algo = None
if "upvoted_ids" not in st.session_state: st.session_state.upvoted_ids = set()
if "safety_tiers_enabled" not in st.session_state: st.session_state.safety_tiers_enabled = [1, 2]

try:
    df = load_safety_data()
except FileNotFoundError:
    st.error(f"Couldn't find data at {DATA_PATH}")
    st.stop()

# Filter to selected team and season
df = filter_by_team_and_season(df, selected_team, selected_season, team_col="team", season_col="season_year")
if len(df) == 0:
    st.warning(f"No {team_name} safeties found for {selected_season}.")
    st.stop()

meta = load_safety_metadata()
stat_tiers = meta.get("stat_tiers", {})
stat_labels = meta.get("stat_labels", {})
stat_methodology = meta.get("stat_methodology", {})

if "algo" in st.query_params and st.session_state.safety_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP:
        apply_algo_weights(linked, BUNDLES)
        st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════
st.subheader(f"{team_name} safeties")
st.markdown("What makes a great safety? **You decide.** Use the sliders on the left to tell us what you value most, and the rankings update instantly.")
st.caption(f"{selected_season} regular season · Compared to all 93 safeties league-wide with 200+ snaps")

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.header("What matters to you?")
st.sidebar.markdown("Each slider controls how much a skill affects the final score. Slide right to prioritize it, or all the way left to ignore it.")
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

if st.session_state.safety_loaded_algo:
    la = st.session_state.safety_loaded_algo
    st.sidebar.info(f"Loaded: **{la['name']}** by {la['author']}\n\n_{la.get('description', '')}_")
    if st.sidebar.button("Clear loaded algorithm"):
        st.session_state.safety_loaded_algo = None

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
                value=(tier in st.session_state.safety_tiers_enabled),
                help=TIER_DESCRIPTIONS[tier],
                key=f"safety_tier_checkbox_{tier}",
            )
            if checked:
                new_enabled.append(tier)
        else:
            st.markdown(f"<span style='opacity:0.35'>{tier_badge(tier)} {TIER_LABELS[tier]}</span>", unsafe_allow_html=True)
            st.caption("No stats available")
st.session_state.safety_tiers_enabled = new_enabled
if not new_enabled:
    st.warning("Check at least one box above to include some stats.")
    st.stop()
active_bundles = filter_bundles_by_tier(BUNDLES, stat_tiers, new_enabled)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR SLIDERS
# ══════════════════════════════════════════════════════════════
advanced_mode = False
bundle_weights = {}
effective_weights = {}

if not active_bundles:
    st.info("No stat bundles available for the selected stat types.")
    st.stop()

for bk, bundle in active_bundles.items():
    st.sidebar.markdown(f"**{bundle['label']}**")
    st.sidebar.markdown(f"{bundle['description']}")
    if f"safety_bundle_{bk}" not in st.session_state:
        st.session_state[f"safety_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
    bundle_weights[bk] = st.sidebar.slider(
        bundle["label"], 0, 100, step=5,
        key=f"safety_bundle_{bk}", label_visibility="collapsed",
        help=bundle.get("why", ""),
    )
    st.sidebar.caption(f"_↑ {bundle.get('why', '')}_")

for bk in BUNDLES:
    if bk not in bundle_weights: bundle_weights[bk] = 0
effective_weights = compute_effective_weights(active_bundles, bundle_weights)

with st.sidebar.expander("Want more control? Adjust individual stats"):
    advanced_mode = st.checkbox("Enable individual stat control", value=False, key="safety_advanced_toggle")
    if advanced_mode:
        st.caption("Set the weight of each individual stat. This overrides the bundle sliders above.")
        effective_weights = {}
        all_enabled_stats = sorted([z for z, t in stat_tiers.items() if t in new_enabled], key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))
        for z_col in all_enabled_stats:
            label = stat_labels.get(z_col, z_col)
            meth = stat_methodology.get(z_col, {})
            help_text = meth.get("what", "")
            if meth.get("limits"): help_text += f"\n\nLimits: {meth['limits']}"
            w = st.slider(f"{tier_badge(stat_tiers.get(z_col, 2))} {label}", 0, 100, 50, 5, key=f"adv_safety_{z_col}", help=help_text if help_text else None)
            if w > 0: effective_weights[z_col] = w
        bundle_weights = {bk: 0 for bk in BUNDLES}

# ══════════════════════════════════════════════════════════════
# COMPUTE SCORES & RANK
# ══════════════════════════════════════════════════════════════
players = df.copy()
if len(players) == 0:
    st.warning("No players found.")
    st.stop()
players = score_players(players, effective_weights)
total_weight = sum(effective_weights.values())
if total_weight == 0:
    st.info("All sliders are at zero — slide at least one to the right to see rankings.")
players = players.sort_values("score", ascending=False).reset_index(drop=True)
players.index = players.index + 1

# ══════════════════════════════════════════════════════════════
# RANKING TABLE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
ranked = players.copy()

# ── Master/detail click-to-detail leaderboard ──────────────────
st.markdown("""
**How to read the score:** 0.00 = avg starter · Positive = above avg starter · Negative = below.
The percentile shows where this player ranks among all qualifying safeties league-wide.
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

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "Player": ranked["player_name"],
    "Games": ranked.get("games", pd.Series([0] * len(ranked))).fillna(0).astype(int),
    "Snaps": ranked.get("def_snaps", pd.Series([0] * len(ranked))).apply(
        lambda s: f"{int(s)} ⚠️" if pd.notna(s) and s < 300 else (f"{int(s)}" if pd.notna(s) else "—")
    ),
    "Your score": ranked["score"].apply(format_score),
})

selected = render_master_detail_leaderboard(
    display_df=display_df,
    name_col="Player",
    key_prefix="s",
    team=selected_team,
    season=selected_season,
    top_banner_html=_top_html,
    top_banner_warn=_top_warn,
    leaderboard_caption=(
        "⚠️ = under 300 snaps — small sample, treat with caution. "
        "**Click any player name above** to view their profile."
    ),
)
if selected is None:
    st.stop()

player = ranked[ranked["player_name"] == selected].iloc[0]
warn = sample_size_warning(player.get("def_snaps", 0))
if warn: st.warning(warn)

# ── Unified Season picker — drives stat bar + bundle table + radar ──
all_safeties_full = load_safety_data()
player_career = all_safeties_full[all_safeties_full["player_id"] == player.get("player_id")] if "player_id" in all_safeties_full.columns else all_safeties_full[0:0]

st.markdown(f"### {selected}")

_yr = render_player_year_picker(
    career_df=player_career,
    default_season=selected_season,
    season_col="season_year",
    team_col="team",
    key_prefix=f"safety_{player.get('player_id') or selected}",
)
view_row = _yr["view_row"] if _yr["view_row"] is not None else player
year_choice = _yr["year_choice"]

if total_weight > 0:
    _view_score = sum(view_row.get(z, 0) * (w / total_weight)
                       for z, w in effective_weights.items()
                       if pd.notna(view_row.get(z)))
else:
    _view_score = float("nan")

SAFETY_STAT_SPECS = [
    ("def_snaps", "{:.0f}", "Snaps"),
    ("games", "{:.0f}", "G"),
    ("interceptions_per_game", "{:.2f}", "INT/G"),
    ("passes_defended_per_game", "{:.2f}", "PD/G"),
    ("solo_tackle_rate", "{:.1%}", "Solo Tkl%"),
    ("forced_fumbles_per_game", "{:.2f}", "FF/G"),
]
NFL_SUM_COLS = {"off_snaps", "def_snaps", "snaps", "games", "targets",
                "receptions", "rec_yards", "rec_tds",
                "attempts", "completions", "passing_yards", "passing_tds",
                "passing_interceptions", "rushing_yards", "rushing_tds",
                "carries", "rushing_attempts", "tackles", "def_tackles",
                "def_sacks", "def_qb_hits", "def_tackles_for_loss",
                "def_tackles_solo", "def_tackle_assists", "def_interceptions",
                "sacks", "tfls", "tackles_for_loss",
                "interceptions", "passes_defensed",
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

_card_narrative = None
try:
    from lib_field_viz import build_position_narrative
    _season_pool = all_safeties_full[all_safeties_full["season_year"] == selected_season]
    _card_narrative = build_position_narrative(
        player_row=view_row, peer_pool=_season_pool,
        stat_labels=stat_labels, position_label="safeties",
    )
except Exception:
    _card_narrative = None

_card_stats = [
    ("Tackles", _safe_fmt(view_row.get("def_tackles")),
                _safe_fmt(view_row.get("def_tackles_for_loss"), "{:.1f} TFL")),
    ("INT",     _safe_fmt(view_row.get("def_interceptions")),
                _safe_fmt(view_row.get("passes_defended"), "{:.0f} PD")),
    ("FF",      _safe_fmt(view_row.get("forced_fumbles")), ""),
    ("Snaps",   _safe_fmt(view_row.get("def_snaps")), ""),
]

from lib_shared import team_theme as _theme
from lib_trading_card import render_card_download_button as _render_card
_render_card(
    player_name=selected,
    position_label=(player.get("position") or "S"),
    season_str=_yr["season_str"] or f"Season {selected_season}",
    score=_view_score,
    narrative=_card_narrative,
    key_stats=_card_stats,
    player_id=player.get("player_id") or selected,
    team_abbr=_team_abbr,
    theme=_theme(_team_abbr),
    preset_name=(st.session_state.safety_loaded_algo.get("name")
                  if st.session_state.get("safety_loaded_algo") else None),
    key_prefix=f"safety_{player.get('player_id') or selected}",
    position_group="safety",
    bundle_weights=bundle_weights,
    season=(None if _yr["is_career_view"] else selected_season),
)

# ════════════════════════════════════════════════════════════════
# TABBED PLAYER DETAIL — Profile / Compare / Career & Combine / Splits
# Trading card hero stays sticky above the tabs.
# ════════════════════════════════════════════════════════════════

_radar_row = view_row if view_row is not None else player


def _safety_score_of(row):
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
            for bundle in active_bundles.values(): shown.update(bundle["stats"].keys())
            for z_col in sorted(shown, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                raw_col = RAW_COL_MAP.get(z_col)
                z = view_row.get(z_col)
                raw = view_row.get(raw_col) if raw_col else None
                pct = zscore_to_percentile(z) if pd.notna(z) else None
                stat_rows.append({"Stat": stat_labels.get(z_col, z_col), "Value": f"{raw:.3f}" if pd.notna(raw) else "—", "Percentile": f"{int(pct)}th" if pct is not None else "—"})
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
                z = view_row.get(z_col)
                raw = view_row.get(raw_col) if raw_col else None
                w = effective_weights.get(z_col, 0)
                contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
                pct = zscore_to_percentile(z) if pd.notna(z) else None
                rows.append({"Stat": stat_labels.get(z_col, z_col), "Value": f"{raw:.3f}" if pd.notna(raw) else "—", "Percentile": f"{int(pct)}th" if pct is not None else "—", "Weight": f"{w}", "Points added": f"{contrib:+.2f}"})
            if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**Percentile profile vs. all league safeties**")
        st.caption("50th = avg starter. Higher = better. (Z-scores baselined on starters at this position.)")
        fig = build_radar_figure(_radar_row, stat_labels, stat_methodology)
        if fig: st.plotly_chart(fig, use_container_width=True)


# ─── ⚔️ COMPARE ─────────────────────────────────────────
with tab_compare:
    from lib_shared import render_player_comparison, team_theme as _theme
    render_player_comparison(
        player_row=view_row,
        player_name=selected,
        league_df=all_safeties_full,
        name_col="player_display_name",
        year_choice=year_choice,
        primary_score=_view_score,
        compute_comparison_score=_safety_score_of,
        radar_builder=build_radar_figure,
        stat_labels=stat_labels,
        stat_methodology=stat_methodology,
        key_prefix=f"safety_cmp_{player.get('player_id', selected)}",
        position_label="safety",
        theme=_theme(player.get("recent_team") or ""),
    )


# ─── 📈 CAREER & COMBINE ────────────────────────────────
with tab_career:
    _WORKOUTS_PATH = Path(__file__).resolve().parent.parent / "data" / "college" / "nfl_all_workouts.parquet"
    render_combine_chart(
        player_name=selected,
        position="S",
        pool_positions=["S", "FS", "SS", "SAF"],
        workouts_path=_WORKOUTS_PATH,
        key=f"safety_combine_chart_{player.get('player_id', selected)}",
    )
    career_arc_section(
        player=player,
        league_parquet_path=DATA_PATH,
        z_score_cols=list(RAW_COL_MAP.keys()),
        stat_labels=stat_labels,
        id_col="player_id",
        name_col="player_display_name",
        position_label="safeties",
    )


# ─── 📅 GAME-BY-GAME SPLITS ─────────────────────────────
with tab_splits:
    from lib_splits import render_splits_section as _render_splits_section
    _render_splits_section(
        player_name=selected,
        season=selected_season,
        position_group="S",
        key_prefix=f"safety_{player.get('player_id') or selected}",
        is_career_view=_yr["is_career_view"],
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Data via [nflverse](https://github.com/nflverse) · 2024 regular season · Compared against 93 safeties with 200+ snaps · Fan project, not affiliated with the NFL or Detroit Lions.")

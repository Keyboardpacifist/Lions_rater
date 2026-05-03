"""
Lions Coach Rater — Coaches page
=================================
Tier-based slider UI for head coach rankings. Parallel structure to the
Receivers, Running Backs, and Offensive Line pages.

Default view:
- Active head coaches with 2+ full seasons
- Stats reflect full career as head coach

Checkboxes:
- Include 1st-year coaches (shown with 🔴 small-sample flag)
- Include recent historical coaches (retired/fired in last ~5 years)

Data expected at data/master_coaches_with_z.parquet. Required columns:
- coach_name, team, current_status ("active" / "historical"),
  seasons_as_hc, games_as_hc
- Raw stats: ats_wins, ats_losses, fourth_down_epa,
  close_game_wins, close_game_losses, record_vs_winning_wins,
  record_vs_winning_losses, pre_snap_penalty_rate, rz_td_rate,
  post_bye_wins, post_bye_losses, timeout_efficiency
- Z-scored stats: ats_win_pct_z, fourth_down_epa_z,
  close_game_win_pct_z, record_vs_winning_win_pct_z,
  pre_snap_penalty_rate_z, rz_td_rate_z, post_bye_win_pct_z,
  timeout_efficiency_z
"""

import json
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm

from lib_shared import (
    apply_algo_weights,
    community_section,
    compute_effective_weights,
    get_algorithm_by_slug,
    inject_css,
    render_master_detail_leaderboard,
    render_player_card,
    score_players,
)
from lib_top_nav import render_home_button

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Lions Coach Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

render_home_button()  # ← back to landing
POSITION_GROUP = "coach"
PAGE_URL = "https://lions-rater.streamlit.app/Coaches"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "master_coaches_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "coach_stat_metadata.json"


# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_coaches_data():
    return pl.read_parquet(DATA_PATH).to_pandas()


@st.cache_data
def load_coaches_metadata():
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH) as f:
        return json.load(f)


# ============================================================
# Stat catalog — raw column names for Player Detail display
# ============================================================
RAW_COL_MAP = {
    "ats_win_pct_z": "ats_win_pct",
    "fourth_down_epa_z": "fourth_down_epa",
    "close_game_win_pct_z": "close_game_win_pct",
    "record_vs_winning_win_pct_z": "record_vs_winning_win_pct",
    "pre_snap_penalty_rate_z": "pre_snap_penalty_rate",
    "rz_td_rate_z": "rz_td_rate",
    "post_bye_win_pct_z": "post_bye_win_pct",
    "timeout_efficiency_z": "timeout_efficiency",
}


# ============================================================
# Bundles
# ============================================================
BUNDLES = {
    "decision_making": {
        "label": "⚡ Decision-making",
        "description": "Smart 4th-down calls and clock management.",
        "why": "Think in-game decisions define a great head coach? Crank this up.",
        "stats": {
            "fourth_down_epa_z": 0.60,
            "timeout_efficiency_z": 0.40,
        },
    },
    "situational": {
        "label": "💪 Situational performance",
        "description": "Comes through in close games, beats good teams, uses bye weeks well.",
        "why": "Value coaches who win the games that matter most? Slide right.",
        "stats": {
            "close_game_win_pct_z": 0.40,
            "record_vs_winning_win_pct_z": 0.40,
            "post_bye_win_pct_z": 0.20,
        },
    },
    "discipline": {
        "label": "🎯 Discipline",
        "description": "Team avoids unforced errors. Low pre-snap penalties.",
        "why": "Think well-coached teams don't beat themselves? Slide right.",
        "stats": {
            "pre_snap_penalty_rate_z": 1.0,
        },
    },
    "efficiency": {
        "label": "📊 Efficiency vs. expectations",
        "description": "Beats the spread and cashes in red zone trips.",
        "why": "Want coaches who outperform what Vegas expects? Slide right.",
        "stats": {
            "ats_win_pct_z": 0.60,
            "rz_td_rate_z": 0.40,
        },
    },
}

DEFAULT_BUNDLE_WEIGHTS = {
    "decision_making": 60,
    "situational": 50,
    "discipline": 30,
    "efficiency": 50,
}


# ============================================================
# Radar chart config — all 8 stats
# ============================================================
# Pre-snap penalty rate is inverted on the radar so higher = fewer penalties.
RADAR_STATS = [
    "ats_win_pct_z",
    "fourth_down_epa_z",
    "close_game_win_pct_z",
    "record_vs_winning_win_pct_z",
    "pre_snap_penalty_rate_z",
    "rz_td_rate_z",
    "post_bye_win_pct_z",
    "timeout_efficiency_z",
]

# Stats where the z-score needs to be flipped for the radar
# (high raw value = bad, so we invert so high = good on the chart)
RADAR_INVERT = {"pre_snap_penalty_rate_z"}

# Custom radar axis labels — override default stat labels for clarity
RADAR_LABEL_OVERRIDES = {
    "pre_snap_penalty_rate_z": "Discipline",
    "ats_win_pct_z": "ATS record",
    "fourth_down_epa_z": "4th-down EPA",
    "close_game_win_pct_z": "Close games",
    "record_vs_winning_win_pct_z": "vs. winning teams",
    "rz_td_rate_z": "Red zone TD %",
    "post_bye_win_pct_z": "Post-bye record",
    "timeout_efficiency_z": "Timeout efficiency",
}


def zscore_to_percentile(z):
    """Convert a z-score to a 0-100 percentile via the normal CDF."""
    if pd.isna(z):
        return None
    return float(norm.cdf(z) * 100)


def build_radar_figure(coach, stat_labels, stat_methodology):
    """Return a Plotly polar figure showing this coach's percentiles
    on the RADAR_STATS axes. Missing values are skipped.
    Stats in RADAR_INVERT have their z-score sign flipped before
    conversion to percentile.
    Hovering a data point shows the stat's 'what' description."""
    axes = []
    values = []
    descriptions = []
    for z_col in RADAR_STATS:
        if z_col not in coach.index:
            continue
        z = coach.get(z_col)
        if pd.isna(z):
            continue
        if z_col in RADAR_INVERT:
            z = -z
        pct = zscore_to_percentile(z)
        label = RADAR_LABEL_OVERRIDES.get(z_col, stat_labels.get(z_col, z_col))
        desc = stat_methodology.get(z_col, {}).get("what", "")
        if z_col in RADAR_INVERT:
            desc = f"{desc} (Higher on chart = fewer penalties.)"
        axes.append(label)
        values.append(pct)
        descriptions.append(desc)

    if not axes:
        return None

    axes_closed = axes + [axes[0]]
    values_closed = values + [values[0]]
    descriptions_closed = descriptions + [descriptions[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=axes_closed,
        customdata=descriptions_closed,
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.25)",
        line=dict(color="rgba(31, 119, 180, 0.9)", width=2),
        marker=dict(size=6, color="rgba(31, 119, 180, 1)"),
        hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<br><br><i>%{customdata}</i><extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                ticktext=["25", "50", "75", "100"],
                tickfont=dict(size=9, color="#888"),
                gridcolor="#ddd",
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                gridcolor="#ddd",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=20, b=20),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ============================================================
# Tier helpers
# ============================================================
TIER_LABELS = {
    1: "Counting stats",
    2: "Rate stats",
    3: "Modeled stats",
    4: "Estimated stats",
}
TIER_DESCRIPTIONS = {
    1: "Raw totals — sacks, tackles, yards, touchdowns.",
    2: "Per-game and per-snap averages that adjust for playing time.",
    3: "Stats adjusted for expected performance based on a model.",
    4: "Inferred from limited data — least reliable. Use with caution.",
}


def tier_badge(tier: int) -> str:
    return {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠"}.get(tier, "⚪")


def filter_bundles_by_tier(bundles: dict, stat_tiers: dict, enabled_tiers: list) -> dict:
    filtered = {}
    for bk, bdef in bundles.items():
        kept_stats = {
            z: w for z, w in bdef["stats"].items()
            if stat_tiers.get(z, 2) in enabled_tiers
        }
        if kept_stats:
            filtered[bk] = {
                "label": bdef["label"],
                "description": bdef["description"],
                "stats": kept_stats,
            }
    return filtered


def bundle_tier_summary(bundle_stats: dict, stat_tiers: dict) -> str:
    counts = {}
    for z in bundle_stats:
        t = stat_tiers.get(z, 2)
        counts[t] = counts.get(t, 0) + 1
    return " ".join(f"{tier_badge(t)}×{c}" for t, c in sorted(counts.items()))


# ============================================================
# Score labels
# ============================================================
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


def sample_size_badge(games: float) -> str:
    """Badge based on absolute games-as-HC count.
    🔴 severe (<34 games = less than 2 full seasons),
    🟡 caution (34–51 games = 2–3 seasons),
    '' otherwise (3+ seasons)."""
    if pd.isna(games):
        return ""
    if games < 34:
        return "🔴"
    if games < 51:
        return "🟡"
    return ""


def sample_size_caption(games: float) -> str:
    if pd.isna(games):
        return ""
    if games < 34:
        return f"⚠️ Very short tenure: {int(games)} games as head coach. Treat as directional only."
    if games < 51:
        return f"⚠️ Short tenure: {int(games)} games as head coach. Score may be noisy."
    return ""


SCORE_EXPLAINER = """
**What this number means.** The score is a weighted average of z-scores —
standardized stats where 0 is the coach-group average, +1 is one standard
deviation above, and −1 is one standard deviation below. Your slider
weights control how much each bundle contributes.

**How to read it:**
- `+1.0` or higher → well above the group average on what you weighted
- `+0.4` to `+1.0` → above average
- `−0.4` to `+0.4` → roughly average
- `−1.0` or lower → well below average

**What this is not.** It's not a PFF-style grade. It's a **comparative**
number telling you how head coaches stack up against each other under
the methodology *you* chose.

**Coach population:** z-scores are computed against active head coaches
with at least 2 full seasons (34+ games). Stats reflect each coach's
full head-coaching career. First-year coaches (if shown) are flagged
🔴 and have extremely noisy scores — treat as directional.

**Coach attribution caveats.** Football is a team game. "Credit" for a
win or a 4th-down decision is shared between the coach, the coordinators,
the roster, and the players. These stats try to isolate coaching signal,
but they can't do so perfectly. Treat big gaps as meaningful and small
gaps as noise.
"""


# ============================================================
# Session state
# ============================================================
if "coach_loaded_algo" not in st.session_state:
    st.session_state.coach_loaded_algo = None
if "upvoted_ids" not in st.session_state:
    st.session_state.upvoted_ids = set()
if "coach_tiers_enabled" not in st.session_state:
    st.session_state.coach_tiers_enabled = [1, 2, 3]  # Tier 4 off by default


# ============================================================
# Header
# ============================================================
st.title("🦁 Lions Coach Rater")
st.markdown(
    "What makes a great player? **You decide.** Drag the sliders to weight what you value, "
    "and watch the head coaches re-rank in real time. "
    "_No 'best coach' — just **your** best coach._"
)
st.caption(
    "Full career stats • Compared against active head coaches with 2+ full seasons • "
    "Every stat has a methodology popover"
)


# ============================================================
# Load data
# ============================================================
try:
    df = load_coaches_data()
except FileNotFoundError:
    st.error(f"Couldn't find the coaches data file at {DATA_PATH}.")
    st.caption(
        "Run the coach data-pull script and upload the parquet + metadata "
        "files to `data/` in the repo."
    )
    st.stop()

meta = load_coaches_metadata()
stat_tiers = meta.get("stat_tiers", {})
stat_labels = meta.get("stat_labels", {})
stat_methodology = meta.get("stat_methodology", {})


# ============================================================
# ?algo= deep link
# ============================================================
if "algo" in st.query_params and st.session_state.coach_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP:
        apply_algo_weights(linked, BUNDLES)
        st.rerun()


# ============================================================
# Sidebar — filters
# ============================================================
st.sidebar.header("What matters to you?")

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle(
    "🔬 Advanced mode", value=False,
    help="Show individual stat sliders with methodology tooltips instead of plain-English bundles.",
)

st.sidebar.markdown("Each slider controls how much a skill affects the final score. Slide right to prioritize, left to ignore.")

if st.session_state.coach_loaded_algo:
    la = st.session_state.coach_loaded_algo
    st.sidebar.info(
        f"Loaded: **{la['name']}** by {la['author']}\n\n"
        f"_{la.get('description', '')}_"
    )
    if st.sidebar.button("Clear loaded algorithm"):
        st.session_state.coach_loaded_algo = None


# ============================================================
# Tier filter (main content area)
# ============================================================
# HIDDEN 2026-05-03 — tier-checkbox UI; defaults
# applied via session_state read below.
if False:
    st.markdown("### Which stats should count?")
    st.caption(
        "Check more boxes to include more types of stats. More boxes = more data, but less certainty."
    )

    tier_cols = st.columns(4)
    new_enabled = []
    for i, tier in enumerate([1, 2, 3, 4]):
        with tier_cols[i]:
            checked = st.checkbox(
                f"{tier_badge(tier)} {TIER_LABELS[tier]}",
                value=(tier in st.session_state.coach_tiers_enabled),
                help=TIER_DESCRIPTIONS[tier],
                key=f"coach_tier_checkbox_{tier}",
            )
            if checked:
                new_enabled.append(tier)

new_enabled = list(
    st.session_state.get(
        "coach_tiers_enabled", [1, 2])
) or [1, 2]
st.session_state.coach_tiers_enabled = new_enabled

if not new_enabled:
    st.warning("Enable at least one tier to see ratings.")
    st.stop()

active_bundles = filter_bundles_by_tier(BUNDLES, stat_tiers, new_enabled)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ============================================================
# Sliders
# ============================================================
bundle_weights = {}
effective_weights = {}

if not advanced_mode:
    if not active_bundles:
        st.info("No bundles have stats in the enabled tiers. Try enabling more tiers.")
        st.stop()

    st.sidebar.markdown("Each slider controls how much a skill affects the final score. Slide right to prioritize, left to ignore.")

    for bk, bundle in active_bundles.items():
        tier_summary = bundle_tier_summary(bundle["stats"], stat_tiers)
        st.sidebar.markdown(f"**{bundle['label']}**")
        st.sidebar.markdown(
            f"<div class='bundle-desc'>{bundle['description']}<br>"
            f"<small>{tier_summary}</small></div>",
            unsafe_allow_html=True,
        )
        if f"coach_bundle_{bk}" not in st.session_state:
            st.session_state[f"coach_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(
            bundle["label"], 0, 100,
            step=5,
            key=f"coach_bundle_{bk}",
            label_visibility="collapsed",
        )

    for bk in BUNDLES:
        if bk not in bundle_weights:
            bundle_weights[bk] = 0

    effective_weights = compute_effective_weights(active_bundles, bundle_weights)

else:
    st.sidebar.caption(
        "Direct control over every underlying stat. Hover the ⓘ icon next to "
        "each slider for methodology."
    )
    st.sidebar.markdown(
        "<div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#888;margin-bottom:-0.5rem'>"
        "<span>\u2190 Low priority</span><span>High priority \u2192</span></div>",
        unsafe_allow_html=True,
    )

    all_enabled_stats = [
        z for z, t in stat_tiers.items() if t in new_enabled
    ]
    all_enabled_stats.sort(key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))

    for z_col in all_enabled_stats:
        tier = stat_tiers.get(z_col, 2)
        label = stat_labels.get(z_col, z_col)
        meth = stat_methodology.get(z_col, {})

        help_parts = []
        if meth.get("what"):
            help_parts.append(f"What: {meth['what']}")
        if meth.get("how"):
            help_parts.append(f"How: {meth['how']}")
        if meth.get("limits"):
            help_parts.append(f"Limits: {meth['limits']}")
        help_text = "\n\n".join(help_parts) if help_parts else None

        w = st.sidebar.slider(
            f"{tier_badge(tier)} {label}",
            min_value=0, max_value=100, value=50, step=5,
            key=f"adv_coach_{z_col}",
            help=help_text,
        )
        if w > 0:
            effective_weights[z_col] = w

    bundle_weights = {bk: 0 for bk in BUNDLES}


# ============================================================
# Filter the coach population
# ============================================================
# Default: active coaches with 2+ full seasons (34+ games)
# Checkboxes let user broaden this.


c_opts = st.columns(2)
with c_opts[0]:
    include_rookies = st.checkbox(
        "Include 1st-year coaches",
        value=False,
        key="coach_include_rookies",
        help="Adds coaches with less than 2 full seasons. Flagged 🔴 for small sample.",
    )
with c_opts[1]:
    include_historical = st.checkbox(
        "Include recent historical coaches",
        value=False,
        key="coach_include_historical",
        help="Adds recently retired/fired coaches for historical comparison.",
    )

coaches = df.copy()

# Always include active with 2+ seasons as the base
keep_mask = (coaches["current_status"] == "active") & (coaches["games_as_hc"].fillna(0) >= 34)

if include_rookies:
    keep_mask = keep_mask | (
        (coaches["current_status"] == "active") & (coaches["games_as_hc"].fillna(0) < 34)
    )

if include_historical:
    keep_mask = keep_mask | (coaches["current_status"] == "historical")

coaches = coaches[keep_mask].copy()

if len(coaches) == 0:
    st.warning("No coaches match the current filters.")
    st.stop()


# ============================================================
# Score
# ============================================================
coaches = score_players(coaches, effective_weights)

total_weight = sum(effective_weights.values())
if total_weight == 0:
    st.info("All weights are zero — drag some sliders to start ranking.")

coaches = coaches.sort_values("score", ascending=False).reset_index(drop=True)
coaches.index = coaches.index + 1


# ============================================================
# Ranking
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Ranking")

# Hide-small-samples checkbox
hide_small = st.checkbox(
    "Hide coaches with severe small samples (<2 full seasons)",
    value=False,
    key="coach_hide_small",
    help="Hides red-flagged coaches. Yellow-flagged coaches still show with a caution.",
)

ranked = coaches.copy()
if hide_small:
    ranked = ranked[ranked["games_as_hc"].fillna(0) >= 34].copy()
    if len(ranked) == 0:
        st.warning("All coaches are below the 2-season threshold. Uncheck the filter to see them.")
        st.stop()
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1

# Top-ranked highlight banner — built up-front so we can pass it
# into the master/detail leaderboard helper.
_top_html = None
_top_warn = None
if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top.get("coach_name", "—")
    top_team = top.get("team", "")
    top_score = top["score"]
    top_games = top.get("games_as_hc", 0)
    badge = sample_size_badge(top_games)
    sign = "+" if top_score >= 0 else ""
    team_part = f" ({top_team})" if top_team else ""
    _top_html = (
        f"<div style='background:#0076B6;color:white;padding:14px 20px;"
        f"border-radius:8px;margin-bottom:8px;font-size:1.1rem;'>"
        f"<span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span>"
        f" &nbsp;·&nbsp; <strong>{top_name}</strong>{team_part} {badge}"
        f" &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{sign}{top_score:.2f}</span>"
        f" <span style='opacity:0.85;'>({format_percentile(zscore_to_percentile(top_score))})</span>"
        f"</div>"
    )
    _top_warn = sample_size_caption(top_games)

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "": ranked["games_as_hc"].apply(sample_size_badge),
    "Coach": ranked["coach_name"],
    "Team": ranked.get("team", pd.Series(["—"] * len(ranked))),
    "Seasons": ranked["seasons_as_hc"].fillna(0).astype(int),
    "Games": ranked["games_as_hc"].fillna(0).astype(int),
    "Your score": ranked["score"].apply(format_score),
})

selected = render_master_detail_leaderboard(
    display_df=display_df,
    name_col="Coach",
    key_prefix="coach",
    team=("hide" if hide_small else "all"),
    season=0,
    top_banner_html=_top_html,
    top_banner_warn=_top_warn,
    leaderboard_caption=(
        "⚠️ Coaches with short tenures have noisier scores — extreme values "
        "reflect small sample sizes. "
        "🔴 = severe small sample (<2 full seasons), 🟡 = caution (2–3 seasons). "
        "**Click any coach name** above to view their full card."
    ),
)
with st.expander("ℹ️ How is this score calculated?"):
    st.markdown(SCORE_EXPLAINER)

if selected is None:
    st.stop()  # browse mode — leaderboard rendered, no detail to show


# ============================================================
# Coach detail
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

coach = ranked[ranked["coach_name"] == selected].iloc[0]

# Sample size warning for the selected coach
warn = sample_size_caption(coach.get("games_as_hc", 0))
if warn:
    st.warning(warn)

# ── Trading-card visual (coach variant) ────────────────────────
HC_STAT_SPECS = [
    ("seasons_as_hc", "{:.0f}", "Seasons"),
    ("games_as_hc", "{:.0f}", "Games"),
    ("wins", "{:.0f}", "Wins"),
    ("losses", "{:.0f}", "Losses"),
    ("close_game_win_pct", "{:.1%}", "Close W%"),
    ("ats_win_pct", "{:.1%}", "ATS W%"),
]
_first = coach.get("first_season")
_last = coach.get("last_season")
_tenure = (f"{int(_first)}–{int(_last)}"
           if pd.notna(_first) and pd.notna(_last) else "")
_team_abbr = coach.get("team") if pd.notna(coach.get("team")) else None
render_player_card(
    player_name=selected,
    position_label="HEAD COACH",
    team_abbr=_team_abbr,
    season_str=_tenure,
    score=coach.get("score"),
    stat_specs=HC_STAT_SPECS,
    view_row=coach,
)

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("**How your score breaks down**")

    if not advanced_mode:
        bundle_rows = []
        for bk, bundle in active_bundles.items():
            bw = bundle_weights.get(bk, 0)
            if bw == 0:
                continue
            contribution = 0.0
            for z_col, internal in bundle["stats"].items():
                z = coach.get(z_col)
                if pd.notna(z) and total_weight > 0:
                    contribution += z * (bw * internal / total_weight)
            bundle_rows.append({
                "Skill": bundle["label"],
                "Your weight": f"{bw}",
                "Points added": f"{contribution:+.2f}",
            })
        if bundle_rows:
            st.dataframe(pd.DataFrame(bundle_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No bundles weighted — drag some sliders.")

        with st.expander("🔬 See the underlying stats"):
            stat_rows = []
            shown_stats = set()
            for bundle in active_bundles.values():
                shown_stats.update(bundle["stats"].keys())
            for z_col in sorted(shown_stats, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                tier = stat_tiers.get(z_col, 2)
                label = stat_labels.get(z_col, z_col)
                raw_col = RAW_COL_MAP.get(z_col)
                z = coach.get(z_col)
                raw = coach.get(raw_col) if raw_col else None
                stat_rows.append({
                    "Tier": tier_badge(tier),
                    "Stat": label,
                    "Raw": f"{raw:.3f}" if pd.notna(raw) else "—",
                    "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                })
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    else:
        st.caption("Stat-by-stat breakdown (z-score vs coach group)")
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            tier = stat_tiers.get(z_col, 2)
            label = stat_labels.get(z_col, z_col)
            raw_col = RAW_COL_MAP.get(z_col)
            z = coach.get(z_col)
            raw = coach.get(raw_col) if raw_col else None
            w = effective_weights.get(z_col, 0)
            contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            rows.append({
                "Tier": tier_badge(tier),
                "Stat": label,
                "Raw": f"{raw:.3f}" if pd.notna(raw) else "—",
                "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                "Weight": f"{w}",
                "Points added": f"{contrib:+.2f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No stats weighted — drag some sliders.")

with c2:
    st.markdown("**Coach profile** (percentiles vs. coach group)")
    fig = build_radar_figure(coach, stat_labels, stat_methodology)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No radar data available for this coach.")
    st.caption(
        "Each axis shows where this coach ranks among the group. "
        "50 = median. The 'Discipline' axis is inverted — higher = fewer pre-snap penalties. "
        "Hover any data point for the stat description."
    )


# ============================================================
# Community algorithms
# ============================================================
community_section(
    position_group=POSITION_GROUP,
    bundles=BUNDLES,
    bundle_weights=bundle_weights,
    advanced_mode=advanced_mode,
    page_url=PAGE_URL,
)


# ============================================================
# Footer
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption(
    "Data via [nflverse](https://github.com/nflverse) • "
    "FTN charting via FTN Data via nflverse (CC-BY-SA 4.0) • "
    "Built as a fan project, not affiliated with the NFL or the Detroit Lions."
)

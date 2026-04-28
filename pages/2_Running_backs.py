"""
Lions Running Back Rater — RB page (tier migration)
===================================================
Tier-based slider UI for RB rankings, matching the Receivers and OL pages.

What the tier system does:
- Loads stat tiers and methodology from data/rb_stat_metadata.json.
- Tier checkboxes at the top of the page let users filter which stats
  participate in scoring. Tier 4 off by default; there are no Tier 4
  stats for RBs, so that checkbox is a no-op for now but we keep
  it for consistency with WR and OL.
- When a tier is disabled, any stat in that tier is removed from every
  bundle. Bundles that end up empty disappear from the sidebar.
- Advanced mode shows per-stat sliders with methodology in help tooltips.
- Leaderboard scores carry a label like "+0.47 (above group)".
- "How is this score calculated?" expander below the leaderboard.

Design note: Tier 1 raw counts (rush_yards_z, rush_tds_z, carries_z,
receptions_z, rec_yards_z, rec_tds_z) are NOT added to any existing
bundle. Adding raw volume to "Efficiency" or "Tackle breaking" would
break the bundle's meaning. Tier 1 stats are still accessible — they
show up in Advanced mode. Bundle mode keeps its clean original design.
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
from lib_shared import (
    apply_algo_weights,
    community_section,
    compute_effective_weights,
    get_algorithm_by_slug,
    inject_css,
    metric_picker,
    radar_season_row,
    render_combine_chart,
    render_master_detail_leaderboard,
    render_player_card,
    render_player_year_picker,
    score_players,
)

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Lions Running Back Rater",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Team & Season selector ────────────────────────────────────
selected_team, selected_season = get_team_and_season()
team_name = NFL_TEAMS.get(selected_team, selected_team)

POSITION_GROUP = "rb"
PAGE_URL = "https://lions-rater.streamlit.app/Running_backs"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "league_rb_all_seasons.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "rb_stat_metadata.json"


# ============================================================
# Data loading
# ============================================================
@st.cache_data
def load_rb_data():
    return pl.read_parquet(DATA_PATH).to_pandas()


@st.cache_data
def load_rb_metadata():
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH) as f:
        return json.load(f)


# ============================================================
# Stat catalog — raw column names for Player Detail display
# ============================================================
RAW_COL_MAP = {
    "rush_yards_z": "rush_yards",
    "rush_tds_z": "rush_tds",
    "carries_z": "carries",
    "receptions_z": "receptions",
    "rec_yards_z": "rec_yards",
    "rec_tds_z": "rec_tds",
    "yards_per_carry_z": "yards_per_carry",
    "rush_success_rate_z": "rush_success_rate",
    "carries_per_game_z": "carries_per_game",
    "snap_share_z": "snap_share",
    "touches_per_game_z": "touches_per_game",
    "targets_per_game_z": "targets_per_game",
    "explosive_run_rate_z": "explosive_run_rate",
    "explosive_15_rate_z": "explosive_15_rate",
    "rz_carry_share_z": "rz_carry_share",
    "goal_line_td_rate_z": "goal_line_td_rate",
    "short_yardage_conv_rate_z": "short_yardage_conv_rate",
    "rec_yards_per_target_z": "rec_yards_per_target",
    "yac_per_reception_z": "yac_per_reception",
    "broken_tackles_per_att_z": "broken_tackles_per_att",
    "yards_before_contact_per_att_z": "yards_before_contact_per_att",
    "yards_after_contact_per_att_z": "yards_after_contact_per_att",
    "epa_per_rush_z": "epa_per_rush",
    "rec_epa_per_target_z": "rec_epa_per_target",
    "ryoe_per_att_z": "ryoe_per_att",
}


# ============================================================
# Bundles — Tier 2/3 organized, unchanged from previous version
# ============================================================
BUNDLES = {
    "efficiency": {
        "label": "⚡ Efficiency",
        "description": "Productive on a per-carry basis. Doesn't waste touches.",
        "why": "Think the best RBs make the most of every carry? Crank this up.",
        "stats": {
            "yards_per_carry_z": 0.25,
            "epa_per_rush_z": 0.35,
            "rush_success_rate_z": 0.20,
            "ryoe_per_att_z": 0.20,
        },
    },
    "tackle_breaking": {
        "label": "💪 Tackle breaking",
        "description": "Makes defenders miss and grinds out yards after contact.",
        "why": "Value backs who create yards on their own after the line does its job? Slide right.",
        "stats": {
            "broken_tackles_per_att_z": 0.40,
            "yards_after_contact_per_att_z": 0.45,
            "yards_before_contact_per_att_z": 0.15,
        },
    },
    "explosive": {
        "label": "💥 Explosive plays",
        "description": "Hits the home run. Big-play threat every carry.",
        "why": "Want the guy who can take it to the house on any play? Slide right.",
        "stats": {
            "explosive_run_rate_z": 0.50,
            "explosive_15_rate_z": 0.50,
        },
    },
    "volume": {
        "label": "📊 Volume & usage",
        "description": "Workhorse. The offense runs through him.",
        "why": "Think the best RB is the one who carries the load? Slide right.",
        "stats": {
            "carries_per_game_z": 0.35,
            "snap_share_z": 0.30,
            "touches_per_game_z": 0.35,
        },
    },
    "receiving": {
        "label": "🤲 Receiving back",
        "description": "Dual threat out of the backfield as a pass catcher.",
        "why": "Value backs who can line up as a receiver and win in the passing game? Slide right.",
        "stats": {
            "rec_yards_per_target_z": 0.25,
            "yac_per_reception_z": 0.20,
            "targets_per_game_z": 0.30,
            "rec_epa_per_target_z": 0.25,
        },
    },
    "short_yardage": {
        "label": "🎯 Short yardage & goal line",
        "description": "Gets the tough yards when the team needs them most.",
        "why": "Think the best ability is converting on 3rd-and-1 and punching it in at the goal line? Slide right.",
        "stats": {
            "short_yardage_conv_rate_z": 0.50,
            "goal_line_td_rate_z": 0.30,
            "rz_carry_share_z": 0.20,
        },
    },
}

DEFAULT_BUNDLE_WEIGHTS = {
    "efficiency": 70,
    "tackle_breaking": 50,
    "explosive": 40,
    "volume": 60,
    "receiving": 30,
    "short_yardage": 30,
}


# ============================================================
# Radar chart config — 8 headline stats, fixed across users
# ============================================================
# Mix of Tier 2 rates and Tier 3 modeled stats. Excludes Tier 1 raw
# counts — those skew the polygon based on volume, not skill profile.
RADAR_STATS = [
    "yards_per_carry_z",              # efficiency
    "rush_success_rate_z",            # consistency
    "broken_tackles_per_att_z",       # elusiveness
    "yards_after_contact_per_att_z",  # toughness
    "explosive_run_rate_z",           # big play ability
    "epa_per_rush_z",                 # modeled efficiency
    "ryoe_per_att_z",                 # rushing over expected
    "rec_yards_per_target_z",         # receiving threat
]


def zscore_to_percentile(z):
    """Convert a z-score to a 0-100 percentile via the normal CDF."""
    if pd.isna(z):
        return None
    return float(norm.cdf(z) * 100)


# Per-stat raw value formatting for the radar benchmark hover.
_RADAR_RAW_FORMATTERS = {
    "yards_per_carry_z": ("YPC", lambda v: f"{v:.2f}"),
    "rush_success_rate_z": ("success rate", lambda v: f"{v*100:.1f}%"),
    "broken_tackles_per_att_z": ("brk tkl/att", lambda v: f"{v:.2f}"),
    "yards_after_contact_per_att_z": ("YACO/att", lambda v: f"{v:.2f}"),
    "explosive_run_rate_z": ("expl run rate", lambda v: f"{v*100:.1f}%"),
    "epa_per_rush_z": ("EPA/rush", lambda v: f"{v:+.2f}"),
    "ryoe_per_att_z": ("RYOE/att", lambda v: f"{v:+.2f}"),
    "rec_yards_per_target_z": ("rec yds/tgt", lambda v: f"{v:.1f}"),
}

def _format_radar_raw(z_col, raw_value):
    if raw_value is None or pd.isna(raw_value):
        return ""
    spec = _RADAR_RAW_FORMATTERS.get(z_col)
    if spec is None:
        return f"{raw_value:.2f}"
    label, fmt = spec
    return f"{label}: {fmt(raw_value)}"


def build_radar_figure(player, stat_labels, stat_methodology,
                        benchmark=None, benchmark_raw=None,
                        benchmark_label="Top 32 starter avg"):
    """Return a Plotly polar figure showing this player's percentiles
    on the 8 RADAR_STATS axes. Missing values are skipped.
    Hovering over a data point shows the stat's 'what' description
    from the methodology metadata.

    If `benchmark` is a dict {z_col: mean_z}, also draw a dashed reference
    polygon (e.g., the top-32-RBs-by-snaps mean) for at-a-glance contrast.
    `benchmark_raw` provides raw means so hover shows actual stat values."""
    axes = []
    values = []
    descriptions = []
    bench_values = []
    bench_raw_strs = []
    for z_col in RADAR_STATS:
        if z_col not in player.index:
            continue
        z = player.get(z_col)
        if pd.isna(z):
            continue
        pct = zscore_to_percentile(z)
        label = stat_labels.get(z_col, z_col).replace(" (raw)", "")
        desc = stat_methodology.get(z_col, {}).get("what", "")
        axes.append(label)
        values.append(pct)
        descriptions.append(desc)
        if benchmark is not None:
            bz = benchmark.get(z_col)
            bench_values.append(zscore_to_percentile(bz) if bz is not None and pd.notna(bz) else None)
            raw_v = benchmark_raw.get(z_col) if benchmark_raw else None
            bench_raw_strs.append(_format_radar_raw(z_col, raw_v))

    if not axes:
        return None

    # Close the polygon by repeating the first point at the end
    axes_closed = axes + [axes[0]]
    values_closed = values + [values[0]]
    descriptions_closed = descriptions + [descriptions[0]]

    fig = go.Figure()

    # Player polygon FIRST so the benchmark layers on top — its diamond
    # markers stay hoverable even where polygons overlap.
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=axes_closed,
        customdata=descriptions_closed,
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.25)",
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
            bench_hover.append(
                f"<b>{ax}</b><br>{benchmark_label}<br>{extra}{pct:.0f}th percentile"
            )
        bench_hover.append(bench_hover[0])
        fig.add_trace(go.Scatterpolar(
            r=bv_clean + [bv_clean[0]],
            theta=axes_closed,
            mode="lines+markers",
            line=dict(color="rgba(102, 102, 102, 0.9)", width=2, dash="dot"),
            marker=dict(size=10, color="rgba(102, 102, 102, 0.95)",
                        symbol="diamond", line=dict(width=2, color="white")),
            name=benchmark_label,
            hovertext=bench_hover,
            hoverinfo="text",
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
        showlegend=(benchmark is not None),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.7)", bordercolor="#ccc", borderwidth=1,
                    font=dict(size=10)),
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
    """Strip disabled-tier stats out of each bundle. Empty bundles drop out."""
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


def sample_size_badge(pct: float) -> str:
    """Return an emoji badge for sample size as a % of group leader.
    🔴 severe (<20%), 🟡 caution (20–50%), '' otherwise."""
    if pd.isna(pct):
        return ""
    if pct < 20:
        return "🔴"
    if pct < 50:
        return "🟡"
    return ""


def sample_size_caption(pct: float) -> str:
    """Plain-English caption for sample size warning, or empty string."""
    if pd.isna(pct):
        return ""
    if pct < 20:
        return f"⚠️ Severe small sample: {pct:.0f}% of group leader's carries. Treat as directional only."
    if pct < 50:
        return f"⚠️ Small sample: {pct:.0f}% of group leader's carries. Score may be noisy."
    return ""


SCORE_EXPLAINER = """
**What this number means.** The score is a weighted average of z-scores —
standardized stats where 0 is the avg starter (z-scored against the position's starter pool), +1 is one standard
deviation above, and −1 is one standard deviation below. Your slider
weights control how much each bundle contributes.

**How to read it:**
- `+1.0` or higher → well above the avg starter on what you weighted
- `+0.4` to `+1.0` → above average
- `−0.4` to `+0.4` → roughly average
- `−1.0` or lower → well below average

**What this is not.** It's not a PFF-style 0-100 grade. It's a
**comparative** number telling you how this team's running backs stack
up against the league's RBs, under the methodology *you* chose.

**League population:** z-scores are computed against all RBs with
100+ offensive snaps (min 6 games played). Every team's RBs with at
least one snap are visible, but players with very few carries will
have noisy scores — read extreme values on low-volume players as
"small sample, not skill."
"""


# ============================================================
# Session state
# ============================================================
if "rb_loaded_algo" not in st.session_state:
    st.session_state.rb_loaded_algo = None
if "upvoted_ids" not in st.session_state:
    st.session_state.upvoted_ids = set()
if "rb_tiers_enabled" not in st.session_state:
    st.session_state.rb_tiers_enabled = [1, 2, 3]  # Tier 4 off by default


# ============================================================
# Header
# ============================================================
st.subheader("Running backs")


# ============================================================
# Load data
# ============================================================
try:
    df = load_rb_data()
except FileNotFoundError:
    st.error("Couldn't find the running backs data file.")
    st.stop()

# Filter to selected team and season
df = filter_by_team_and_season(df, selected_team, selected_season, team_col="recent_team", season_col="season_year")
if len(df) == 0:
    st.warning(f"No {team_name} running backs found for {selected_season}.")
    st.stop()

meta = load_rb_metadata()
stat_tiers = meta.get("stat_tiers", {})
stat_labels = meta.get("stat_labels", {})
stat_methodology = meta.get("stat_methodology", {})


# ============================================================
# ?algo= deep link
# ============================================================
if "algo" in st.query_params and st.session_state.rb_loaded_algo is None:
    linked = get_algorithm_by_slug(st.query_params["algo"])
    if linked and linked.get("position_group") == POSITION_GROUP:
        apply_algo_weights(linked, BUNDLES)
        st.rerun()


# ============================================================
# Sidebar — filters
# ============================================================
st.sidebar.header("What matters to you?")

min_carries = st.sidebar.slider(
    "Minimum carries", 0, 300, 20, step=5,
    help="Hide backs who barely touched the ball.",
)

st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle(
    "🔬 Advanced mode", value=False,
    help="Show individual stat sliders with methodology tooltips instead of plain-English bundles.",
)

st.sidebar.markdown("Each slider controls how much a skill affects the final score. Slide right to prioritize, left to ignore.")

if st.session_state.rb_loaded_algo:
    la = st.session_state.rb_loaded_algo
    st.sidebar.info(
        f"Loaded: **{la['name']}** by {la['author']}\n\n"
        f"_{la.get('description', '')}_"
    )
    if st.sidebar.button("Clear loaded algorithm"):
        st.session_state.rb_loaded_algo = None


# ============================================================
# Tier filter (main content area)
# ============================================================
tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        checked = st.checkbox(
            f"{tier_badge(tier)} {TIER_LABELS[tier]}",
            value=(tier in st.session_state.rb_tiers_enabled),
            help=TIER_DESCRIPTIONS[tier],
            key=f"rb_tier_checkbox_{tier}",
        )
        if checked:
            new_enabled.append(tier)

st.session_state.rb_tiers_enabled = new_enabled

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
        if f"rb_bundle_{bk}" not in st.session_state:
            st.session_state[f"rb_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(
            bundle["label"], 0, 100,
            step=5,
            key=f"rb_bundle_{bk}",
            label_visibility="collapsed",
        )

    # Bundles not in active_bundles still need a zero entry for save
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

    # Build list of all stats in enabled tiers, sorted by tier then by label
    all_enabled_stats = [
        z for z, t in stat_tiers.items() if t in new_enabled
    ]
    all_enabled_stats.sort(key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z)))

    for z_col in all_enabled_stats:
        tier = stat_tiers.get(z_col, 2)
        label = stat_labels.get(z_col, z_col)
        meth = stat_methodology.get(z_col, {})

        # Build a rich help tooltip with What/How/Limits
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
            key=f"adv_rb_{z_col}",
            help=help_text,
        )
        if w > 0:
            effective_weights[z_col] = w

    # For save compatibility — advanced mode doesn't save, but community_section
    # expects bundle_weights to exist
    bundle_weights = {bk: 0 for bk in BUNDLES}


# ============================================================
# Filter & score
# ============================================================
filtered = df[df["carries"].fillna(0) >= min_carries].copy()

if len(filtered) == 0:
    st.warning("No backs match the current filters. Try lowering the carry threshold.")
    st.stop()

filtered = score_players(filtered, effective_weights)

# Metric picker — sort leaderboard by any nerd metric
RB_METRICS = {
    "Rushing yards": ("rush_yards", False),
    "Rushing TDs": ("rush_tds", False),
    "Carries": ("carries", False),
    "Receptions": ("receptions", False),
    "Yards per carry": ("yards_per_carry", False),
    "EPA per rush": ("epa_per_rush", False),
    "Rush success rate": ("rush_success_rate", False),
    "Snap share": ("snap_share", False),
    "Touches per game": ("touches_per_game", False),
    "Targets per game": ("targets_per_game", False),
    "Explosive run rate (10+)": ("explosive_run_rate", False),
    "15+ yard run rate": ("explosive_15_rate", False),
    "Red zone carry share": ("rz_carry_share", False),
    "Goal-line TD rate": ("goal_line_td_rate", False),
    "Yards after contact / att": ("yards_after_contact_per_att", False),
    "Yards before contact / att": ("yards_before_contact_per_att", False),
    "Broken tackles / att": ("broken_tackles_per_att", False),
    "RYOE per attempt (NGS)": ("ryoe_per_att", False),
    "Receiving EPA / target": ("rec_epa_per_target", False),
}
sort_label, sort_col, sort_ascending = metric_picker(RB_METRICS, key="rb_metric_picker")

total_weight = sum(effective_weights.values())
if total_weight == 0:
    st.info("All weights are zero — drag some sliders to start ranking.")

if sort_col in filtered.columns:
    filtered = filtered.sort_values(sort_col, ascending=sort_ascending, na_position="last").reset_index(drop=True)
else:
    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)
filtered.index = filtered.index + 1

# Compute sample size as % of group leader's carries
max_carries = filtered["carries"].fillna(0).max()
if max_carries > 0:
    filtered["sample_pct"] = (filtered["carries"].fillna(0) / max_carries) * 100
else:
    filtered["sample_pct"] = 0


# ============================================================
# Ranking table
# ============================================================
# Hide-small-samples checkbox
hide_small = st.checkbox(
    "Hide players with severe small samples (<20% of group leader's carries)",
    value=False,
    key="rb_hide_small",
    help="Hides red-flagged players. Yellow-flagged players still show with a caution.",
)

ranked = filtered.copy()
if hide_small:
    ranked = ranked[ranked["sample_pct"] >= 20].copy()
    if len(ranked) == 0:
        st.warning("All backs are below 20% sample size. Uncheck the filter to see them.")
        st.stop()
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1

# ── Master/detail click-to-detail leaderboard ──────────────────
# Top scorer banner (browse-only)
_top_html = None
_top_warn = None
if len(ranked) > 0:
    _top = ranked.iloc[0]
    _top_name = _top["player_display_name"]
    _top_score = _top["score"]
    _top_pct = _top.get("sample_pct", 100)
    _badge = sample_size_badge(_top_pct)
    _sign = "+" if _top_score >= 0 else ""
    _top_html = (
        f"<div style='background:#0076B6;color:white;padding:14px 20px;"
        f"border-radius:8px;margin-bottom:8px;font-size:1.1rem;'>"
        f"<span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span>"
        f" &nbsp;·&nbsp; <strong>{_top_name}</strong> {_badge}"
        f" &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{_sign}{_top_score:.2f}</span>"
        f" <span style='opacity:0.85;'>({format_percentile(zscore_to_percentile(_top_score))})</span>"
        f"</div>"
    )
    _top_warn = sample_size_caption(_top_pct)

def _fmt_int(v): return f"{int(v)}" if pd.notna(v) else "—"
def _fmt_signed(v, places=2): return f"{v:+.{places}f}" if pd.notna(v) else "—"
def _fmt_float(v, places=2): return f"{v:.{places}f}" if pd.notna(v) else "—"

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "": ranked["sample_pct"].apply(sample_size_badge),
    "Player": ranked["player_display_name"],
    "Att": ranked["carries"].apply(_fmt_int),
    "Yds": ranked["rush_yards"].apply(_fmt_int),
    "TDs": ranked["rush_tds"].apply(_fmt_int),
    "Rec": ranked["receptions"].apply(_fmt_int),
    "YPC": ranked.get("yards_per_carry", pd.Series([float("nan")] * len(ranked))).apply(lambda v: _fmt_float(v, 2)),
    "EPA/rush": ranked.get("epa_per_rush", pd.Series([float("nan")] * len(ranked))).apply(lambda v: _fmt_signed(v, 2)),
    "YACO/att": ranked.get("yards_after_contact_per_att", pd.Series([float("nan")] * len(ranked))).apply(lambda v: _fmt_float(v, 2)),
    "Your score": ranked["score"].apply(format_score),
})

selected = render_master_detail_leaderboard(
    display_df=display_df,
    name_col="Player",
    key_prefix="rb",
    team=selected_team,
    season=selected_season,
    top_banner_html=_top_html,
    top_banner_warn=_top_warn,
    leaderboard_caption="",
)
if selected is None:
    with st.expander("ℹ️ How is this score calculated?"):
        st.markdown(SCORE_EXPLAINER)
    st.stop()

# ============================================================
# Player detail
# ============================================================
player = ranked[ranked["player_display_name"] == selected].iloc[0]

# Sample size warning for the selected player
warn = sample_size_caption(player.get("sample_pct", 100))
if warn:
    st.warning(warn)

# ── Split-season panel: surface other stints if traded mid-season ──
all_rbs_full = load_rb_data()
season_stints = all_rbs_full[
    (all_rbs_full["player_id"] == player.get("player_id"))
    & (all_rbs_full["season_year"] == selected_season)
].copy()
if len(season_stints) > 1:
    n = len(season_stints)
    st.info(f"**Split season** — {selected} played for {n} teams in {selected_season}.")
    season_stints = season_stints.sort_values("first_week" if "first_week" in season_stints.columns else "off_snaps", ascending=True)
    split_rows = []
    for _, stint in season_stints.iterrows():
        team_disp = display_abbr(stint["recent_team"])
        is_current = stint["recent_team"] == player["recent_team"]
        split_rows.append({
            "Team": f"⮕ {team_disp}" if is_current else team_disp,
            "Games": _fmt_int(stint.get("games")),
            "Snaps": _fmt_int(stint.get("off_snaps")),
            "Att": _fmt_int(stint.get("carries")),
            "Yds": _fmt_int(stint.get("rush_yards")),
            "TDs": _fmt_int(stint.get("rush_tds")),
            "Rec": _fmt_int(stint.get("receptions")),
            "YPC": _fmt_float(stint.get("yards_per_carry"), 2),
            "EPA/rush": _fmt_signed(stint.get("epa_per_rush"), 2),
            "YACO/att": _fmt_float(stint.get("yards_after_contact_per_att"), 2),
        })
    # Season-total row (weighted aggregates)
    def _safe_sum(col):
        return season_stints[col].fillna(0).sum() if col in season_stints.columns else float("nan")
    def _weighted_mean(value_col, weight_col):
        if value_col not in season_stints.columns or weight_col not in season_stints.columns:
            return float("nan")
        v = season_stints[value_col]; w = season_stints[weight_col]
        mask = v.notna() & w.notna() & (w > 0)
        if not mask.any():
            return float("nan")
        return (v[mask] * w[mask]).sum() / w[mask].sum()

    total_games = _safe_sum("games")
    total_snaps = _safe_sum("off_snaps")
    total_carries = _safe_sum("carries")
    total_rush_yards = _safe_sum("rush_yards")
    total_rush_tds = _safe_sum("rush_tds")
    total_receptions = _safe_sum("receptions")
    season_ypc = (total_rush_yards / total_carries) if total_carries > 0 else float("nan")
    season_epa = _weighted_mean("epa_per_rush", "carries")
    season_yaco = _weighted_mean("yards_after_contact_per_att", "carries")

    split_rows.append({
        "Team": f"**Total ({selected_season})**",
        "Games": _fmt_int(total_games),
        "Snaps": _fmt_int(total_snaps),
        "Att": _fmt_int(total_carries),
        "Yds": _fmt_int(total_rush_yards),
        "TDs": _fmt_int(total_rush_tds),
        "Rec": _fmt_int(total_receptions),
        "YPC": _fmt_float(season_ypc, 2),
        "EPA/rush": _fmt_signed(season_epa, 2),
        "YACO/att": _fmt_float(season_yaco, 2),
    })
    st.dataframe(pd.DataFrame(split_rows), use_container_width=True, hide_index=True)

# ── Unified Season picker — drives stat bar + bundle table + radar ──
player_career = all_rbs_full[all_rbs_full["player_id"] == player.get("player_id")]

st.markdown(f"### {selected}")

_yr = render_player_year_picker(
    career_df=player_career,
    default_season=selected_season,
    season_col="season_year",
    team_col="recent_team",
    key_prefix=f"rb_{player.get('player_id') or selected}",
)
view_row = _yr["view_row"] if _yr["view_row"] is not None else player
year_choice = _yr["year_choice"]

if total_weight > 0:
    _view_score = sum(view_row.get(z, 0) * (w / total_weight)
                       for z, w in effective_weights.items()
                       if pd.notna(view_row.get(z)))
else:
    _view_score = float("nan")

RB_STAT_SPECS = [
    ("carries", "{:.0f}", "Car"),
    ("rush_yards", "{:.0f}", "Rush Yds"),
    ("rush_tds", "{:.0f}", "TD"),
    ("yards_per_carry", "{:.1f}", "Y/Car"),
    ("epa_per_rush", "{:+.2f}", "EPA/Rush"),
    ("receptions", "{:.0f}", "Rec"),
]
NFL_SUM_COLS = {"off_snaps", "def_snaps", "snaps", "games", "targets",
                "receptions", "rec_yards", "rec_tds",
                "attempts", "completions", "passing_yards", "passing_tds",
                "passing_interceptions", "rushing_yards", "rushing_tds",
                "carries", "rush_yards", "rush_tds", "rushing_attempts",
                "tackles", "def_tackles",
                "sacks", "tfls", "tackles_for_loss",
                "interceptions", "def_interceptions", "passes_defensed",
                "passes_defended", "qb_hits", "fg_made", "fg_attempts",
                "fg_att", "xp_made", "punts", "punt_yards", "total_yards"}
# ── Trading-card visual ────────────────────────────────────────
_team_abbr = _yr["team_str"] if _yr["team_str"] else (player.get("recent_team") or "")
# In-page banner removed — the trading card below is now the page hero.

# ── Trading-card export ──────────────────────────────────────────
# One-click PNG download for sharing. Card is composed from the
# current slider preset's score + signature/weakness narrative +
# 4 headline counting/rate stats.
def _safe_format(val, fmt: str = "{:.0f}") -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        return fmt.format(val)
    except (ValueError, TypeError):
        return str(val)

# Build the narrative for the card (same engine the page panel uses).
_card_narrative = None
try:
    from lib_field_viz import build_rb_narrative
    from lib_splits import _classify_gap, _load_rusher_plays, _load_rb_peer_pools
    rp_full = _load_rusher_plays()
    if rp_full is not None and player.get("player_id"):
        pf_career = rp_full[rp_full["player_id"] == player.get("player_id")].copy()
        if not pf_career.empty:
            pf_career["gap_code"] = pf_career.apply(_classify_gap, axis=1)
            _card_narrative = build_rb_narrative(
                pf_career, peer_pools=_load_rb_peer_pools(),
            )
except Exception:
    _card_narrative = None

# 4 headline stats for the card stats row.
_card_stats = [
    ("Carries",  _safe_format(view_row.get("carries")),
                 _safe_format(view_row.get("yards_per_carry"), "{:.1f} YPC")),
    ("Rush yds", _safe_format(view_row.get("rush_yards")),
                 _safe_format(view_row.get("rush_tds"), "{:.0f} TD")),
    ("EPA/Car",  _safe_format(view_row.get("epa_per_rush"), "{:+.2f}"), ""),
    ("Success%", _safe_format(view_row.get("rush_success_rate"), "{:.0%}"),
                 "FO definition"),
]

from lib_shared import team_theme as _theme
from lib_trading_card import render_card_download_button as _render_card

_render_card(
    player_name=selected,
    position_label=(player.get("position") or "RB"),
    season_str=_yr["season_str"] or f"Season {selected_season}",
    score=_view_score,
    narrative=_card_narrative,
    key_stats=_card_stats,
    player_id=player.get("player_id") or selected,
    team_abbr=_team_abbr,
    theme=_theme(_team_abbr),
    preset_name=(st.session_state.rb_loaded_algo.get("name")
                  if st.session_state.get("rb_loaded_algo") else None),
    key_prefix=f"rb_{player.get('player_id') or selected}",
    position_group="rb",
    bundle_weights=bundle_weights,
    season=(None if _yr["is_career_view"] else selected_season),
)

# ── Combine workout chart vs. all-time RB pool ────────────────
_WORKOUTS_PATH = Path(__file__).resolve().parent.parent / "data" / "college" / "nfl_all_workouts.parquet"
render_combine_chart(
    player_name=selected,
    position="RB",
    workouts_path=_WORKOUTS_PATH,
    key=f"rb_combine_chart_{player.get('player_id', selected)}",
)

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f"**Your score:** {format_score(_view_score)}")
    st.markdown("---")
    st.markdown("**How your score breaks down**")

    if not advanced_mode:
        # ── Underlying stats — primary view (what fans care about) ──
        stat_rows = []
        shown_stats = set()
        for bundle in active_bundles.values():
            shown_stats.update(bundle["stats"].keys())
        if 1 in new_enabled:
            for z_col, t in stat_tiers.items():
                if t == 1:
                    shown_stats.add(z_col)
        for z_col in sorted(shown_stats, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            tier = stat_tiers.get(z_col, 2)
            label = stat_labels.get(z_col, z_col)
            raw_col = RAW_COL_MAP.get(z_col)
            z = view_row.get(z_col)
            raw = view_row.get(raw_col) if raw_col else None
            stat_rows.append({
                "Tier": tier_badge(tier),
                "Stat": label,
                "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
                "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
            })
        if stat_rows:
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No stats to show — enable more tiers in the sidebar.")

        # ── Slider/bundle breakdown — collapsible (the methodology) ──
        with st.expander("⚙️  How your slider preset weights this player"):
            bundle_rows = []
            for bk, bundle in active_bundles.items():
                bw = bundle_weights.get(bk, 0)
                if bw == 0:
                    continue
                contribution = 0.0
                for z_col, internal in bundle["stats"].items():
                    z = view_row.get(z_col)
                    if pd.notna(z) and total_weight > 0:
                        contribution += z * (bw * internal / total_weight)
                bundle_rows.append({
                    "Skill": bundle["label"],
                    "Your weight": f"{bw}",
                    "Points added": f"{contribution:+.2f}",
                })
            if bundle_rows:
                st.dataframe(pd.DataFrame(bundle_rows),
                              use_container_width=True, hide_index=True)
            else:
                st.caption("No bundles weighted — drag some sliders.")

    else:
        st.caption("Stat-by-stat breakdown (z-score vs league)")
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            tier = stat_tiers.get(z_col, 2)
            label = stat_labels.get(z_col, z_col)
            raw_col = RAW_COL_MAP.get(z_col)
            z = view_row.get(z_col)
            raw = view_row.get(raw_col) if raw_col else None
            w = effective_weights.get(z_col, 0)
            contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            rows.append({
                "Tier": tier_badge(tier),
                "Stat": label,
                "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
                "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                "Weight": f"{w}",
                "Points added": f"{contrib:+.2f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No stats weighted — drag some sliders.")

with c2:
    st.markdown("**Stat profile** (percentiles vs. league reference)")
    st.caption("Solid blue = this player. Dashed gray = top-32 starter average.")
    radar_row = view_row if view_row is not None else player
    season_pool = all_rbs_full[all_rbs_full["season_year"] == selected_season]
    top32 = season_pool.sort_values("off_snaps", ascending=False).head(32)
    radar_bench = {z: top32[z].mean() for z in RADAR_STATS if z in top32.columns and top32[z].notna().any()}
    radar_bench_raw = {}
    for z in RADAR_STATS:
        raw_col = RAW_COL_MAP.get(z)
        if raw_col and raw_col in top32.columns and top32[raw_col].notna().any():
            radar_bench_raw[z] = top32[raw_col].mean()
    fig = build_radar_figure(radar_row, stat_labels, stat_methodology,
                              benchmark=radar_bench, benchmark_raw=radar_bench_raw)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No radar data available for this player.")
    st.caption(
        "Each axis shows where this player ranks among RBs with 100+ snaps. "
        "50 = league median, 84 = +1 SD, 97 = +2 SD. "
        "Hover any data point for stat details."
    )

    # ── Compare radar to another running back ────────────
    # Side-by-side radars + score-comparison headline. Helper handles
    # dropdown, toggle, layout. Score for the comparison player is
    # computed via the same effective_weights formula as the primary.
    def _rb_score_of(row):
        if row is None or total_weight <= 0:
            return float("nan")
        return sum(
            row.get(z, 0) * (w / total_weight)
            for z, w in effective_weights.items()
            if pd.notna(row.get(z))
        )

    from lib_shared import render_player_comparison, team_theme as _theme
    render_player_comparison(
        player_row=view_row,
        player_name=selected,
        league_df=all_rbs_full,
        name_col="player_display_name",
        year_choice=year_choice,
        primary_score=_view_score,
        compute_comparison_score=_rb_score_of,
        radar_builder=build_radar_figure,
        benchmark=radar_bench,
        benchmark_raw=radar_bench_raw,
        stat_labels=stat_labels,
        stat_methodology=stat_methodology,
        key_prefix=f"rb_cmp_{player.get('player_id', selected)}",
        position_label="running back",
        theme=_theme(_team_abbr),
    )


# ── Run scheme profile (exposed, full-width) ─────────────────
from lib_splits import render_run_scheme_section as _render_run_scheme_section
_render_run_scheme_section(
    player_name=selected,
    season=selected_season,
    key_prefix=f"rb_run_{player.get('player_id') or selected}",
    is_career_view=_yr["is_career_view"],
)

# ── Coverage matchup profile (exposed, full-width) ───────────
from lib_splits import render_coverage_matchup_section as _render_coverage_matchup_section
_render_coverage_matchup_section(
    player_name=selected,
    season=selected_season,
    position_group="RB",
    key_prefix=f"rb_cov_{player.get('player_id') or selected}",
    is_career_view=_yr["is_career_view"],
)

# ── Game-by-game splits explorer (NEW) ───────────────────────
# Schedule-adjusted form / strength / consistency tiles + filters
# (opponent strength, roof, surface, weather, location, result).
# Single-season only; renders nothing in all-career view.
from lib_splits import render_splits_section as _render_splits_section
_render_splits_section(
    player_name=selected,
    season=selected_season,
    position_group="RB",
    key_prefix=f"rb_{player.get('player_id') or selected}",
    is_career_view=_yr["is_career_view"],
)


# ============================================================
# Community algorithms
# ============================================================
career_arc_section(
    player=player,
    league_parquet_path=DATA_PATH,
    z_score_cols=list(RAW_COL_MAP.keys()),
    stat_labels=stat_labels,
    id_col="player_id",
    name_col="player_display_name",
    position_label="running backs",
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

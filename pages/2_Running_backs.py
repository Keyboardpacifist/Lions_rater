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

from team_selector import get_team_and_season, filter_by_team_and_season, NFL_TEAMS
from career_arc import career_arc_section
from lib_shared import (
    apply_algo_weights,
    community_section,
    compute_effective_weights,
    get_algorithm_by_slug,
    inject_css,
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


def build_radar_figure(player, stat_labels, stat_methodology):
    """Return a Plotly polar figure showing this player's percentiles
    on the 8 RADAR_STATS axes. Missing values are skipped.
    Hovering over a data point shows the stat's 'what' description
    from the methodology metadata."""
    axes = []
    values = []
    descriptions = []
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

    if not axes:
        return None

    # Close the polygon by repeating the first point at the end
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
standardized stats where 0 is the league average, +1 is one standard
deviation above, and −1 is one standard deviation below. Your slider
weights control how much each bundle contributes.

**How to read it:**
- `+1.0` or higher → well above the league average on what you weighted
- `+0.4` to `+1.0` → above average
- `−0.4` to `+0.4` → roughly average
- `−1.0` or lower → well below average

**What this is not.** It's not a PFF-style 0-100 grade. It's a
**comparative** number telling you how Lions running backs stack up
against the top 32 RBs in the league, under the methodology *you* chose.

**League population:** z-scores are computed against the top 32 RBs by
offensive snaps (min 6 games played). Every Lions RB with at least one
offensive snap is visible, but players with very few carries will have
noisy scores — read extreme values on low-volume players as "small
sample, not skill."
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
st.subheader(f"{team_name} running backs")
st.markdown(
    "What makes a great player? **You decide.** Drag the sliders to weight what you "
    "value, and watch the Lions running backs re-rank in real time. "
    "_No 'best back' — just **your** best back._"
)
st.caption(
    f"{selected_season} regular season • Compared against top 32 RBs by snaps • "
    "Every Lions RB visible"
)


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

total_weight = sum(effective_weights.values())
if total_weight == 0:
    st.info("All weights are zero — drag some sliders to start ranking.")

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
st.subheader("Ranking")

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

# Top-ranked highlight banner
if len(ranked) > 0:
    top = ranked.iloc[0]
    top_name = top["player_display_name"]
    top_score = top["score"]
    top_pct = top.get("sample_pct", 100)
    badge = sample_size_badge(top_pct)
    sign = "+" if top_score >= 0 else ""
    st.markdown(
        f"<div style='background:#0076B6;color:white;padding:14px 20px;"
        f"border-radius:8px;margin-bottom:8px;font-size:1.1rem;'>"
        f"<span style='font-size:1.4rem;font-weight:bold;'>#1 of {len(ranked)}</span>"
        f" &nbsp;·&nbsp; <strong>{top_name}</strong> {badge}"
        f" &nbsp;·&nbsp; <span style='font-size:1.4rem;font-weight:bold;'>{sign}{top_score:.2f}</span>"
        f" <span style='opacity:0.85;'>({format_percentile(zscore_to_percentile(top_score))})</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    warn = sample_size_caption(top_pct)
    if warn:
        st.warning(warn)

st.caption(
    "⚠️ Backs with very few carries have noisy scores — extreme values "
    "reflect small sample sizes, not skill. Use the 'Minimum carries' "
    "filter in the sidebar to hide low-volume backs if desired. "
    "🔴 = severe small sample (<20% of group leader's carries), 🟡 = caution (20–50%)."
)

display_df = pd.DataFrame({
    "Rank": ranked.index,
    "": ranked["sample_pct"].apply(sample_size_badge),
    "Player": ranked["player_display_name"],
    "Carries": ranked["carries"].fillna(0).astype(int),
    "Rush yds": ranked["rush_yards"].fillna(0).astype(int),
    "Rush TDs": ranked["rush_tds"].fillna(0).astype(int),
    "Rec": ranked["receptions"].fillna(0).astype(int),
    "Rec yds": ranked["rec_yards"].fillna(0).astype(int),
    "Your score": ranked["score"].apply(format_score),
})
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

with st.expander("ℹ️ How is this score calculated?"):
    st.markdown(SCORE_EXPLAINER)


# ============================================================
# Player detail
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")

selected = st.selectbox(
    "Pick a back to see how their score breaks down",
    options=ranked["player_display_name"].tolist(),
    index=0,
)

player = ranked[ranked["player_display_name"] == selected].iloc[0]

# Sample size warning for the selected player
warn = sample_size_caption(player.get("sample_pct", 100))
if warn:
    st.warning(warn)

c1, c2 = st.columns([1, 1])
with c1:
    # Player heading
    st.markdown(f"### {selected}")
    st.caption(f"{int(player.get('carries') or 0)} carries · "
               f"{int(player.get('rush_yards') or 0)} rush yds · "
               f"{int(player.get('rush_tds') or 0)} rush TDs · "
               f"{int(player.get('receptions') or 0)} rec · "
               f"{int(player.get('rec_yards') or 0)} rec yds")
    st.markdown(f"**Your score:** {format_score(player['score'])}")
    st.markdown("---")
    st.markdown("**How your score breaks down**")

    if not advanced_mode:
        bundle_rows = []
        for bk, bundle in active_bundles.items():
            bw = bundle_weights.get(bk, 0)
            if bw == 0:
                continue
            contribution = 0.0
            for z_col, internal in bundle["stats"].items():
                z = player.get(z_col)
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
            if 1 in new_enabled:
                for z_col, t in stat_tiers.items():
                    if t == 1:
                        shown_stats.add(z_col)
            for z_col in sorted(shown_stats, key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
                tier = stat_tiers.get(z_col, 2)
                label = stat_labels.get(z_col, z_col)
                raw_col = RAW_COL_MAP.get(z_col)
                z = player.get(z_col)
                raw = player.get(raw_col) if raw_col else None
                stat_rows.append({
                    "Tier": tier_badge(tier),
                    "Stat": label,
                    "Raw": f"{raw:.2f}" if pd.notna(raw) else "—",
                    "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                })
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

    else:
        st.caption("Stat-by-stat breakdown (z-score vs league)")
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 2), stat_labels.get(z, z))):
            tier = stat_tiers.get(z_col, 2)
            label = stat_labels.get(z_col, z_col)
            raw_col = RAW_COL_MAP.get(z_col)
            z = player.get(z_col)
            raw = player.get(raw_col) if raw_col else None
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
    fig = build_radar_figure(player, stat_labels, stat_methodology)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No radar data available for this player.")
    st.caption(
        "Each axis shows where this player ranks among the league reference "
        "population (top 32 RBs). 50 = league median, 84 = +1 SD, "
        "97 = +2 SD. Hover any data point for the stat description."
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

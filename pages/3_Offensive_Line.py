"""
Lions Offensive Line Rater
==========================
Tier-based slider UI for OL rankings, with save/load/browse community
algorithms scoped to position_group='ol'.

Layout matches the Receivers and Running Backs pages: ranking table at the
top, then team context banner, then player detail, then community section.
"""

from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm

from lib_shared import (
    apply_algo_weights,
    community_section,
    compute_effective_weights,
    inject_css,
    score_players,
)

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Lions Offensive Line Rater",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

POSITION_GROUP = "ol"
PAGE_URL = "https://lions-rater.streamlit.app/Offensive_Line"

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "master_lions_ol_with_z.parquet"
METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ol_stat_metadata.json"


# ============================================================
# Bundle definitions
# ============================================================
OL_BUNDLES = {
    "run_blocking": {
        "label": "Run blocking",
        "description": "Creates space on running plays.",
        "stats": {
            "z_gap_success_rate": 1.0,
            "z_gap_epa_per_play": 1.0,
            "z_garsr": 1.0,
            "z_rb_adjusted_gap_epa": 1.0,
            "z_explosive_enablement": 1.0,
        },
    },
    "pass_protection": {
        "label": "Pass protection",
        "description": "Keeps the QB upright.",
        "stats": {
            "z_on_off_sack_rate_diff": 1.0,
        },
    },
    "discipline": {
        "label": "Discipline",
        "description": "Avoids costly penalties.",
        "stats": {
            "z_penalties_total": 1.0,
            "z_penalty_rate": 1.0,
            "z_penalty_leverage_cost": 1.0,
        },
    },
    "availability": {
        "label": "Availability",
        "description": "On the field when it matters.",
        "stats": {
            "z_snaps_played": 1.0,
            "z_availability_index": 1.0,
        },
    },
    "experimental": {
        "label": "Experimental",
        "description": "Speculative stats — use with skepticism.",
        "stats": {
            "z_mobility_index": 1.0,
            "z_leverage_rating": 1.0,
            "z_pass_run_balance": 1.0,
        },
    },
}

DEFAULT_BUNDLE_WEIGHTS = {
    "run_blocking": 60,
    "pass_protection": 50,
    "discipline": 30,
    "availability": 20,
    "experimental": 0,
}

# Methodology — per-stat What/How/Limits used in Advanced mode tooltips
OL_METHODOLOGY = {
    "z_snaps_played": {
        "what": "Total offensive snaps played in the season.",
        "how": "Sum of offense_snaps from nflverse snap counts.",
        "limits": "Doesn't distinguish run from pass snaps.",
    },
    "z_penalties_total": {
        "what": "Count of offensive penalties charged to this player.",
        "how": "Filter PBP where penalty_player_name matches, restricted to OL penalty types.",
        "limits": "Raw counts ignore context — a holding wiping out 40 yards counts the same as one on a 2-yard loss. Penalty Leverage Cost addresses this.",
    },
    "z_penalty_rate": {
        "what": "Penalties per offensive snap.",
        "how": "Total penalties divided by offense snaps.",
        "limits": "Season-rate smoothing means one bad game can move the number meaningfully.",
    },
    "z_gap_success_rate": {
        "what": "Success rate on runs through this lineman's assigned gap.",
        "how": "Filter Lions runs to the gap owned by this player's position (strict attribution), then take the mean of nflverse's built-in 'success' field.",
        "limits": "Gap attribution is approximate — linemen pull and combo-block on plays the play-by-play doesn't know about. Guards get smaller samples because most interior runs get coded as 'middle' rather than 'guard'.",
    },
    "z_gap_epa_per_play": {
        "what": "Average Expected Points Added on runs through this lineman's gap.",
        "how": "Mean of nflverse EPA on gap-attributed plays.",
        "limits": "Same gap attribution caveats as Gap Success Rate.",
    },
    "z_availability_index": {
        "what": "Share of team snaps played, weighted by games played.",
        "how": "(player_snaps / max_possible_snaps) × (games_played / 17)",
        "limits": "A player benched for performance looks the same as one benched for injury.",
    },
    "z_garsr": {
        "what": "Gap Run Success Rate adjusted for situational difficulty.",
        "how": "Actual gap success rate minus predicted success rate from a league-wide linear regression (features: down, distance, yardline, gap, location).",
        "limits": "The baseline model is deliberately simple for transparency. R² ~0.04 because run success is inherently noisy.",
    },
    "z_rb_adjusted_gap_epa": {
        "what": "Gap EPA minus what you'd expect from the backs who ran through it.",
        "how": "For each gap run, compute (actual EPA) - (that rusher's season average EPA per carry). Average the residuals.",
        "limits": "Adjusts for rusher quality but not for situational mix.",
    },
    "z_penalty_leverage_cost": {
        "what": "Total EPA cost of penalties committed by this player.",
        "how": "Sum the nflverse EPA value on each penalty play attributed to the player.",
        "limits": "Leverage weighting is a methodological choice — some analysts think it mixes talent measurement with clutch narrative.",
    },
    "z_explosive_enablement": {
        "what": "Rate of 15+ yard runs through this gap, relative to league baseline.",
        "how": "Percent of gap runs gaining 15+ yards, minus the same rate for comparable league-wide runs.",
        "limits": "Explosive runs require the line AND the back. Separating 'line sprung it' from 'back made it' is fundamentally hard.",
    },
    "z_on_off_sack_rate_diff": {
        "what": "Team sack rate when this player was out of the lineup vs. in the lineup.",
        "how": "(sack rate in games they missed) - (sack rate in games they played). Positive = team was sacked more often without them.",
        "limits": "Game-level, not play-level. NaN for players who didn't miss any games. Small samples for players who missed only 1-2 games.",
    },
    "z_mobility_index": {
        "what": "EXPERIMENTAL. Rough inference of pulling success for guards.",
        "how": "Success rate on runs to the opposite side of where the guard lines up, minus success rate on same-side runs.",
        "limits": "We can't actually see which plays involved pulls. This uses direction as a noisy proxy.",
    },
    "z_leverage_rating": {
        "what": "EXPERIMENTAL. Gap run EPA weighted by each play's win probability impact.",
        "how": "sum(EPA × |WPA|) / sum(|WPA|). Plays in close games get weighted more than blowouts.",
        "limits": "Leverage weighting is philosophically contested.",
    },
    "z_pass_run_balance": {
        "what": "EXPERIMENTAL. Whether this player is better at pass protection or run blocking.",
        "how": "z-score of Pass Pro On/Off Split minus z-score of Gap Run Success Rate.",
        "limits": "Derived from other stats, so it inherits all their weaknesses.",
    },
}


# ============================================================
# Radar chart config — 8 headline stats, fixed across users
# ============================================================
# Mix of run blocking, pass protection, discipline, and availability.
# Penalty rate is INVERTED on the radar so higher = fewer penalties = "Discipline".
RADAR_STATS = [
    "z_gap_success_rate",        # run blocking
    "z_gap_epa_per_play",        # run blocking quality
    "z_garsr",                   # adjusted run success
    "z_rb_adjusted_gap_epa",     # RB-adjusted run blocking
    "z_explosive_enablement",    # big play creation
    "z_on_off_sack_rate_diff",   # pass protection
    "z_penalty_rate",            # discipline (inverted)
    "z_availability_index",      # durability
]

# Stats where the z-score needs to be flipped for the radar
# (high raw value = bad, so we invert so high = good on the chart)
RADAR_INVERT = {"z_penalty_rate"}

# Custom radar axis labels — override default stat labels for clarity
RADAR_LABEL_OVERRIDES = {
    "z_penalty_rate": "Discipline",
    "z_on_off_sack_rate_diff": "Pass protection",
}


def zscore_to_percentile(z):
    """Convert a z-score to a 0-100 percentile via the normal CDF."""
    if pd.isna(z):
        return None
    return float(norm.cdf(z) * 100)


def build_radar_figure(player, stat_labels):
    """Return a Plotly polar figure showing this player's percentiles
    on the RADAR_STATS axes. Missing values are skipped.
    Stats in RADAR_INVERT have their z-score sign flipped before
    conversion to percentile."""
    axes = []
    values = []
    for z_col in RADAR_STATS:
        if z_col not in player.index:
            continue
        z = player.get(z_col)
        if pd.isna(z):
            continue
        # Flip inverted stats so higher percentile = better
        if z_col in RADAR_INVERT:
            z = -z
        pct = zscore_to_percentile(z)
        # Use override label if available, else fall back to metadata label
        label = RADAR_LABEL_OVERRIDES.get(z_col, stat_labels.get(z_col, z_col))
        axes.append(label)
        values.append(pct)

    if not axes:
        return None

    # Close the polygon by repeating the first point at the end
    axes_closed = axes + [axes[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=axes_closed,
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.25)",
        line=dict(color="rgba(31, 119, 180, 0.9)", width=2),
        marker=dict(size=6, color="rgba(31, 119, 180, 1)"),
        hovertemplate="<b>%{theta}</b><br>%{r:.0f}th percentile<extra></extra>",
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
# Data loading
# ============================================================
@st.cache_data
def load_ol_data():
    if not DATA_PATH.exists():
        return None, None
    df = pd.read_parquet(DATA_PATH)
    meta = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            meta = json.load(f)
    return df, meta


# ============================================================
# Tier helpers (matching WR/RB)
# ============================================================
TIER_LABELS = {
    1: "Tier 1 — Counted",
    2: "Tier 2 — Contextualized",
    3: "Tier 3 — Adjusted",
    4: "Tier 4 — Inferred",
}
TIER_DESCRIPTIONS = {
    1: "Pure recorded facts. No modeling.",
    2: "Counts divided by opportunity. Still no modeling.",
    3: "Compared against a modeled baseline. Model is simple and visible.",
    4: "Inferred from patterns the data can't directly see. Use with skepticism.",
}


def tier_badge(tier: int) -> str:
    return {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠"}.get(tier, "⚪")


def filter_bundles_by_tier(bundles: dict, stat_tiers: dict, enabled_tiers: list) -> dict:
    """Strip disabled-tier stats out of each bundle. Empty bundles drop out."""
    filtered = {}
    for bk, bdef in bundles.items():
        kept_stats = {
            z: w for z, w in bdef["stats"].items()
            if stat_tiers.get(z, 1) in enabled_tiers
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
        t = stat_tiers.get(z, 1)
        counts[t] = counts.get(t, 0) + 1
    return " ".join(f"{tier_badge(t)}×{c}" for t, c in sorted(counts.items()))


# ============================================================
# Score labels (matching WR/RB)
# ============================================================
def score_label(score):
    if pd.isna(score):
        return "—"
    if score >= 1.0:
        return "well above group"
    if score >= 0.4:
        return "above group"
    if score >= -0.4:
        return "about average"
    if score >= -1.0:
        return "below group"
    return "well below group"


def format_score(score):
    if pd.isna(score):
        return "—"
    sign = "+" if score >= 0 else ""
    return f"{sign}{score:.2f} ({score_label(score)})"


SCORE_EXPLAINER = """
**What this number means.** The score is a weighted average of z-scores —
standardized stats where 0 is the group average, +1 is one standard
deviation above, and −1 is one standard deviation below. Your slider
weights control how much each bundle contributes.

**How to read it:**
- `+1.0` or higher → well above the group average on what you weighted
- `+0.4` to `+1.0` → above average
- `−0.4` to `+0.4` → roughly average
- `−1.0` or lower → well below average

**What this is not.** It's not a PFF-style 0-100 grade. It's a
**comparative** number telling you how the Lions starters stack up
against each other under the methodology *you* chose.

**Pass protection limitation.** Pass protection is genuinely harder to
measure from free data than run blocking. The pass-protection bundle
relies on a single game-level on/off split, while the run-blocking
bundle has five complementary stats. Treat the pass-protection
contribution accordingly.

**Small-sample warning.** Scores here are computed within the Lions
starting five (n=5), so distributions are noisy and a player's "+1.0"
means one SD above the other Lions starters, not one SD above NFL
starting OL. Treat directional differences seriously, but don't over-read
small gaps. (League-wide z-scores for OL are on the project's roadmap.)
"""


# ============================================================
# Session state
# ============================================================
if "ol_loaded_algo" not in st.session_state:
    st.session_state.ol_loaded_algo = None
if "upvoted_ids" not in st.session_state:
    st.session_state.upvoted_ids = set()
if "ol_tiers_enabled" not in st.session_state:
    st.session_state.ol_tiers_enabled = [1, 2, 3]  # Tier 4 off by default


# ============================================================
# Header
# ============================================================
st.title("🦁 Lions Offensive Line Rater")
st.markdown(
    "**Build your own algorithm.** Drag the sliders to weight what you "
    "value, and watch the Lions starting five re-rank in real time. "
    "_No 'best lineman' — just **your** best lineman._"
)
st.caption(
    "2024 regular season • Compared within the Lions starting five • "
    "Transparency-first: every stat has a methodology popover"
)


# ============================================================
# Load data
# ============================================================
df, meta = load_ol_data()
if df is None:
    st.error(f"OL data not found at {DATA_PATH}")
    st.caption(
        "Run the data-pull notebook and upload the parquet + metadata "
        "files to `data/` in the repo."
    )
    st.stop()

stat_tiers = meta.get("stat_tiers", {}) if meta else {}
stat_labels = meta.get("stat_labels", {}) if meta else {}
ctx = meta.get("team_context", {}) if meta else {}


# ============================================================
# Loaded algorithm indicator (sidebar to match WR/RB)
# ============================================================
if st.session_state.ol_loaded_algo:
    la = st.session_state.ol_loaded_algo
    st.sidebar.info(
        f"Loaded: **{la['name']}** by {la['author']}\n\n"
        f"_{la.get('description', '')}_"
    )
    if st.sidebar.button("Clear loaded algorithm"):
        st.session_state.ol_loaded_algo = None


# ============================================================
# Sidebar — Advanced mode toggle (matching WR/RB)
# ============================================================
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
advanced_mode = st.sidebar.toggle(
    "🔬 Advanced mode", value=False,
    key="ol_advanced_mode",
    help="Show individual stat sliders with methodology tooltips instead of plain-English bundles.",
)

st.sidebar.header("What do you value?")


# ============================================================
# Tier filter (main content area)
# ============================================================
st.markdown("### How speculative do you want to get?")
st.caption(
    "Each stat is labeled by how much trust it asks from you. "
    "Uncheck tiers you don't want to include. Philosophy in a checkbox."
)
tier_cols = st.columns(4)
new_enabled = []
for i, tier in enumerate([1, 2, 3, 4]):
    with tier_cols[i]:
        checked = st.checkbox(
            f"{tier_badge(tier)} {TIER_LABELS[tier]}",
            value=(tier in st.session_state.ol_tiers_enabled),
            help=TIER_DESCRIPTIONS[tier],
            key=f"ol_tier_checkbox_{tier}",
        )
        if checked:
            new_enabled.append(tier)
st.session_state.ol_tiers_enabled = new_enabled

if not new_enabled:
    st.warning("Enable at least one tier to see ratings.")
    st.stop()

active_bundles = filter_bundles_by_tier(OL_BUNDLES, stat_tiers, new_enabled)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ============================================================
# Sliders (in sidebar to match WR/RB)
# ============================================================
bundle_weights = {}
effective_weights = {}

if not advanced_mode:
    if not active_bundles:
        st.info("No bundles have stats in the enabled tiers. Try enabling more tiers.")
        st.stop()

    st.sidebar.caption("Drag to weight what matters to you. 0 = ignore, 100 = max.")
    for bk, bundle in active_bundles.items():
        tier_summary = bundle_tier_summary(bundle["stats"], stat_tiers)
        st.sidebar.markdown(f"**{bundle['label']}**")
        st.sidebar.markdown(
            f"<div class='bundle-desc'>{bundle['description']}<br>"
            f"<small>{tier_summary}</small></div>",
            unsafe_allow_html=True,
        )
        if f"ol_bundle_{bk}" not in st.session_state:
            st.session_state[f"ol_bundle_{bk}"] = DEFAULT_BUNDLE_WEIGHTS.get(bk, 50)
        bundle_weights[bk] = st.sidebar.slider(
            bundle["label"], 0, 100,
            step=5,
            key=f"ol_bundle_{bk}",
            label_visibility="collapsed",
        )
    # Bundles not currently active still need a zero entry for save
    for bk in OL_BUNDLES:
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
    # Collect all active stats across bundles
    all_active_stats = set()
    for bdef in active_bundles.values():
        all_active_stats.update(bdef["stats"].keys())

    # Sort by tier then by label
    sorted_stats = sorted(
        all_active_stats,
        key=lambda s: (stat_tiers.get(s, 1), stat_labels.get(s, s)),
    )

    for stat in sorted_stats:
        tier = stat_tiers.get(stat, 1)
        label = stat_labels.get(stat, stat)
        meth = OL_METHODOLOGY.get(stat, {})

        # Build a rich help tooltip
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
            key=f"ol_stat_{stat}",
            help=help_text,
        )
        if w > 0:
            effective_weights[stat] = w

    # For save compatibility — advanced mode doesn't save, but community_section
    # expects bundle_weights to exist
    bundle_weights = {bk: 0 for bk in OL_BUNDLES}


# ============================================================
# Score and ranking
# ============================================================
scored = score_players(df, effective_weights)
scored_sorted = scored.sort_values("score", ascending=False).reset_index(drop=True)
scored_sorted.index = scored_sorted.index + 1

st.subheader("Ranking")
st.caption(
    "⚠️ Scores here compare the five Lions starters to each other, not to "
    "league-wide OL. With a sample of five, small differences are noisy — "
    "treat directional results seriously, but don't over-read close gaps."
)

display_cols = []
if "player" in scored_sorted.columns:
    display_cols.append("player")
if "slot" in scored_sorted.columns:
    display_cols.append("slot")
if "games_played" in scored_sorted.columns:
    display_cols.append("games_played")

display_df = scored_sorted[display_cols].copy()
display_df["Score"] = scored_sorted["score"].apply(format_score)

# Friendlier column names
rename_map = {
    "player": "Player",
    "slot": "Position",
    "games_played": "Games",
}
display_df = display_df.rename(columns=rename_map)

st.dataframe(display_df, use_container_width=True)

with st.expander("ℹ️ How is this score calculated?"):
    st.markdown(SCORE_EXPLAINER)


# ============================================================
# Team context banner (now BELOW the ranking)
# ============================================================
if ctx:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### How did the line perform as a unit?")
    col1, col2, col3 = st.columns(3)
    with col1:
        if ctx.get("lions_ybc_per_att") is not None:
            delta = ctx['lions_ybc_per_att'] - ctx.get('league_ybc_per_att', 0)
            st.metric(
                "Yards before contact / att",
                f"{ctx['lions_ybc_per_att']:.2f}",
                delta=f"{delta:+.2f} vs league",
            )
    with col2:
        if ctx.get("lions_yac_per_att") is not None:
            delta = ctx['lions_yac_per_att'] - ctx.get('league_yac_per_att', 0)
            st.metric(
                "Yards after contact / att",
                f"{ctx['lions_yac_per_att']:.2f}",
                delta=f"{delta:+.2f} vs league",
            )
    with col3:
        if ctx.get("lions_sack_rate") is not None:
            delta = ctx['lions_sack_rate'] - ctx.get('league_sack_rate', 0)
            st.metric(
                "Sack rate",
                f"{ctx['lions_sack_rate']:.1%}",
                delta=f"{delta:+.1%} vs league",
                delta_color="inverse",
            )
    st.caption(
        "Team-level numbers for the whole OL. Individual ratings above "
        "attribute play-by-play results to specific linemen by position."
    )


# ============================================================
# Player detail (matching WR/RB)
# ============================================================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("Player detail")

selected = st.selectbox(
    "Pick a lineman to see how their score breaks down",
    options=scored_sorted["player"].tolist(),
    index=0,
)
player = scored_sorted[scored_sorted["player"] == selected].iloc[0]

c1, c2 = st.columns([1, 1])

with c1:
    # Player heading
    slot = player.get("slot", "") if pd.notna(player.get("slot")) else ""
    st.markdown(f"### {selected}")
    st.caption(
        f"**{slot}** · "
        f"{int(player.get('snaps_played') or 0)} snaps · "
        f"{int(player.get('games_played') or 0)} games · "
        f"{int(player.get('penalties_total') or 0)} penalties"
    )
    st.markdown(f"**Your score:** {format_score(player['score'])}")
    st.markdown("---")

    total_weight = sum(effective_weights.values())

    if not advanced_mode:
        st.markdown("**How your score breaks down**")
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
                "Bundle": bundle["label"],
                "Your weight": f"{bw}",
                "Contribution": f"{contribution:+.2f}",
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
            for z_col in sorted(shown_stats, key=lambda z: (stat_tiers.get(z, 1), stat_labels.get(z, z))):
                tier = stat_tiers.get(z_col, 1)
                label = stat_labels.get(z_col, z_col)
                z = player.get(z_col)
                stat_rows.append({
                    "Tier": tier_badge(tier),
                    "Stat": label,
                    "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                })
            if stat_rows:
                st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("**Stat-by-stat breakdown** (z-score within Lions starters)")
        rows = []
        for z_col in sorted(effective_weights.keys(), key=lambda z: (stat_tiers.get(z, 1), stat_labels.get(z, z))):
            tier = stat_tiers.get(z_col, 1)
            label = stat_labels.get(z_col, z_col)
            z = player.get(z_col)
            w = effective_weights.get(z_col, 0)
            contrib = (z if pd.notna(z) else 0) * (w / total_weight) if total_weight > 0 else 0
            rows.append({
                "Tier": tier_badge(tier),
                "Stat": label,
                "Z-score": f"{z:+.2f}" if pd.notna(z) else "—",
                "Weight": f"{w}",
                "Contribution": f"{contrib:+.2f}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No stats weighted — drag some sliders.")

with c2:
    st.markdown("**Stat profile** (percentiles within Lions starters)")
    fig = build_radar_figure(player, stat_labels)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No radar data available for this player.")
    st.caption(
        "Each axis shows where this player ranks among the Lions starting five. "
        "50 = group median. The 'Discipline' axis is inverted — higher = fewer penalties."
    )


# ============================================================
# Community algorithms
# ============================================================
community_section(
    position_group=POSITION_GROUP,
    bundles=OL_BUNDLES,
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

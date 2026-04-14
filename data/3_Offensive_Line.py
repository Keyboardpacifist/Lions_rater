"""
Lions Offensive Line Rater
==========================
Tier-based slider UI for OL rankings, with save/load/browse community
algorithms scoped to position_group='ol'.

What's different from the Receivers/RB pages:
- Tier filter at the top — users pick how speculative they want to get
- Methodology popover on every stat in Advanced mode
- Team context banner at the top showing Lions-as-unit run/pass block numbers
- Score explainer below the leaderboard
"""

from pathlib import Path
import json
import pandas as pd
import streamlit as st

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

DATA_PATH = Path("data/master_lions_ol_with_z.parquet")
METADATA_PATH = Path("data/ol_stat_metadata.json")


# ============================================================
# Bundle definitions
# ============================================================
# Shape matches lib_shared: each bundle has a label and a stats dict
# mapping z-score columns to their internal weights within the bundle.
OL_BUNDLES = {
    "run_blocking": {
        "label": "Run blocking",
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
        "stats": {
            "z_on_off_sack_rate_diff": 1.0,
        },
    },
    "discipline": {
        "label": "Discipline",
        "stats": {
            "z_penalties_total": 1.0,
            "z_penalty_rate": 1.0,
            "z_penalty_leverage_cost": 1.0,
        },
    },
    "availability": {
        "label": "Availability",
        "stats": {
            "z_snaps_played": 1.0,
            "z_availability_index": 1.0,
        },
    },
    "experimental": {
        "label": "Experimental",
        "stats": {
            "z_mobility_index": 1.0,
            "z_leverage_rating": 1.0,
            "z_pass_run_balance": 1.0,
        },
    },
}

BUNDLE_DESCRIPTIONS = {
    "run_blocking": "Creates space on running plays.",
    "pass_protection": "Keeps the QB upright.",
    "discipline": "Avoids costly penalties.",
    "availability": "On the field when it matters.",
    "experimental": "Speculative stats — use with skepticism.",
}

# Methodology for the popovers in Advanced mode
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
# Data loading
# ============================================================
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None, None
    df = pd.read_parquet(DATA_PATH)
    meta = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            meta = json.load(f)
    return df, meta


# ============================================================
# Tier helpers
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
            filtered[bk] = {"label": bdef["label"], "stats": kept_stats}
    return filtered


# ============================================================
# Score meaning labels
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
**What this number actually means.** The score is a weighted average of
z-scores — standardized stats where 0 is the group average, +1 is one
standard deviation above, and −1 is one standard deviation below. Your
slider weights control how much each bundle contributes.

**How to read it:**
- `+1.0` or higher → well above the group average on the stats you weighted
- `+0.4` to `+1.0` → above average
- `−0.4` to `+0.4` → roughly average
- `−1.0` or lower → well below average

**What this is not.** It's not a PFF-style 0-100 grade. It's a
**comparative** number telling you how each player stacks up against the
others in the group, given the methodology *you* chose.

**Small-sample warning:** scores here are computed within the Lions starting
five (n=5), so distributions are noisy. Treat directional differences
seriously, but don't over-read small gaps between players.
"""


# ============================================================
# Main page
# ============================================================
def main():
    st.title("🦁 Lions Offensive Line Rater")
    st.caption(
        "Build your own OL rating. Transparency-first: every stat has a "
        "methodology popover, and you choose how speculative you want to get."
    )

    df, meta = load_data()
    if df is None:
        st.error(f"OL data not found at {DATA_PATH}")
        st.caption(
            "Run the data-pull notebook and upload the parquet + metadata "
            "files to `data/` in the repo."
        )
        return

    # ---- Team context banner ----
    ctx = meta.get("team_context", {}) if meta else {}
    if ctx:
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
            "Team-level numbers for the whole OL. Individual ratings below "
            "attribute play-by-play results to specific linemen by position."
        )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- Loaded algorithm indicator ----
    if st.session_state.get("loaded_algo"):
        algo = st.session_state.loaded_algo
        st.info(f"Loaded algorithm: **{algo['name']}** — adjust sliders to fork it")

    # ---- Tier filter ----
    stat_tiers = meta.get("stat_tiers", {}) if meta else {}
    stat_labels = meta.get("stat_labels", {}) if meta else {}

    if "ol_tiers_enabled" not in st.session_state:
        st.session_state.ol_tiers_enabled = [1, 2, 3]  # Tier 4 off by default

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
        return

    # Filter bundles to only include stats from enabled tiers
    active_bundles = filter_bundles_by_tier(OL_BUNDLES, stat_tiers, new_enabled)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ---- Advanced mode toggle (in sidebar, matching other pages' convention) ----
    with st.sidebar:
        st.markdown("### Controls")
        advanced_mode = st.toggle(
            "Advanced mode (individual stats)",
            value=False,
            key="ol_advanced_mode",
            help="Show per-stat sliders with methodology popovers."
        )

    # ---- Sliders ----
    bundle_weights = {}

    if not advanced_mode:
        st.subheader("What matters to you?")
        for bk, bdef in active_bundles.items():
            tier_counts = {}
            for z in bdef["stats"]:
                t = stat_tiers.get(z, 1)
                tier_counts[t] = tier_counts.get(t, 0) + 1
            tier_summary = " ".join(
                f"{tier_badge(t)}×{c}" for t, c in sorted(tier_counts.items())
            )

            default = 50
            if f"bundle_{bk}" not in st.session_state:
                st.session_state[f"bundle_{bk}"] = default

            bundle_weights[bk] = st.slider(
                f"**{bdef['label']}** — {BUNDLE_DESCRIPTIONS[bk]}  \n*{tier_summary}*",
                min_value=0,
                max_value=100,
                key=f"bundle_{bk}",
            )

        # Compute effective weights using lib_shared
        effective_weights = compute_effective_weights(active_bundles, bundle_weights)

    else:
        st.subheader("Stat weights")
        st.caption("Tier badge next to each stat. Click ℹ️ for methodology.")
        # Collect all active stats across bundles
        all_active_stats = set()
        for bdef in active_bundles.values():
            all_active_stats.update(bdef["stats"].keys())

        effective_weights = {}
        for stat in sorted(all_active_stats, key=lambda s: stat_tiers.get(s, 1)):
            tier = stat_tiers.get(stat, 1)
            label = stat_labels.get(stat, stat)
            meth = OL_METHODOLOGY.get(stat, {})

            row = st.columns([3, 1])
            with row[0]:
                w = st.slider(
                    f"{tier_badge(tier)} {label}",
                    min_value=-100,
                    max_value=100,
                    value=0,
                    key=f"ol_stat_{stat}",
                )
            with row[1]:
                with st.popover("ℹ️ methodology"):
                    st.markdown(f"**{label}** — {TIER_LABELS[tier]}")
                    if meth:
                        st.markdown(f"**What:** {meth.get('what', '')}")
                        st.markdown(f"**How:** {meth.get('how', '')}")
                        st.markdown(f"**Limits:** {meth.get('limits', '')}")

            if w != 0:
                effective_weights[stat] = w

        # For advanced mode, fill bundle_weights with zeros so save still works
        # (lib_shared expects bundle-shaped data for saves, but advanced mode
        # doesn't save — the Save UI in community_section gracefully refuses)
        bundle_weights = {bk: 0 for bk in OL_BUNDLES}

    # ---- Score and leaderboard ----
    scored = score_players(df, effective_weights)
    scored["Score"] = scored["score"].apply(format_score)

    display_cols = ["player", "slot", "games_played", "Score"]
    display_cols = [c for c in display_cols if c in scored.columns]
    leaderboard = (
        scored.sort_values("score", ascending=False)
        .reset_index(drop=True)[display_cols]
    )
    leaderboard.index = leaderboard.index + 1

    st.subheader("Leaderboard")
    st.dataframe(leaderboard, width="stretch")

    with st.expander("ℹ️ How is this score calculated?"):
        st.markdown(SCORE_EXPLAINER)

    # ---- Community section (save/browse/fork/upvote) ----
    # Only render in bundle mode — community_section handles the
    # advanced-mode case with an info message.
    community_section(
        position_group=POSITION_GROUP,
        bundles=OL_BUNDLES,
        bundle_weights=bundle_weights,
        advanced_mode=advanced_mode,
        page_url=PAGE_URL,
    )


if __name__ == "__main__":
    main()

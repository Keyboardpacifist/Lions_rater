"""
Lions Rater — Stage 5
A transparent, customizable alternative to PFF.
Fans build and share their own rating methodologies.

Ethos: strive for the greatest accuracy while being transparent about
limitations and how we addressed them.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
import uuid

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


st.set_page_config(page_title="Lions Rater", page_icon="🦁", layout="wide")

WR_DATA_PATH = Path("data/master_lions_with_z.parquet")
OL_DATA_PATH = Path("data/master_lions_ol_with_z.parquet")
OL_METADATA_PATH = Path("data/ol_stat_metadata.json")

# ── Receivers (unchanged from Stage 4) ───────────────────────────────────────
WR_STAT_COLUMNS = [
    "z_catch_rate", "z_yards_per_route", "z_yards_per_reception",
    "z_air_yards_per_target", "z_adot", "z_targets_per_game",
    "z_receptions_per_game", "z_yac_per_reception",
    "z_broken_tackles_per_reception", "z_first_downs_per_reception",
    "z_td_rate",
]
WR_STAT_LABELS = {
    "z_catch_rate": "Catch rate",
    "z_yards_per_route": "Yards per route run",
    "z_yards_per_reception": "Yards per reception",
    "z_air_yards_per_target": "Air yards per target",
    "z_adot": "Average depth of target",
    "z_targets_per_game": "Targets per game",
    "z_receptions_per_game": "Receptions per game",
    "z_yac_per_reception": "YAC per reception",
    "z_broken_tackles_per_reception": "Broken tackles per reception",
    "z_first_downs_per_reception": "First downs per reception",
    "z_td_rate": "Touchdown rate",
}
WR_BUNDLES = {
    "Reliability": ["z_catch_rate", "z_first_downs_per_reception"],
    "Explosive plays": ["z_yards_per_reception", "z_td_rate"],
    "Field stretcher": ["z_air_yards_per_target", "z_adot"],
    "Volume & usage": ["z_targets_per_game", "z_receptions_per_game", "z_yards_per_route"],
    "After the catch": ["z_yac_per_reception", "z_broken_tackles_per_reception"],
}
WR_BUNDLE_DESCRIPTIONS = {
    "Reliability": "Catches the ball, moves the chains.",
    "Explosive plays": "Big gains and touchdowns.",
    "Field stretcher": "Gets targeted deep down the field.",
    "Volume & usage": "How often they're involved in the offense.",
    "After the catch": "What they do once the ball is in their hands.",
}

# ── Offensive Line ───────────────────────────────────────────────────────────
OL_BUNDLES = {
    "Run blocking": [
        "z_gap_success_rate", "z_gap_epa_per_play",
        "z_garsr", "z_rb_adjusted_gap_epa", "z_explosive_enablement",
    ],
    "Pass protection": ["z_on_off_sack_rate_diff"],
    "Discipline": ["z_penalties_total", "z_penalty_rate", "z_penalty_leverage_cost"],
    "Availability": ["z_snaps_played", "z_availability_index"],
    "Experimental": ["z_mobility_index", "z_leverage_rating", "z_pass_run_balance"],
}
OL_BUNDLE_DESCRIPTIONS = {
    "Run blocking": "Creates space on running plays.",
    "Pass protection": "Keeps the QB upright.",
    "Discipline": "Avoids costly penalties.",
    "Availability": "On the field when it matters.",
    "Experimental": "Speculative stats — use with skepticism.",
}

OL_METHODOLOGY = {
    "z_snaps_played": {
        "what": "Total offensive snaps played in the season.",
        "how": "Sum of offense_snaps from nflverse snap counts.",
        "limits": "Doesn't distinguish run from pass snaps.",
    },
    "z_penalties_total": {
        "what": "Count of offensive penalties charged to this player.",
        "how": "Filter play-by-play where penalty_player_name matches, restricted to OL penalty types (holding, false start, illegal formation, etc.).",
        "limits": "Raw counts ignore context — a holding that wipes out a 40-yard gain counts the same as one on a 2-yard loss. Penalty Leverage Cost addresses this.",
    },
    "z_penalty_rate": {
        "what": "Penalties per offensive snap.",
        "how": "Total penalties divided by offense snaps.",
        "limits": "Season-rate smoothing means one bad game can move the number meaningfully.",
    },
    "z_gap_success_rate": {
        "what": "Success rate on runs through this lineman's assigned gap.",
        "how": "Filter Lions runs to the gap owned by this player's position (strict attribution), then take the mean of nflverse's built-in 'success' field.",
        "limits": "Gap attribution is approximate — linemen pull and combo-block on plays the play-by-play doesn't know about. For stable starting fives like the 2024 Lions, it's a strong proxy. Guards get smaller samples because most interior runs get coded as 'middle' rather than 'guard'.",
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
        "limits": "The baseline model is deliberately simple for transparency. It has R² ~0.04 because run success is inherently noisy, so GARSR is close to raw gap success rate with a small situational adjustment.",
    },
    "z_rb_adjusted_gap_epa": {
        "what": "Gap EPA minus what you'd expect from the backs who ran through it.",
        "how": "For each gap run, compute (actual EPA) - (that rusher's season average EPA per carry). Average the residuals.",
        "limits": "Adjusts for rusher quality but not for situational mix.",
    },
    "z_penalty_leverage_cost": {
        "what": "Total EPA cost of penalties committed by this player.",
        "how": "Sum the nflverse EPA value on each penalty play attributed to the player.",
        "limits": "Leverage weighting mixes talent measurement with situational importance — you may disagree with that framing, and you can weight this low if so.",
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
        "what": "EXPERIMENTAL. Rough inference of how well this guard does on plays where a pull is likely.",
        "how": "Success rate on runs to the opposite side of where the guard lines up, minus success rate on same-side runs.",
        "limits": "We can't actually see which plays involved pulls. This uses direction as a noisy proxy.",
    },
    "z_leverage_rating": {
        "what": "EXPERIMENTAL. Gap run EPA weighted by each play's win probability impact.",
        "how": "sum(EPA × |WPA|) / sum(|WPA|). Plays in close games get weighted more than blowouts.",
        "limits": "Leverage weighting is philosophically contested — it mixes 'performed well' with 'performed well in close games.'",
    },
    "z_pass_run_balance": {
        "what": "EXPERIMENTAL. Whether this player is better at pass protection or run blocking, relative to peers.",
        "how": "z-score of Pass Pro On/Off Split minus z-score of Gap Run Success Rate.",
        "limits": "Derived from other stats, so it inherits all their weaknesses.",
    },
}


# ─── Supabase ────────────────────────────────────────────────────────────────

@st.cache_resource
def get_supabase():
    if not SUPABASE_AVAILABLE:
        return None
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception:
        return None


# ─── Data ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_wr_data():
    if not WR_DATA_PATH.exists():
        return None
    return pd.read_parquet(WR_DATA_PATH)


@st.cache_data
def load_ol_data():
    if not OL_DATA_PATH.exists():
        return None, None
    df = pd.read_parquet(OL_DATA_PATH)
    if OL_METADATA_PATH.exists():
        with open(OL_METADATA_PATH) as f:
            meta = json.load(f)
    else:
        meta = {}
    return df, meta


# ─── Scoring ─────────────────────────────────────────────────────────────────

def score_players(df, weights, mode, stat_columns, bundles):
    out = df.copy()

    if mode == "bundle":
        stat_weights = {col: 0.0 for col in stat_columns}
        for bundle, w in weights.items():
            stats = bundles.get(bundle, [])
            if not stats or w == 0:
                continue
            per_stat = w / len(stats)
            for s in stats:
                if s in stat_weights:
                    stat_weights[s] += per_stat
    else:
        stat_weights = {col: weights.get(col, 0.0) for col in stat_columns}

    total_weight = sum(abs(w) for w in stat_weights.values())
    if total_weight == 0:
        out["score"] = 0.0
        return out

    score = np.zeros(len(out))
    for col, w in stat_weights.items():
        if w == 0 or col not in out.columns:
            continue
        col_vals = out[col].fillna(0).to_numpy()
        score += col_vals * w

    out["score"] = score / total_weight
    return out


def score_label(score):
    """Convert a numeric score into a plain-English description.

    Scores are weighted averages of z-scores, so they're already in
    standard-deviation units. 0 = group average. +1 = one std dev above. Etc.
    """
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


def format_score_with_label(score):
    if pd.isna(score):
        return "—"
    sign = "+" if score >= 0 else ""
    return f"{sign}{score:.2f} ({score_label(score)})"


SCORE_EXPLAINER_MD = """
**What this number actually means.** The score is a weighted average of
z-scores — standardized stats where 0 is the group average, +1 is one
standard deviation above the group, and −1 is one standard deviation below.
Your slider weights control how much each bundle (or each stat, in Advanced
mode) contributes.

**How to read a score:**
- `+1.0` or higher → well above the group average on the stats you weighted
- `+0.4` to `+1.0` → above average
- `−0.4` to `+0.4` → roughly average
- `−1.0` or lower → well below average

**What this is not.** It's not a PFF-style 0-to-100 grade. It's not an
absolute rating of how good a player is in some universal sense. It's a
**comparative** number that tells you how each player stacks up against the
others in the group, given the methodology *you* chose. Change the weights
and the scores change. That's the point — this is your rating, not ours.

For details on any individual stat, switch to Advanced mode and click the ℹ️
methodology button next to it. Every stat has its formula and known
limitations documented.
"""


# ─── Community algorithms (namespaced by position) ───────────────────────────

def slugify(name):
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s-]+", "-", s)
    return s[:60] or f"algo-{uuid.uuid4().hex[:8]}"


def save_algorithm(position, name, author, description, weights, mode, tiers_enabled=None):
    sb = get_supabase()
    if sb is None:
        return False, "Supabase not configured."
    base_slug = f"{position}-{slugify(name)}"
    slug = base_slug
    payload = {
        "slug": slug,
        "name": name.strip(),
        "author": (author or "anonymous").strip(),
        "description": (description or "").strip(),
        "bundle_weights": {
            "position": position,
            "mode": mode,
            "weights": weights,
            "tiers_enabled": tiers_enabled or [1, 2, 3, 4],
        },
        "upvotes": 0,
    }
    for _ in range(3):
        try:
            sb.table("algorithms").insert(payload).execute()
            return True, slug
        except Exception as e:
            msg = str(e)
            if "duplicate" in msg.lower() or "unique" in msg.lower():
                slug = f"{base_slug}-{uuid.uuid4().hex[:4]}"
                payload["slug"] = slug
                continue
            return False, msg
    return False, "Could not generate unique slug."


def list_algorithms(position=None, sort="upvotes"):
    sb = get_supabase()
    if sb is None:
        return []
    try:
        order_col = "upvotes" if sort == "upvotes" else "created_at"
        res = sb.table("algorithms").select("*").order(order_col, desc=True).limit(200).execute()
        algos = res.data or []
        if position:
            filtered = []
            for a in algos:
                bw = a.get("bundle_weights") or {}
                algo_pos = bw.get("position") if isinstance(bw, dict) else None
                if algo_pos == position:
                    filtered.append(a)
                elif algo_pos is None and position == "wr":
                    # Legacy receiver algorithms from Stage 4 have no position field
                    filtered.append(a)
            return filtered
        return algos
    except Exception as e:
        st.warning(f"Couldn't load algorithms: {e}")
        return []


def upvote_algorithm(slug):
    sb = get_supabase()
    if sb is None:
        return False
    try:
        res = sb.table("algorithms").select("upvotes").eq("slug", slug).limit(1).execute()
        if not res.data:
            return False
        new_count = (res.data[0].get("upvotes") or 0) + 1
        sb.table("algorithms").update({"upvotes": new_count}).eq("slug", slug).execute()
        return True
    except Exception as e:
        st.warning(f"Upvote failed: {e}")
        return False


# ─── Session state ───────────────────────────────────────────────────────────

def init_state():
    if "wr_advanced_mode" not in st.session_state:
        st.session_state.wr_advanced_mode = False
    if "wr_bundle_weights" not in st.session_state:
        st.session_state.wr_bundle_weights = {b: 50 for b in WR_BUNDLES.keys()}
    if "wr_stat_weights" not in st.session_state:
        st.session_state.wr_stat_weights = {s: 0 for s in WR_STAT_COLUMNS}
    if "wr_loaded_algo" not in st.session_state:
        st.session_state.wr_loaded_algo = None

    if "ol_advanced_mode" not in st.session_state:
        st.session_state.ol_advanced_mode = False
    if "ol_bundle_weights" not in st.session_state:
        st.session_state.ol_bundle_weights = {b: 50 for b in OL_BUNDLES.keys()}
    if "ol_stat_weights" not in st.session_state:
        st.session_state.ol_stat_weights = {}
    if "ol_tiers_enabled" not in st.session_state:
        st.session_state.ol_tiers_enabled = [1, 2, 3]  # Tier 4 off by default
    if "ol_loaded_algo" not in st.session_state:
        st.session_state.ol_loaded_algo = None


def apply_loaded_algorithm_wr(algo):
    bw = algo.get("bundle_weights") or {}
    if isinstance(bw, dict) and "weights" in bw:
        weights = bw["weights"]
        mode = bw.get("mode", "bundle")
    else:
        weights = bw if isinstance(bw, dict) else {}
        mode = "bundle" if any(k in WR_BUNDLES for k in weights.keys()) else "advanced"

    if mode == "advanced":
        for s in WR_STAT_COLUMNS:
            st.session_state.wr_stat_weights[s] = int(weights.get(s, 0))
        st.session_state.wr_advanced_mode = True
    else:
        for b in WR_BUNDLES.keys():
            st.session_state.wr_bundle_weights[b] = int(weights.get(b, 0))
        st.session_state.wr_advanced_mode = False
    st.session_state.wr_loaded_algo = algo.get("name")


def apply_loaded_algorithm_ol(algo):
    bw = algo.get("bundle_weights") or {}
    if not isinstance(bw, dict):
        return
    weights = bw.get("weights", {})
    mode = bw.get("mode", "bundle")
    tiers = bw.get("tiers_enabled", [1, 2, 3, 4])
    if mode == "advanced":
        st.session_state.ol_stat_weights = {k: int(v) for k, v in weights.items()}
        st.session_state.ol_advanced_mode = True
    else:
        for b in OL_BUNDLES.keys():
            st.session_state.ol_bundle_weights[b] = int(weights.get(b, 0))
        st.session_state.ol_advanced_mode = False
    st.session_state.ol_tiers_enabled = tiers
    st.session_state.ol_loaded_algo = algo.get("name")


# ─── Tier helpers ────────────────────────────────────────────────────────────

TIER_LABELS = {
    1: "Tier 1 — Counted",
    2: "Tier 2 — Contextualized",
    3: "Tier 3 — Adjusted",
    4: "Tier 4 — Inferred",
}
TIER_DESCRIPTIONS = {
    1: "Pure recorded facts. No modeling.",
    2: "Raw counts divided by opportunity. Still no modeling.",
    3: "Compared against a modeled baseline. Model is simple and visible.",
    4: "Inferred from patterns the data can't directly see. Use with skepticism.",
}


def tier_badge(tier):
    colors = {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠"}
    return colors.get(tier, "⚪")


# ─── Receivers page (Stage 4, deprecation fixes) ─────────────────────────────

def page_receivers(df):
    st.title("🦁 Lions Receiver Rater")
    st.caption("Build your own receiver rating. Adjust the sliders, see who comes out on top.")

    if st.session_state.wr_loaded_algo:
        st.info(f"Loaded algorithm: **{st.session_state.wr_loaded_algo}**")

    st.session_state.wr_advanced_mode = st.toggle(
        "Advanced mode (individual stats)",
        value=st.session_state.wr_advanced_mode,
        key="wr_adv_toggle",
    )

    if st.session_state.wr_advanced_mode:
        st.subheader("Stat weights")
        cols = st.columns(2)
        for i, stat in enumerate(WR_STAT_COLUMNS):
            with cols[i % 2]:
                st.session_state.wr_stat_weights[stat] = st.slider(
                    WR_STAT_LABELS.get(stat, stat),
                    min_value=-100, max_value=100,
                    value=int(st.session_state.wr_stat_weights.get(stat, 0)),
                    key=f"wr_stat_{stat}",
                )
        weights = st.session_state.wr_stat_weights
        mode = "advanced"
    else:
        st.subheader("What matters to you?")
        for bundle in WR_BUNDLES.keys():
            st.session_state.wr_bundle_weights[bundle] = st.slider(
                f"**{bundle}** — {WR_BUNDLE_DESCRIPTIONS[bundle]}",
                min_value=0, max_value=100,
                value=int(st.session_state.wr_bundle_weights.get(bundle, 50)),
                key=f"wr_bundle_{bundle}",
            )
        weights = st.session_state.wr_bundle_weights
        mode = "bundle"

    scored = score_players(df, weights, mode, WR_STAT_COLUMNS, WR_BUNDLES)
    name_col = next(
        (c for c in ["player_name", "player", "name", "full_name"] if c in scored.columns),
        scored.columns[0],
    )
    extra = [c for c in ["season", "team", "position"] if c in scored.columns]
    scored["Score"] = scored["score"].apply(format_score_with_label)
    display_cols = [name_col] + extra + ["Score"]
    leaderboard = (
        scored.sort_values("score", ascending=False)
        .reset_index(drop=True)
        .head(25)[display_cols]
    )
    leaderboard.index = leaderboard.index + 1

    st.subheader("Leaderboard")
    st.dataframe(leaderboard, width="stretch")

    with st.expander("ℹ️ How is this score calculated?"):
        st.markdown(SCORE_EXPLAINER_MD)

    st.divider()
    with st.expander("💾 Save this algorithm", expanded=False):
        sb = get_supabase()
        if sb is None:
            st.warning("Saving unavailable — Supabase not configured.")
        else:
            with st.form("wr_save_form"):
                name = st.text_input("Name", max_chars=80)
                author = st.text_input("Your name", max_chars=40)
                desc = st.text_area("Description", max_chars=500)
                if st.form_submit_button("Save"):
                    if not name.strip():
                        st.error("Name required.")
                    else:
                        ok, result = save_algorithm("wr", name, author, desc, weights, mode)
                        if ok:
                            st.success(f"Saved! Slug: `{result}`")
                            st.balloons()
                        else:
                            st.error(f"Save failed: {result}")


# ─── Offensive Line page ─────────────────────────────────────────────────────

def page_offensive_line(df, meta):
    st.title("🦁 Lions Offensive Line Rater")
    st.caption(
        "Build your own OL rating. Transparency-first: every stat has a "
        "methodology popover, and you choose how speculative you want to get."
    )

    ctx = meta.get("team_context", {}) if meta else {}
    if ctx:
        st.markdown("### How did the line perform as a unit?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if ctx.get("lions_ybc_per_att") is not None:
                st.metric(
                    "Yards before contact / att",
                    f"{ctx['lions_ybc_per_att']:.2f}",
                    delta=f"{ctx['lions_ybc_per_att'] - ctx.get('league_ybc_per_att', 0):+.2f} vs league",
                )
        with col2:
            if ctx.get("lions_yac_per_att") is not None:
                st.metric(
                    "Yards after contact / att",
                    f"{ctx['lions_yac_per_att']:.2f}",
                    delta=f"{ctx['lions_yac_per_att'] - ctx.get('league_yac_per_att', 0):+.2f} vs league",
                )
        with col3:
            if ctx.get("lions_sack_rate") is not None:
                delta_val = ctx['lions_sack_rate'] - ctx.get('league_sack_rate', 0)
                st.metric(
                    "Sack rate",
                    f"{ctx['lions_sack_rate']:.1%}",
                    delta=f"{delta_val:+.1%} vs league",
                    delta_color="inverse",
                )
        st.caption(
            "Team-level numbers for the whole OL. Individual ratings below "
            "attribute play-by-play results to specific linemen by position."
        )
        st.divider()

    if st.session_state.ol_loaded_algo:
        st.info(f"Loaded algorithm: **{st.session_state.ol_loaded_algo}** (adjust to fork)")

    stat_tiers = meta.get("stat_tiers", {}) if meta else {}
    stat_labels = meta.get("stat_labels", {}) if meta else {}

    # Tier filter
    st.markdown("### How speculative do you want to get?")
    st.caption(
        "Each stat is labeled by how much trust it asks from you. "
        "Uncheck tiers you don't want. This is philosophy in a checkbox."
    )
    tier_cols = st.columns(4)
    new_enabled = []
    for i, tier in enumerate([1, 2, 3, 4]):
        with tier_cols[i]:
            checked = st.checkbox(
                f"{tier_badge(tier)} {TIER_LABELS[tier]}",
                value=(tier in st.session_state.ol_tiers_enabled),
                help=TIER_DESCRIPTIONS[tier],
                key=f"ol_tier_{tier}",
            )
            if checked:
                new_enabled.append(tier)
    st.session_state.ol_tiers_enabled = new_enabled

    if not new_enabled:
        st.warning("Enable at least one tier to see ratings.")
        return

    all_ol_stats = list(stat_tiers.keys())
    enabled_stats = [s for s in all_ol_stats if stat_tiers.get(s, 1) in new_enabled]

    st.divider()

    st.session_state.ol_advanced_mode = st.toggle(
        "Advanced mode (individual stats)",
        value=st.session_state.ol_advanced_mode,
        key="ol_adv_toggle",
    )

    if st.session_state.ol_advanced_mode:
        st.subheader("Stat weights")
        st.caption("Tier badge next to each stat. Click ℹ️ for methodology.")
        for stat in enabled_stats:
            tier = stat_tiers.get(stat, 1)
            label = stat_labels.get(stat, stat)
            methodology = OL_METHODOLOGY.get(stat, {})

            row = st.columns([3, 1])
            with row[0]:
                current = int(st.session_state.ol_stat_weights.get(stat, 0))
                st.session_state.ol_stat_weights[stat] = st.slider(
                    f"{tier_badge(tier)} {label}",
                    min_value=-100, max_value=100,
                    value=current,
                    key=f"ol_stat_{stat}",
                )
            with row[1]:
                with st.popover("ℹ️ methodology"):
                    st.markdown(f"**{label}** — {TIER_LABELS[tier]}")
                    if methodology:
                        st.markdown(f"**What it measures:** {methodology.get('what', '')}")
                        st.markdown(f"**How it's computed:** {methodology.get('how', '')}")
                        st.markdown(f"**Known limits:** {methodology.get('limits', '')}")
        weights = st.session_state.ol_stat_weights
        mode = "advanced"
    else:
        st.subheader("What matters to you?")
        for bundle, bundle_stats in OL_BUNDLES.items():
            bundle_enabled = [s for s in bundle_stats if s in enabled_stats]
            if not bundle_enabled:
                continue

            tier_counts = {}
            for s in bundle_enabled:
                t = stat_tiers.get(s, 1)
                tier_counts[t] = tier_counts.get(t, 0) + 1
            tier_summary = " ".join(f"{tier_badge(t)}×{c}" for t, c in sorted(tier_counts.items()))

            st.session_state.ol_bundle_weights[bundle] = st.slider(
                f"**{bundle}** — {OL_BUNDLE_DESCRIPTIONS[bundle]}  \n*{tier_summary}*",
                min_value=0, max_value=100,
                value=int(st.session_state.ol_bundle_weights.get(bundle, 50)),
                key=f"ol_bundle_{bundle}",
            )
        weights = st.session_state.ol_bundle_weights
        mode = "bundle"

    filtered_bundles = {
        b: [s for s in stats if s in enabled_stats]
        for b, stats in OL_BUNDLES.items()
    }
    filtered_bundles = {b: s for b, s in filtered_bundles.items() if s}

    scored = score_players(df, weights, mode, enabled_stats, filtered_bundles)

    scored["Score"] = scored["score"].apply(format_score_with_label)
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
        st.markdown(SCORE_EXPLAINER_MD)
        st.markdown(
            "**One more thing for the OL page specifically:** scores here "
            "are computed within the Lions starting five (n=5), so z-scores "
            "are noisier than they'd be with a league-wide sample. A single "
            "player can shift the distribution meaningfully. Treat directional "
            "differences seriously, but don't over-read small gaps between players."
        )

    st.divider()
    with st.expander("💾 Save this algorithm", expanded=False):
        sb = get_supabase()
        if sb is None:
            st.warning("Saving unavailable — Supabase not configured.")
        else:
            with st.form("ol_save_form"):
                name = st.text_input("Name", max_chars=80)
                author = st.text_input("Your name", max_chars=40)
                desc = st.text_area("Description — what does this algorithm value, and why?", max_chars=500)
                if st.form_submit_button("Save"):
                    if not name.strip():
                        st.error("Name required.")
                    else:
                        ok, result = save_algorithm(
                            "ol", name, author, desc, weights, mode,
                            tiers_enabled=new_enabled,
                        )
                        if ok:
                            st.success(f"Saved! Slug: `{result}`")
                            st.balloons()
                        else:
                            st.error(f"Save failed: {result}")


# ─── Community page ──────────────────────────────────────────────────────────

def page_browse():
    st.title("🏆 Community algorithms")
    st.caption("Browse what other fans have built.")

    sb = get_supabase()
    if sb is None:
        st.warning("Community browsing unavailable — Supabase not configured.")
        return

    pos_filter = st.radio("Position", ["Receivers", "Offensive Line", "All"], horizontal=True)
    sort = st.radio("Sort by", ["Top upvoted", "Newest"], horizontal=True)
    sort_key = "upvotes" if sort == "Top upvoted" else "created_at"

    pos_map = {"Receivers": "wr", "Offensive Line": "ol", "All": None}
    algos = list_algorithms(position=pos_map[pos_filter], sort=sort_key)

    if not algos:
        st.info("No algorithms yet — be the first to save one!")
        return

    for algo in algos:
        with st.container(border=True):
            top = st.columns([5, 1, 1])
            with top[0]:
                bw = algo.get("bundle_weights") or {}
                pos = bw.get("position", "wr") if isinstance(bw, dict) else "wr"
                pos_badge = "🎯 WR" if pos == "wr" else "🛡️ OL"
                st.markdown(f"### {pos_badge} · {algo.get('name', 'Untitled')}")
                st.caption(
                    f"by **{algo.get('author') or 'anonymous'}** · "
                    f"{algo.get('created_at', '')[:10]}"
                )
                if algo.get("description"):
                    st.write(algo["description"])
                if isinstance(bw, dict) and pos == "ol":
                    tiers = bw.get("tiers_enabled", [1, 2, 3, 4])
                    tier_str = " ".join(tier_badge(t) for t in sorted(tiers))
                    st.caption(f"Tiers used: {tier_str}")
            with top[1]:
                st.metric("👍", algo.get("upvotes") or 0)
            with top[2]:
                if st.button("Upvote", key=f"up_{algo['slug']}"):
                    if upvote_algorithm(algo["slug"]):
                        st.rerun()

            actions = st.columns([1, 1, 4])
            with actions[0]:
                if st.button("Load & fork", key=f"load_{algo['slug']}"):
                    if pos == "ol":
                        apply_loaded_algorithm_ol(algo)
                        st.success("Loaded into OL rater — switch tabs.")
                    else:
                        apply_loaded_algorithm_wr(algo)
                        st.success("Loaded into Receiver rater — switch tabs.")
            with actions[1]:
                with st.popover("View weights"):
                    st.json(bw if isinstance(bw, dict) else {})


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    init_state()

    with st.sidebar:
        st.markdown("## 🦁 Lions Rater")
        st.caption("A transparent alternative to PFF — built by fans, for fans.")
        page = st.radio("Navigate", ["Receivers", "Offensive Line", "Community algorithms"])
        st.divider()
        st.markdown(
            "**Ethos:** strive for the greatest accuracy while being "
            "transparent about limitations and how we addressed them."
        )
        st.markdown(
            "**The vision:** a Wikipedia of athletic performance. "
            "Build your own rating, share it, see how others see the game."
        )

    if page == "Receivers":
        df = load_wr_data()
        if df is None:
            st.error(f"Receiver data not found at {WR_DATA_PATH}")
            return
        page_receivers(df)
    elif page == "Offensive Line":
        df, meta = load_ol_data()
        if df is None:
            st.error(f"OL data not found at {OL_DATA_PATH}")
            st.caption("Upload the parquet + metadata files to the repo.")
            return
        page_offensive_line(df, meta)
    else:
        page_browse()


if __name__ == "__main__":
    main()

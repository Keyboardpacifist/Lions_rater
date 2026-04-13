"""
Lions Receiver Rater — Stage 4
A transparent, customizable receiver rating tool.
Fans build and share their own rating methodologies.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
import uuid

# Supabase is optional at import time so the app still loads if secrets are missing
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lions Receiver Rater",
    page_icon="🦁",
    layout="wide",
)

DATA_PATH = Path("data/master_lions_with_z.parquet")

# The 11 individual stats (z-score columns) shown in Advanced mode.
# These should match the z-score columns in your parquet file.
STAT_COLUMNS = [
    "z_catch_rate",
    "z_yards_per_route",
    "z_yards_per_reception",
    "z_air_yards_per_target",
    "z_adot",
    "z_targets_per_game",
    "z_receptions_per_game",
    "z_yac_per_reception",
    "z_broken_tackles_per_reception",
    "z_first_downs_per_reception",
    "z_td_rate",
]

# Plain-English labels for the 11 stats
STAT_LABELS = {
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

# The 5 bundles, each mapped to the underlying stats it averages.
BUNDLES = {
    "Reliability": [
        "z_catch_rate",
        "z_first_downs_per_reception",
    ],
    "Explosive plays": [
        "z_yards_per_reception",
        "z_td_rate",
    ],
    "Field stretcher": [
        "z_air_yards_per_target",
        "z_adot",
    ],
    "Volume & usage": [
        "z_targets_per_game",
        "z_receptions_per_game",
        "z_yards_per_route",
    ],
    "After the catch": [
        "z_yac_per_reception",
        "z_broken_tackles_per_reception",
    ],
}

BUNDLE_DESCRIPTIONS = {
    "Reliability": "Catches the ball, moves the chains.",
    "Explosive plays": "Big gains and touchdowns.",
    "Field stretcher": "Gets targeted deep down the field.",
    "Volume & usage": "How often they're involved in the offense.",
    "After the catch": "What they do once the ball is in their hands.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Supabase client
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_supabase() -> "Client | None":
    if not SUPABASE_AVAILABLE:
        return None
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Data file not found at {DATA_PATH}")
        st.stop()
    return pd.read_parquet(DATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_players(
    df: pd.DataFrame,
    weights: dict,
    mode: str = "bundle",
) -> pd.DataFrame:
    """
    weights: dict
      - if mode == 'bundle': {bundle_name: weight}
      - if mode == 'advanced': {stat_column: weight}
    Returns df with a new 'score' column.
    """
    out = df.copy()

    if mode == "bundle":
        # Expand each bundle weight across its constituent stats.
        stat_weights = {col: 0.0 for col in STAT_COLUMNS}
        for bundle, w in weights.items():
            stats = BUNDLES.get(bundle, [])
            if not stats or w == 0:
                continue
            per_stat = w / len(stats)
            for s in stats:
                if s in stat_weights:
                    stat_weights[s] += per_stat
    else:
        stat_weights = {col: weights.get(col, 0.0) for col in STAT_COLUMNS}

    total_weight = sum(abs(w) for w in stat_weights.values())
    if total_weight == 0:
        out["score"] = 0.0
        return out

    score = np.zeros(len(out))
    for col, w in stat_weights.items():
        if w == 0:
            continue
        if col not in out.columns:
            continue
        col_vals = out[col].fillna(0).to_numpy()
        score += col_vals * w

    out["score"] = score / total_weight
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm storage (Supabase)
# ─────────────────────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s-]+", "-", s)
    return s[:60] or f"algo-{uuid.uuid4().hex[:8]}"


def save_algorithm(name, author, description, bundle_weights) -> tuple[bool, str]:
    sb = get_supabase()
    if sb is None:
        return False, "Supabase not configured."
    base_slug = slugify(name)
    slug = base_slug
    # Handle collisions by appending a short hash
    for _ in range(3):
        try:
            sb.table("algorithms").insert({
                "slug": slug,
                "name": name.strip(),
                "author": (author or "anonymous").strip(),
                "description": (description or "").strip(),
                "bundle_weights": bundle_weights,
                "upvotes": 0,
            }).execute()
            return True, slug
        except Exception as e:
            msg = str(e)
            if "duplicate" in msg.lower() or "unique" in msg.lower():
                slug = f"{base_slug}-{uuid.uuid4().hex[:4]}"
                continue
            return False, msg
    return False, "Could not generate a unique slug."


def list_algorithms(sort: str = "upvotes") -> list[dict]:
    sb = get_supabase()
    if sb is None:
        return []
    try:
        order_col = "upvotes" if sort == "upvotes" else "created_at"
        res = (
            sb.table("algorithms")
            .select("*")
            .order(order_col, desc=True)
            .limit(100)
            .execute()
        )
        return res.data or []
    except Exception as e:
        st.warning(f"Couldn't load algorithms: {e}")
        return []


def get_algorithm(slug: str) -> dict | None:
    sb = get_supabase()
    if sb is None:
        return None
    try:
        res = sb.table("algorithms").select("*").eq("slug", slug).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception:
        return None


def upvote_algorithm(slug: str) -> bool:
    sb = get_supabase()
    if sb is None:
        return False
    try:
        current = get_algorithm(slug)
        if not current:
            return False
        new_count = (current.get("upvotes") or 0) + 1
        sb.table("algorithms").update({"upvotes": new_count}).eq("slug", slug).execute()
        return True
    except Exception as e:
        st.warning(f"Upvote failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    if "advanced_mode" not in st.session_state:
        st.session_state.advanced_mode = False
    if "bundle_weights" not in st.session_state:
        st.session_state.bundle_weights = {b: 50 for b in BUNDLES.keys()}
    if "stat_weights" not in st.session_state:
        st.session_state.stat_weights = {s: 0 for s in STAT_COLUMNS}
    if "loaded_algo_name" not in st.session_state:
        st.session_state.loaded_algo_name = None


def apply_loaded_algorithm(algo: dict):
    """Set sliders to match a loaded algorithm."""
    bw = algo.get("bundle_weights") or {}
    if isinstance(bw, dict):
        # If it stores bundle weights
        if any(k in BUNDLES for k in bw.keys()):
            for b in BUNDLES.keys():
                st.session_state.bundle_weights[b] = int(bw.get(b, 0))
            st.session_state.advanced_mode = False
        else:
            # It stores stat weights (advanced mode algorithm)
            for s in STAT_COLUMNS:
                st.session_state.stat_weights[s] = int(bw.get(s, 0))
            st.session_state.advanced_mode = True
    st.session_state.loaded_algo_name = algo.get("name")


# ─────────────────────────────────────────────────────────────────────────────
# UI: Pages
# ─────────────────────────────────────────────────────────────────────────────

def page_rater(df: pd.DataFrame):
    st.title("🦁 Lions Receiver Rater")
    st.caption(
        "Build your own receiver rating. Adjust the sliders, "
        "see who comes out on top, then save and share your formula."
    )

    if st.session_state.loaded_algo_name:
        st.info(f"Loaded algorithm: **{st.session_state.loaded_algo_name}** "
                f"(adjust sliders to fork it)")

    # Mode toggle
    st.session_state.advanced_mode = st.toggle(
        "Advanced mode (individual stats)",
        value=st.session_state.advanced_mode,
    )

    # ── Sliders ────────────────────────────────────────────────────────────
    if st.session_state.advanced_mode:
        st.subheader("Stat weights")
        cols = st.columns(2)
        for i, stat in enumerate(STAT_COLUMNS):
            with cols[i % 2]:
                st.session_state.stat_weights[stat] = st.slider(
                    STAT_LABELS.get(stat, stat),
                    min_value=-100, max_value=100,
                    value=int(st.session_state.stat_weights.get(stat, 0)),
                    key=f"stat_{stat}",
                )
        weights = st.session_state.stat_weights
        mode = "advanced"
    else:
        st.subheader("What matters to you?")
        for bundle in BUNDLES.keys():
            st.session_state.bundle_weights[bundle] = st.slider(
                f"**{bundle}** — {BUNDLE_DESCRIPTIONS[bundle]}",
                min_value=0, max_value=100,
                value=int(st.session_state.bundle_weights.get(bundle, 50)),
                key=f"bundle_{bundle}",
            )
        weights = st.session_state.bundle_weights
        mode = "bundle"

    # ── Score & display ────────────────────────────────────────────────────
    scored = score_players(df, weights, mode=mode)

    # Pick a sensible name column
    name_col = next(
        (c for c in ["player_name", "player", "name", "full_name"] if c in scored.columns),
        scored.columns[0],
    )

    display_cols = [name_col, "score"]
    extra_cols = [c for c in ["season", "team", "position"] if c in scored.columns]
    display_cols = [name_col] + extra_cols + ["score"]

    leaderboard = (
        scored[display_cols]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
        .head(25)
    )
    leaderboard.index = leaderboard.index + 1

    st.subheader("Leaderboard")
    st.dataframe(leaderboard, use_container_width=True)

    # ── Save algorithm ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("💾 Save this algorithm", expanded=False):
        sb = get_supabase()
        if sb is None:
            st.warning(
                "Saving is unavailable — Supabase isn't configured. "
                "Check that SUPABASE_URL and SUPABASE_KEY are set in Streamlit secrets."
            )
        else:
            with st.form("save_form", clear_on_submit=False):
                algo_name = st.text_input("Name your algorithm", max_chars=80)
                algo_author = st.text_input("Your name (or 'anonymous')", max_chars=40)
                algo_desc = st.text_area(
                    "Description — what does this algorithm value?",
                    max_chars=500,
                )
                submitted = st.form_submit_button("Save to community")
                if submitted:
                    if not algo_name.strip():
                        st.error("Please give your algorithm a name.")
                    else:
                        ok, result = save_algorithm(
                            name=algo_name,
                            author=algo_author,
                            description=algo_desc,
                            bundle_weights=weights,
                        )
                        if ok:
                            st.success(f"Saved! Slug: `{result}`")
                            st.balloons()
                        else:
                            st.error(f"Save failed: {result}")


def page_browse():
    st.title("🏆 Community algorithms")
    st.caption("Browse what other fans have built. Click any to load and fork.")

    sb = get_supabase()
    if sb is None:
        st.warning(
            "Community browsing is unavailable — Supabase isn't configured. "
            "Check that SUPABASE_URL and SUPABASE_KEY are set in Streamlit secrets."
        )
        return

    sort = st.radio("Sort by", ["Top upvoted", "Newest"], horizontal=True)
    sort_key = "upvotes" if sort == "Top upvoted" else "created_at"

    algos = list_algorithms(sort=sort_key)

    if not algos:
        st.info("No algorithms yet — be the first to save one!")
        return

    for algo in algos:
        with st.container(border=True):
            top = st.columns([5, 1, 1])
            with top[0]:
                st.markdown(f"### {algo.get('name', 'Untitled')}")
                st.caption(
                    f"by **{algo.get('author') or 'anonymous'}** · "
                    f"{algo.get('created_at', '')[:10]}"
                )
                desc = algo.get("description") or ""
                if desc:
                    st.write(desc)
            with top[1]:
                st.metric("👍", algo.get("upvotes") or 0)
            with top[2]:
                if st.button("Upvote", key=f"up_{algo['slug']}"):
                    if upvote_algorithm(algo["slug"]):
                        st.rerun()

            actions = st.columns([1, 1, 4])
            with actions[0]:
                if st.button("Load & fork", key=f"load_{algo['slug']}"):
                    apply_loaded_algorithm(algo)
                    st.success(f"Loaded '{algo['name']}' — switch to the Rater tab.")
            with actions[1]:
                with st.popover("View weights"):
                    st.json(algo.get("bundle_weights") or {})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    init_state()
    df = load_data()

    with st.sidebar:
        st.markdown("## 🦁 Lions Rater")
        st.caption("A transparent alternative to PFF — built by fans, for fans.")
        page = st.radio("Navigate", ["Rater", "Community algorithms"])
        st.divider()
        st.markdown(
            "**The vision:** a Wikipedia of athletic performance. "
            "Build your own rating, share it, see how others see the game."
        )

    if page == "Rater":
        page_rater(df)
    else:
        page_browse()


if __name__ == "__main__":
    main()

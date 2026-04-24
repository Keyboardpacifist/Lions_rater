"""
Shared helpers for the Lions Receiver/RB Rater pages.

Both pages/1_Receivers.py and pages/2_Running_backs.py import from here:
  - Supabase client + CRUD
  - Score computation given effective per-stat weights
  - Save/browse UI components, parameterized by position group

Keeping this in one place means slider math and Supabase logic
only live in one file.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any

import pandas as pd
import streamlit as st
from supabase import Client, create_client


# ============================================================
# Supabase
# ============================================================
@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "algo"
    short = hashlib.md5(f"{s}{time.time()}".encode()).hexdigest()[:6]
    return f"{s}-{short}"


def save_algorithm(
    name: str,
    author: str,
    description: str,
    bundle_weights: dict,
    position_group: str,
) -> dict | None:
    sb = get_supabase()
    row = {
        "slug": slugify(name),
        "name": name,
        "author": author or "Anonymous",
        "description": description,
        "bundle_weights": bundle_weights,
        "position_group": position_group,
        "upvotes": 0,
    }
    try:
        resp = sb.table("algorithms").insert(row).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        st.error(f"Save failed: {e}")
        return None


def list_algorithms(
    position_group: str,
    order_by: str = "upvotes",
    limit: int = 50,
) -> list[dict]:
    sb = get_supabase()
    try:
        resp = (
            sb.table("algorithms")
            .select("*")
            .eq("position_group", position_group)
            .order(order_by, desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def get_algorithm_by_slug(slug: str) -> dict | None:
    sb = get_supabase()
    try:
        resp = (
            sb.table("algorithms")
            .select("*")
            .eq("slug", slug)
            .limit(1)
            .execute()
        )
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def upvote_algorithm(algo_id: str, current: int) -> bool:
    sb = get_supabase()
    try:
        sb.table("algorithms").update(
            {"upvotes": current + 1}
        ).eq("id", algo_id).execute()
        return True
    except Exception:
        return False


# ============================================================
# Scoring
# ============================================================
def compute_effective_weights(
    bundles: dict, bundle_weights: dict
) -> dict[str, float]:
    """Turn user-facing bundle weights into per-z-stat effective weights."""
    eff: dict[str, float] = {}
    for bk, bw in bundle_weights.items():
        if bw == 0:
            continue
        for z_col, internal in bundles[bk]["stats"].items():
            eff[z_col] = eff.get(z_col, 0) + bw * internal
    return eff


def score_players(
    df: pd.DataFrame, effective_weights: dict[str, float]
) -> pd.DataFrame:
    """Add a 'score' column = weighted sum of z-stats."""
    total = sum(effective_weights.values())
    out = df.copy()
    if total == 0:
        out["score"] = 0.0
        return out
    score = pd.Series(0.0, index=out.index)
    for z_col, w in effective_weights.items():
        if w == 0 or z_col not in out.columns:
            continue
        score += out[z_col].fillna(0) * (w / total)
    out["score"] = score
    return out


# ============================================================
# Shared UI: page styling
# ============================================================
SHARED_CSS = """
<style>
h1, h2, h3 { color: #0076B6 !important; }
.stSlider [data-baseweb="slider"] > div > div > div > div {
    background-color: #0076B6;
}
.section-divider {
    border-top: 2px solid #B0B7BC;
    margin: 1.5rem 0 1rem 0;
}
.bundle-desc {
    font-size: 0.8rem;
    color: #6c757d;
    margin-top: -0.5rem;
    margin-bottom: 0.5rem;
}
.stDataFrame { margin-top: 0.5rem; }
.algo-card {
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    background: #fafbfc;
}
.algo-card h4 { margin: 0 0 0.25rem 0; color: #0076B6 !important; }
.algo-meta { font-size: 0.8rem; color: #6c757d; }
</style>
"""


def inject_css():
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


# ============================================================
# Shared UI: Community algorithms section
# ============================================================
def community_section(
    *,
    position_group: str,
    bundles: dict,
    bundle_weights: dict,
    advanced_mode: bool,
    page_url: str,
):
    """
    Render the save / browse / load / fork / upvote section.

    `position_group` is "receiver" or "rb" — used to scope Supabase
    queries so each page only sees its own algorithms.
    """
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("Community algorithms")

    tab_save, tab_browse = st.tabs(
        ["💾 Save your algorithm", "🌐 Browse community"]
    )

    # ---- Save tab ----
    with tab_save:
        if advanced_mode:
            st.info(
                "Saving is available in **bundle mode** (toggle off Advanced "
                "in the sidebar). This keeps saved algorithms portable and "
                "comparable."
            )
        else:
            st.markdown(
                "Happy with your slider positions? Save them so others can "
                "load, fork, and upvote your creation."
            )
            with st.form(f"save_algo_form_{position_group}"):
                name = st.text_input(
                    "Algorithm name", max_chars=80,
                    placeholder="e.g. The Gibbs Special",
                )
                author = st.text_input(
                    "Your name", max_chars=60, placeholder="Anonymous",
                )
                desc = st.text_area(
                    "Short description (optional)", max_chars=280,
                    placeholder="Prioritizes explosive YAC monsters over "
                                "safe possession guys.",
                )
                submitted = st.form_submit_button("Save algorithm")

            if submitted:
                if not name.strip():
                    st.warning("Give your algorithm a name first.")
                else:
                    saved = save_algorithm(
                        name=name.strip(),
                        author=author.strip(),
                        description=desc.strip(),
                        bundle_weights=bundle_weights,
                        position_group=position_group,
                    )
                    if saved:
                        slug = saved["slug"]
                        st.success(f"Saved! Slug: **{slug}**")
                        st.code(f"{page_url}?algo={slug}", language=None)
                        st.caption(
                            "Share that link — anyone who opens it will "
                            "load your weights."
                        )

    # ---- Browse tab ----
    with tab_browse:
        sort_col = st.selectbox(
            "Sort by",
            options=["upvotes", "created_at"],
            format_func=lambda x: "Most upvoted" if x == "upvotes" else "Newest first",
            key=f"browse_sort_{position_group}",
        )

        algos = list_algorithms(
            position_group=position_group, order_by=sort_col, limit=50
        )

        if not algos:
            st.caption(
                "No community algorithms yet for this position. "
                "Be the first to save one!"
            )
            return

        for algo in algos:
            st.markdown(
                f"<div class='algo-card'>"
                f"<h4>{algo['name']}</h4>"
                f"<div class='algo-meta'>"
                f"by {algo['author']} · {algo.get('upvotes', 0)} upvote(s)"
                f"</div>"
                f"<p style='margin:0.4rem 0 0 0; font-size:0.9rem;'>"
                f"{algo.get('description') or '<em>No description</em>'}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            bw = algo.get("bundle_weights") or {}
            labels = [
                f"{bundles[bk]['label']} {bv}"
                for bk, bv in bw.items()
                if bk in bundles and bv and bv > 0
            ]
            if labels:
                st.caption(" · ".join(labels))

            cols = st.columns([1, 1, 1, 4])

            with cols[0]:
                if st.button("Load", key=f"load_{position_group}_{algo['id']}"):
                    apply_algo_weights(algo, bundles)
                    st.rerun()

            with cols[1]:
                if st.button("Fork", key=f"fork_{position_group}_{algo['id']}"):
                    apply_algo_weights(algo, bundles)
                    st.session_state.loaded_algo = {
                        **algo,
                        "name": f"{algo['name']} (fork)",
                        "id": None,
                    }
                    st.rerun()

            with cols[2]:
                voted = algo["id"] in st.session_state.get("upvoted_ids", set())
                label = "Upvoted" if voted else "Upvote"
                if st.button(
                    f"👍 {label} ({algo.get('upvotes', 0)})",
                    key=f"vote_{position_group}_{algo['id']}",
                    disabled=voted,
                ):
                    if upvote_algorithm(algo["id"], algo.get("upvotes", 0)):
                        st.session_state.setdefault("upvoted_ids", set()).add(
                            algo["id"]
                        )
                        st.rerun()


def apply_algo_weights(algo: dict, bundles: dict):
    """Push an algorithm's bundle_weights into session_state slider keys."""
    bw = algo.get("bundle_weights") or {}
    for bk in bundles:
        st.session_state[f"bundle_{bk}"] = bw.get(bk, 0)
    st.session_state.loaded_algo = algo


# ──────────────────────────────────────────────────────────────────────
# Metric picker — third selector that lets fans sort a leaderboard by
# any single nerd metric instead of the user-weighted composite.
# ──────────────────────────────────────────────────────────────────────

def metric_picker(metrics, default_label="Your score", key="metric_picker",
                   label="🔍 Sort leaderboard by"):
    """Render a dropdown that lets the user pick a sort metric.

    Args:
        metrics: dict {display_label: (column_name, ascending)}.
                 e.g., {"Receiving yards": ("rec_yards", False),
                        "INT rate": ("int_rate", True)}
        default_label: which option to default to. "Your score" is always
                       inserted as the first option (sorts by composite score).
        key: streamlit widget key.
        label: dropdown label.

    Returns:
        (selected_label, column_name, ascending) — the column to sort by
        and the direction. When "Your score" is selected, returns
        ("Your score", "score", False).
    """
    import streamlit as st
    full = {"Your score": ("score", False), **metrics}
    options = list(full.keys())
    default_idx = options.index(default_label) if default_label in options else 0
    selected = st.selectbox(label, options=options, index=default_idx, key=key)
    col, ascending = full[selected]
    return selected, col, ascending


def radar_season_row(career_df, current_season, season_col="season_year",
                      key="radar_year", label="Radar season"):
    """Render a season dropdown above a radar chart and return the row
    (Series) to use for the radar values.

    Options: each season the player played, plus "All-career mean".
    For traded players (multiple stints in a season), the season's row
    is averaged across stints.

    Args:
        career_df: DataFrame containing all of this player's rows
                   (already filtered to the one player).
        current_season: the season currently selected on the page —
                        used as the default selection.
        season_col: column name for season year.
        key: streamlit widget key.
        label: dropdown label.

    Returns:
        pandas Series with the row values to plot.
    """
    import streamlit as st
    import pandas as pd

    if career_df is None or len(career_df) == 0 or season_col not in career_df.columns:
        return None

    seasons = sorted(set(int(s) for s in career_df[season_col].dropna().unique()), reverse=True)
    if not seasons:
        return None

    options = [int(s) for s in seasons]
    if len(seasons) > 1:
        options = options + ["All-career mean"]

    try:
        default_idx = options.index(int(current_season))
    except (ValueError, TypeError):
        default_idx = 0

    def _fmt(v):
        return f"Season {v}" if isinstance(v, int) else v

    selected = st.selectbox(label, options=options, index=default_idx,
                             key=key, format_func=_fmt)

    if selected == "All-career mean":
        numeric = career_df.select_dtypes(include="number").mean()
        return numeric

    season_rows = career_df[career_df[season_col] == selected]
    if len(season_rows) == 0:
        return career_df.iloc[0]
    if len(season_rows) == 1:
        return season_rows.iloc[0]
    return season_rows.select_dtypes(include="number").mean()

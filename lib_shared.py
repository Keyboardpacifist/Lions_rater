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
# Community trading-card gallery
# ============================================================
def save_card(*,
                player_id: str,
                player_name: str,
                position_group: str,
                team_abbr: str | None,
                season: int | None,
                season_label: str,
                bundle_weights: dict,
                score: float | None,
                author: str,
                caption: str,
                algorithm_id: str | None = None) -> dict | None:
    """Persist a (player, season, preset) trading card to the gallery.

    `cards` table — one row per shared card. Caller is responsible for
    rendering / regenerating the PNG itself; we only store the metadata
    needed to reproduce it.
    """
    sb = get_supabase()
    row = {
        "player_id": player_id,
        "player_name": player_name,
        "position_group": position_group,
        "team_abbr": team_abbr,
        "season": season,
        "season_label": season_label,
        "bundle_weights": bundle_weights,
        "score": float(score) if (score is not None and pd.notna(score)) else None,
        "author": author or "Anonymous",
        "caption": caption,
        "algorithm_id": algorithm_id,
    }
    try:
        resp = sb.table("cards").insert(row).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        st.error(f"Save failed: {e}")
        return None


def list_cards(*,
                position_group: str | None = None,
                team_abbr: str | None = None,
                order_by: str = "created_at",
                limit: int = 60) -> list[dict]:
    """Browse the gallery — optional position / team filters, with a
    sort axis (created_at | upvotes | score)."""
    sb = get_supabase()
    try:
        q = sb.table("cards").select("*")
        if position_group:
            q = q.eq("position_group", position_group)
        if team_abbr:
            q = q.eq("team_abbr", team_abbr)
        resp = q.order(order_by, desc=True).limit(limit).execute()
        return resp.data or []
    except Exception:
        return []


def upvote_card(card_id: str, current: int) -> bool:
    sb = get_supabase()
    try:
        sb.table("cards").update(
            {"upvotes": current + 1}
        ).eq("id", card_id).execute()
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


# ──────────────────────────────────────────────────────────────────────
# Master/detail click-to-detail leaderboard
# ──────────────────────────────────────────────────────────────────────

def render_master_detail_leaderboard(
    *,
    display_df,
    name_col,
    key_prefix,
    team,
    season,
    top_banner_html=None,
    top_banner_warn=None,
    leaderboard_caption=None,
    top_n=6,
):
    """Click-to-detail master/detail leaderboard for an NFL position page.

    Renders one of two views:
      - BROWSE: optional top banner + clickable-name leaderboard (capped
        to `top_n` with a "Show all" button) + caption.
      - DETAIL: a primary "← Back to leaderboard" button at the top.

    Returns:
      str | None — selected player name when in DETAIL view (caller
      should render the player's full detail card next), or None when in
      BROWSE view (caller should `st.stop()` to skip the detail card).

    Args:
      display_df: pd.DataFrame whose rows are leaderboard rows. Must
        contain a column matching `name_col` for the click handler.
        Other columns are rendered as plain markdown.
      name_col: column in `display_df` whose values are the player names
        used for the click button + the returned selection. Often
        "Player" if you've renamed the column for display, or the raw
        column name like "player_display_name".
      key_prefix: short string unique per page (e.g. "qb", "wr", "ol").
        Used to scope all session-state keys + widget keys.
      team, season: current page filter context. When either changes
        across reruns, all sticky detail markers for this page are
        cleared so the user lands back in browse for the new context.
      top_banner_html: optional HTML to render above the leaderboard
        in browse mode (e.g., "#1 of 23 — Player Name — +1.50" card).
      top_banner_warn: optional warning string shown right after the
        banner (e.g. small-sample warning for the #1 player).
      leaderboard_caption: optional caption rendered below the
        leaderboard table.
      top_n: row cap in browse mode before the "Show all" button
        appears. Defaults to 6 to match the College mode pattern.
    """
    import streamlit as st

    sel_key = f"{key_prefix}_selected_player_{team}_{season}"
    expand_key = f"{key_prefix}_lb_expanded_{team}_{season}"
    ctx_key = f"_{key_prefix}_filter_ctx"

    # Filter-ctx auto-clear: when the page's (team, season) changed
    # from the previous render, drop any sticky detail markers for
    # this position so navigation always lands the user in the new
    # leaderboard, not in a stale player.
    cur_ctx = (str(team), int(season) if season is not None else None)
    prev_ctx = st.session_state.get(ctx_key)
    if prev_ctx is not None and prev_ctx != cur_ctx:
        for _k in list(st.session_state.keys()):
            if _k.startswith(f"{key_prefix}_selected_player_") or \
               _k.startswith(f"{key_prefix}_lb_expanded_"):
                st.session_state.pop(_k, None)
    st.session_state[ctx_key] = cur_ctx

    # Detail-mode is active iff the marker names a player still in the
    # current leaderboard. Stale markers (player no longer present) get
    # silently dropped so the user falls back into browse mode.
    name_pool = (display_df[name_col].tolist()
                 if name_col in display_df.columns else [])
    marker = st.session_state.get(sel_key)
    in_detail = bool(marker and marker in name_pool)
    if marker and not in_detail:
        st.session_state.pop(sel_key, None)

    if in_detail:
        # Detail view — caller renders the full card.
        if st.button("← Back to leaderboard", type="primary",
                      key=f"{key_prefix}_back_to_lb_{team}_{season}"):
            st.session_state.pop(sel_key, None)
            st.rerun()
        return marker

    # Browse view — render the leaderboard ourselves and signal None.
    if top_banner_html:
        st.markdown(top_banner_html, unsafe_allow_html=True)
    if top_banner_warn:
        st.warning(top_banner_warn)

    # ── Click-to-sort header state ──
    # First click: sort that column descending (best→worst). Second
    # click on the same column: ascending (worst→best). Click a
    # different column: sort that one descending.
    sort_col_key = f"_lb_sort_col_{key_prefix}"
    sort_asc_key = f"_lb_sort_asc_{key_prefix}"

    def _on_hdr_click(col):
        if st.session_state.get(sort_col_key) == col:
            st.session_state[sort_asc_key] = not st.session_state.get(sort_asc_key, False)
        else:
            st.session_state[sort_col_key] = col
            st.session_state[sort_asc_key] = False  # default: best→worst

    import re as _re
    _NUM_RE = _re.compile(r'-?\d+\.?\d*')

    def _coerce_numeric(s):
        # Extract the first numeric substring so sort works on cells
        # like "5⭐", "✅ 2026", "🟡 2026", "+1.50", "12.5%", "23rd",
        # not just bare floats.
        if s is None: return float("nan")
        if isinstance(s, float) and pd.isna(s): return float("nan")
        s = str(s).strip()
        if s in ("—", "-", ""): return float("nan")
        m = _NUM_RE.search(s.replace(",", ""))
        if m:
            try: return float(m.group())
            except ValueError: return float("nan")
        return float("nan")

    # Apply current sort BEFORE capping rows so we sort the full pool.
    sort_col = st.session_state.get(sort_col_key)
    sort_asc = bool(st.session_state.get(sort_asc_key, False))
    sorted_df = display_df.copy()
    if sort_col and sort_col in sorted_df.columns:
        _key_series = sorted_df[sort_col].apply(_coerce_numeric)
        if _key_series.notna().any():
            sorted_df = (sorted_df.assign(_lb_sort_key=_key_series)
                                  .sort_values("_lb_sort_key", ascending=sort_asc, na_position="last")
                                  .drop(columns="_lb_sort_key"))
        else:
            sorted_df = sorted_df.sort_values(sort_col, ascending=sort_asc, na_position="last")

    # Cap rows; offer "Show all" if there's more than top_n.
    expanded = st.session_state.get(expand_key, False)
    visible_df = (sorted_df if expanded else sorted_df.head(top_n)).reset_index(drop=True)
    cols_list = list(visible_df.columns)

    def _w(c):
        if c == name_col: return 2.4
        if c in ("Rank", "#"): return 0.4
        if c in ("Your score", "Score", "Pctl", "Percentile"): return 0.8
        return 0.7

    weights = [_w(c) for c in cols_list]

    # Header row — each cell is a clickable text-button that toggles sort.
    hdrs = st.columns(weights)
    for i_h, c_h in enumerate(cols_list):
        active = (sort_col == c_h)
        arrow = " ▼" if active and not sort_asc else (" ▲" if active and sort_asc else "")
        hdrs[i_h].button(
            f"{c_h}{arrow}",
            key=f"{key_prefix}_hdr_btn_{team}_{season}_{c_h}",
            on_click=_on_hdr_click,
            args=(c_h,),
            type="tertiary",
            use_container_width=True,
            help="Click to sort: 1st click = best→worst, 2nd click = worst→best.",
        )

    for i_r, row in visible_df.iterrows():
        row_cols = st.columns(weights)
        for j, c in enumerate(cols_list):
            val = row[c]
            try:
                is_nan = isinstance(val, float) and pd.isna(val)
            except Exception:
                is_nan = False
            val_str = "—" if val is None or is_nan else str(val)
            if c == name_col:
                if row_cols[j].button(
                    val_str,
                    key=f"{key_prefix}_lb_btn_{team}_{season}_{i_r}_{val_str}",
                    type="tertiary",
                    use_container_width=True,
                ):
                    st.session_state[sel_key] = val_str
                    st.rerun()
            else:
                row_cols[j].markdown(val_str)

    if not expanded and len(sorted_df) > top_n:
        if st.button(
            f"Show all {len(sorted_df)} players →",
            key=f"{key_prefix}_show_all_{team}_{season}",
            use_container_width=True,
        ):
            st.session_state[expand_key] = True
            st.rerun()

    if leaderboard_caption:
        st.caption(leaderboard_caption)

    return None


# ──────────────────────────────────────────────────────────────────────
# Player detail card — unified Season picker + stylized stat bar
# ──────────────────────────────────────────────────────────────────────

def render_player_year_picker(*, career_df, default_season,
                               season_col="season_year",
                               team_col="recent_team",
                               key_prefix=""):
    """Render a single Season dropdown above the player detail card.

    Returns a dict so a single picker can drive everything below it
    (counting/stat bar, value/Z/percentile table, radar).

    Returned keys:
      year_choice    — int year or "All-career mean"
      view_row       — pd.Series with the year-scoped values (single
                       row for one season, numeric mean across career
                       in all-career mode)
      season_str     — display label e.g. "2025" or "All-career · 4 seasons"
      team_str       — team for that season; "" in all-career mode
      is_career_view — bool
      n_seasons      — number of player seasons aggregated
    """
    import streamlit as st

    if (career_df is None or len(career_df) == 0
            or season_col not in career_df.columns):
        return {"year_choice": None, "view_row": None,
                "season_str": "", "team_str": "",
                "is_career_view": False, "n_seasons": 0}

    year_options = sorted(
        set(int(s) for s in career_df[season_col].dropna().unique()),
        reverse=True,
    )
    year_options_full = (
        year_options + (["All-career mean"] if len(year_options) > 1 else [])
    )

    try:
        default_idx = year_options_full.index(int(default_season))
    except (ValueError, TypeError, KeyError):
        default_idx = 0

    year_choice = st.selectbox(
        "Season",
        options=year_options_full,
        index=default_idx,
        key=f"player_year_pick_{key_prefix}",
        format_func=lambda v: f"Season {v}" if isinstance(v, int) else v,
    )

    if year_choice == "All-career mean":
        view_row = career_df.select_dtypes(include="number").mean()
        return {
            "year_choice": year_choice,
            "view_row": view_row,
            "season_str": f"All-career · {len(career_df)} seasons",
            "team_str": "",
            "is_career_view": True,
            "n_seasons": len(career_df),
        }

    yr_rows = career_df[career_df[season_col] == year_choice]
    if len(yr_rows) == 1:
        view_row = yr_rows.iloc[0]
    elif len(yr_rows) > 1:
        view_row = yr_rows.select_dtypes(include="number").mean()
    else:
        view_row = career_df.iloc[0]
    team_str = (yr_rows.iloc[0].get(team_col, "")
                if len(yr_rows) >= 1 else "")
    return {
        "year_choice": year_choice,
        "view_row": view_row,
        "season_str": str(int(year_choice)),
        "team_str": str(team_str) if team_str else "",
        "is_career_view": False,
        "n_seasons": 1,
    }


def render_player_stat_bar(*, view_row, career_df, stat_specs, ctx_str,
                            sum_cols=None, measurable_cols=None,
                            is_career_view=False):
    """Render a stylized blue stat-tile bar under the player name.

    `stat_specs` is a list of (col_name, format_str, label) — typically
    up to 6 entries mixing counting stats and a couple modern/nerd
    metrics.

    In all-career mode:
      - cols in `sum_cols` are summed across the career
      - cols in `measurable_cols` (height/weight/etc.) stay as the mean
      - other cols use the view_row value (which is the numeric mean
        across the player's seasons)
    """
    import streamlit as st

    if not stat_specs or view_row is None:
        return

    sum_cols = sum_cols or set()
    measurable_cols = measurable_cols or set()

    tiles = []
    for col, fmt, label in stat_specs:
        if is_career_view and col in sum_cols:
            v = (career_df[col].sum()
                 if (career_df is not None and col in career_df.columns
                     and career_df[col].notna().any())
                 else None)
        else:
            v = view_row.get(col)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        try:
            val_str = fmt.format(v)
        except (ValueError, TypeError):
            continue
        tiles.append((label, val_str))

    if not tiles:
        return

    tiles_html = "".join(
        f"<div style='background:rgba(255,255,255,0.13);border-radius:6px;"
        f"padding:5px 10px;text-align:center;min-width:62px;flex:0 0 auto;'>"
        f"<div style='font-size:0.65rem;color:#a8c5e0;text-transform:uppercase;"
        f"letter-spacing:0.4px;margin-bottom:1px;'>{lbl}</div>"
        f"<div style='font-size:1.0rem;font-weight:bold;color:#fff;line-height:1.1;'>{val}</div>"
        f"</div>"
        for lbl, val in tiles
    )
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#0a3d62 0%,#1b5e8c 100%);"
        f"border-radius:10px;padding:10px 14px;margin:6px 0 12px 0;'>"
        f"<div style='color:#cfe2f3;font-size:0.72rem;margin-bottom:6px;"
        f"letter-spacing:0.5px;font-weight:600;'>📊 {ctx_str}</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:4px;'>{tiles_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# Trading-card player banner
# ============================================================
# ESPN serves NFL team logos at stable URLs keyed by lowercased
# abbreviation. A few nflverse abbrs don't match ESPN's slugs.
_ESPN_LOGO_OVERRIDES = {"WAS": "wsh", "WSH": "wsh", "LA": "lar"}


@st.cache_data
def _load_team_colors_cached(path_str: str):
    import json as _json
    from pathlib import Path
    p = Path(path_str)
    if not p.exists():
        return {}
    return _json.loads(p.read_text())


# Default fallback theme — Lions colors. Used when an abbreviation
# isn't in team_colors.json (college players, free agents, anomalies).
_FALLBACK_THEME = {
    "abbr": "",
    "name": "",
    "primary": "#0076B6",
    "secondary": "#B0B7BC",
    "logo": "",
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return f"#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def _adjust_lightness(hex_color: str, factor: float) -> str:
    """Multiply HSL lightness by `factor`, clamp to [0.15, 0.78] so the
    palette stays readable on a white background — colors never get so
    light they vanish or so dark they look black. factor>1 lightens,
    <1 darkens. Saturation is also nudged up slightly when lightening
    so that washed-out secondaries (silver, gold) keep some chroma."""
    import colorsys
    r, g, b = _hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    l = max(0.15, min(0.78, l * factor))
    if factor > 1.0 and s < 0.4:
        s = min(1.0, s + 0.15)  # boost chroma on lightened pale colors
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb_to_hex(r2 * 255, g2 * 255, b2 * 255)


def compute_rank_in_pool(value: float,
                          peer_values,
                          ascending: bool = False) -> tuple[int | None, int]:
    """Return (rank, total) where `rank` is this player's position in
    a peer pool sorted by their value. By default higher value =
    better (rank 1 = best). Set ascending=True for "lower is better"
    metrics like INT rate or sack rate.

    Returns (None, total) if `value` is null or peer pool is empty.
    Pool is silently de-duped by index — pass a Series indexed by
    player_id to keep it honest.
    """
    import pandas as pd
    if value is None or pd.isna(value):
        peer_values = pd.Series(peer_values).dropna()
        return None, len(peer_values)
    peer_values = pd.Series(peer_values).dropna()
    if peer_values.empty:
        return None, 0
    if ascending:
        # rank 1 = lowest value
        better = (peer_values < value).sum()
    else:
        # rank 1 = highest value
        better = (peer_values > value).sum()
    return int(better) + 1, len(peer_values)


def format_rank(rank: int | None, total: int) -> str:
    """Tidy 1-line rank display. '3rd of 47' / '#12 of 84' style."""
    if rank is None or total == 0:
        return "—"
    suffix = "th"
    if rank % 100 not in (11, 12, 13):
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")
    return f"{rank}{suffix} of {total}"


def render_player_comparison(*,
                               player_row,
                               player_name: str,
                               league_df,
                               name_col: str,
                               year_choice,
                               year_col: str = "season_year",
                               primary_score: float | None = None,
                               compute_comparison_score=None,
                               radar_builder=None,
                               benchmark=None,
                               benchmark_raw=None,
                               stat_labels: dict | None = None,
                               stat_methodology: dict | None = None,
                               key_prefix: str,
                               position_label: str = "player",
                               theme: dict | None = None) -> None:
    """Surface the player-vs-player comparison feature at the top of
    a player detail panel.

    Replaces the buried "🔍 Compare radar to another player" checkbox
    that lived at the bottom of every page. New layout:

        ┌ Compare with: [<- dropdown ->]   [×]   ┐
        │                                          │
        │ ┌ Score: Player A +1.23  vs  Player B +0.89  ·  +0.34 diff ┐
        │ │                                                            │
        │ │  ┌─ Radar A ─┐    ┌─ Radar B ─┐                           │
        │ │  │           │    │           │                           │
        │ │  └───────────┘    └───────────┘                           │
        │ └────────────────────────────────────────────────────────┘
        └──────────────────────────────────────────────────────────────┘

    Args:
        player_row: pandas Series for the current player (the page's
            view_row used for the radar).
        player_name: display name of the current player.
        league_df: full league dataframe (used to source comparison
            options + look up the comparison player's row).
        name_col: column name in league_df that has player display
            names ("player_display_name" / "full_name" / "player").
        year_choice: the page's year picker value (number or
            "All-career mean" sentinel) so the comparison uses the
            same season slice.
        year_col: column name for season year (default season_year).
        score_col: optional column name in league_df with the
            scored composite — drives the score-comparison headline.
            None to skip the score line.
        radar_builder: callable (row, stat_labels, stat_methodology,
            **opts) -> plotly.Figure or None. Each page passes its
            own build_radar_figure.
        benchmark / benchmark_raw: passed through to radar_builder.
        stat_labels / stat_methodology: passed through to radar_builder.
        key_prefix: unique key prefix for Streamlit widgets on this page.
        position_label: short label for the comparison dropdown
            placeholder ("running back", "wide receiver", etc.).
        theme: team theme dict for accent coloring.
    """
    import streamlit as st

    accent = (theme or {}).get("primary", "#0076B6")

    # ── Comparison picker (player — season aware) ────────────
    # Build a list of (player, season) options so users can pick
    # ANY season of ANY player — including the same player at a
    # different season ("rookie Gibbs vs sophomore Gibbs"). Plus an
    # "all career" entry per player at the bottom.
    name_series = league_df.get(name_col, pd.Series())
    if name_series is None or name_series.empty:
        return
    options_pairs: list[tuple[str, object, str]] = []   # (label, season_value, player_name)

    distinct_players = sorted(set(
        str(n) for n in name_series.dropna().unique() if str(n).strip()
    ))
    if not distinct_players:
        return

    # One entry per player (career-aggregate) — the picker stays scannable.
    # Per-season comparison is handled by setting the page's own season
    # filter; comparing a 2024 view to a 2024 view of another QB is the
    # same thing as picking "Mahomes" here.
    if year_col in league_df.columns:
        for p in distinct_players:
            p_rows = league_df[name_series == p]
            if p_rows.empty:
                continue
            n_seasons = p_rows[year_col].dropna().nunique()
            label = (f"{p} — career ({n_seasons} season{'s' if n_seasons != 1 else ''})"
                     if n_seasons > 1 else f"{p} — {int(p_rows[year_col].dropna().iloc[0])}")
            if p == player_name:
                # Don't offer the same player the user is already viewing.
                continue
            options_pairs.append((label, "All-career mean", p))
    else:
        for p in distinct_players:
            if p == player_name:
                continue
            options_pairs.append((p, "All-career mean", p))

    if not options_pairs:
        return

    options_labels = ["— off —"] + [opt[0] for opt in options_pairs]

    cmp_col, _ = st.columns([1, 2])
    with cmp_col:
        cmp_pick_label = st.selectbox(
            f"⚔️  Compare with another {position_label}",
            options=options_labels,
            index=0,
            key=f"{key_prefix}_cmp_pick",
            help="Pick any player at any season — including the same "
                 "player at a different season ('rookie vs sophomore'). "
                 "Both radars render side-by-side with a score delta.",
        )

    if cmp_pick_label == "— off —":
        return

    # Look up which (season, player) the chosen label refers to.
    chosen = next(((s, n) for (lbl, s, n) in options_pairs
                    if lbl == cmp_pick_label), None)
    if chosen is None:
        return
    cmp_season, cmp_choice = chosen

    # ── Resolve the comparison player's row ────────────────────
    cmp_career = league_df[league_df[name_col] == cmp_choice]
    if cmp_career.empty:
        st.caption(f"_No data for {cmp_choice}._")
        return

    if cmp_season == "All-career mean":
        cmp_radar_row = cmp_career.select_dtypes(include="number").mean()
        cmp_year_label = f"All-career · {len(cmp_career)} seasons"
    else:
        cmp_yr = cmp_career[cmp_career.get(year_col, pd.Series()) == cmp_season]
        if len(cmp_yr) == 1:
            cmp_radar_row = cmp_yr.iloc[0]
            cmp_year_label = f"{int(cmp_season)}"
        elif len(cmp_yr) > 1:
            cmp_radar_row = cmp_yr.select_dtypes(include="number").mean()
            cmp_year_label = f"{int(cmp_season)} · weighted"
        else:
            cmp_radar_row = cmp_career.iloc[0]
            cmp_year_label = "(closest available)"
    cmp_score_row = cmp_radar_row

    # ── Score-comparison headline ─────────────────────────────
    # Caller provides primary_score directly (already computed against
    # their slider preset) and a callable to compute comparison_score
    # using the same formula on the comparison row.
    cmp_score = float("nan")
    if compute_comparison_score is not None:
        try:
            cmp_score = float(compute_comparison_score(cmp_score_row))
        except Exception:
            cmp_score = float("nan")
    if (primary_score is not None and not pd.isna(primary_score)
            and not pd.isna(cmp_score)):
        diff = float(primary_score) - cmp_score
        diff_color = "#1a8c3d" if diff >= 0 else "#b3261e"
        st.markdown(
                f"<div style='background:#fff;border:1px solid #e6e9ee;"
                f"border-left:4px solid {accent};border-radius:6px;"
                f"padding:14px 18px;margin:8px 0 14px 0;'>"
                f"<div style='display:flex;justify-content:space-around;"
                f"align-items:center;gap:20px;'>"
                f"<div style='text-align:center;'>"
                f"<div style='font-size:0.65rem;color:#5b6b7e;letter-spacing:1.4px;"
                f"text-transform:uppercase;font-weight:700;'>{player_name}</div>"
                f"<div style='font-size:1.8rem;font-weight:900;color:#0a3d62;'>"
                f"{primary_score:+.2f}</div></div>"
                f"<div style='font-size:1.2rem;color:#5b6b7e;font-weight:600;'>vs</div>"
                f"<div style='text-align:center;'>"
                f"<div style='font-size:0.65rem;color:#5b6b7e;letter-spacing:1.4px;"
                f"text-transform:uppercase;font-weight:700;'>{cmp_choice}</div>"
                f"<div style='font-size:1.8rem;font-weight:900;color:#0a3d62;'>"
                f"{cmp_score:+.2f}</div></div>"
                f"<div style='font-size:1.2rem;color:#5b6b7e;'>·</div>"
                f"<div style='text-align:center;'>"
                f"<div style='font-size:0.65rem;color:#5b6b7e;letter-spacing:1.4px;"
                f"text-transform:uppercase;font-weight:700;'>diff</div>"
                f"<div style='font-size:1.5rem;font-weight:900;color:{diff_color};'>"
                f"{diff:+.2f}</div></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # ── Side-by-side radars ───────────────────────────────────
    if radar_builder is None:
        st.caption(f"_Comparison set to {cmp_choice} — radar build callable not provided._")
        return

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(f"**{player_name}**")
        try:
            fig_a = radar_builder(player_row, stat_labels or {}, stat_methodology or {},
                                   benchmark=benchmark, benchmark_raw=benchmark_raw)
        except TypeError:
            fig_a = radar_builder(player_row, stat_labels or {}, stat_methodology or {})
        if fig_a:
            st.plotly_chart(fig_a, use_container_width=True,
                              key=f"{key_prefix}_cmp_radar_a")
        else:
            st.caption("_No radar data for this player._")
    with c_right:
        st.markdown(f"**{cmp_choice}** — _{cmp_year_label}_")
        try:
            fig_b = radar_builder(cmp_radar_row, stat_labels or {}, stat_methodology or {},
                                   benchmark=benchmark, benchmark_raw=benchmark_raw)
        except TypeError:
            fig_b = radar_builder(cmp_radar_row, stat_labels or {}, stat_methodology or {})
        if fig_b:
            st.plotly_chart(fig_b, use_container_width=True,
                              key=f"{key_prefix}_cmp_radar_b")
        else:
            st.caption(f"_No radar data for {cmp_choice}._")


def heatmap_color(value: float, lo: float = 0.0, hi: float = 1.0,
                  reverse: bool = False) -> str:
    """Map a numeric value to a binary-diverging red↔green heatmap.

    Below the midpoint = shades of red (vivid at the floor, pale near
    the middle). Above the midpoint = shades of green (pale near the
    middle, vivid at the ceiling). **No yellow zone** — the signal is
    binary "good or bad," intensity says "how far."

    Used anywhere a number means good or bad — combine percentile
    bars, gap-chart EPA bars, ordinal cohort buckets. Colors are
    HSL-derived at controlled saturation/lightness so they stay
    vivid yet readable on a white background.

    `lo` and `hi` define the value range; values outside the range
    clamp to the extreme color. `reverse=True` flips the meaning
    (use when "higher is worse" — e.g., heavy-box-rate where
    Heavy Box is the unfavorable cohort for the player).
    """
    if hi == lo:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    if reverse:
        t = 1.0 - t

    # Direct RGB interpolation between fixed red and green endpoints —
    # NO hue rotation through orange/yellow. Distance from midpoint
    # controls intensity (0 = very pale, 1 = vivid).
    #   Pale red  (250, 200, 200)  →  Vivid red  (200,  20,  28)
    #   Pale green (200, 245, 205) →  Vivid green ( 20, 180,  40)
    distance = abs(t - 0.5) * 2.0
    if t >= 0.5:
        r = 200.0 - 180.0 * distance
        g = 245.0 -  65.0 * distance
        b = 205.0 - 165.0 * distance
    else:
        r = 250.0 -  50.0 * distance
        g = 200.0 - 180.0 * distance
        b = 200.0 - 172.0 * distance
    return f"#{int(round(r)):02x}{int(round(g)):02x}{int(round(b)):02x}"


def team_palette(theme: dict, n: int) -> list[str]:
    """Generate `n` distinct chart colors derived from the team's
    primary + secondary. Used for any chart with multiple categories
    (cohort splits, multi-line, multi-category bar) so every visual
    on a player's page stays in the team's color family.

    Pattern for n ≥ 3: alternate primary and secondary, walking through
    increasing lightness contrast. Each team produces a unique-looking
    palette that's still tribally identifiable.
    """
    primary = (theme or {}).get("primary", "#0076B6")
    secondary = (theme or {}).get("secondary", "#B0B7BC")
    if n <= 0:
        return []
    if n == 1:
        return [primary]
    if n == 2:
        return [primary, secondary]
    # Lightness factors: base, light, dark, very-light, very-dark.
    factors = [1.0, 1.4, 0.65, 1.7, 0.4]
    bases = [primary, secondary]
    palette: list[str] = []
    for i in range(n):
        layer = i // 2
        base = bases[i % 2]
        factor = factors[layer % len(factors)]
        palette.append(_adjust_lightness(base, factor))
    return palette


def team_theme(team_abbr: str | None) -> dict:
    """Return the visual theme for an NFL team — the single source of
    truth for color and logo on every player surface (panels, cards,
    trading-card export). One call returns:

        {abbr, name, primary, secondary, logo}

    Falls back to Lions blue when the abbr isn't recognized. The logo
    URL comes from data/team_colors.json (sourced from nflverse, served
    by ESPN's CDN). When the JSON entry has no logo (legacy / college
    case), falls back to building the ESPN URL from the abbr.
    """
    from pathlib import Path
    if not team_abbr:
        return dict(_FALLBACK_THEME)
    colors_path = Path(__file__).resolve().parent / "data" / "team_colors.json"
    tc_all = _load_team_colors_cached(str(colors_path))
    info = tc_all.get(team_abbr)
    if not info:
        return dict(_FALLBACK_THEME, abbr=team_abbr)
    logo = info.get("logo")
    if not logo:
        espn_abbr = _ESPN_LOGO_OVERRIDES.get(team_abbr, team_abbr.lower())
        logo = f"https://a.espncdn.com/i/teamlogos/nfl/500/{espn_abbr}.png"
    return {
        "abbr": team_abbr,
        "name": info.get("name", team_abbr),
        "primary": info.get("primary", _FALLBACK_THEME["primary"]),
        "secondary": info.get("secondary", _FALLBACK_THEME["secondary"]),
        "logo": logo,
    }


def render_player_card(*, player_name, position_label, season_str,
                       score, stat_specs, view_row,
                       team_abbr=None,
                       team_label=None, primary_color=None,
                       secondary_color=None, logo_url=None,
                       player_career=None, is_career_view=False,
                       sum_cols=None):
    """Render a Topps/MUT-style banner with a team-color gradient,
    team logo, large name, score/percentile banner, and stat tiles.

    Resolution order for colors / label / logo:
      1. Explicit `primary_color` / `secondary_color` / `team_label` /
         `logo_url` if given (used for college, where we don't have a
         per-school colors file).
      2. Otherwise, look up `team_abbr` in data/team_colors.json and
         build the ESPN NFL logo URL.
      3. Otherwise, fall back to Lions blue.

    `stat_specs` is a list of (col_name, format_str, label) — typically
    up to 6 entries. In all-career mode, cols in `sum_cols` are summed
    across the career; everything else uses view_row's value.
    """
    import streamlit as st
    from pathlib import Path
    from scipy.stats import norm

    colors_path = Path(__file__).resolve().parent / "data" / "team_colors.json"
    tc_all = _load_team_colors_cached(str(colors_path))
    team_info = tc_all.get(team_abbr, {}) if team_abbr else {}

    primary = primary_color or team_info.get("primary", "#0076B6")
    secondary = secondary_color or team_info.get("secondary", "#B0B7BC")
    team_name = team_label or team_info.get("name", team_abbr or "")

    if logo_url is None:
        # Auto-generate ESPN NFL logo URL only when we recognize the
        # abbr as an NFL team. College/unknowns get no logo.
        if team_abbr and team_abbr in tc_all:
            espn_abbr = _ESPN_LOGO_OVERRIDES.get(team_abbr, team_abbr.lower())
            logo_url = f"https://a.espncdn.com/i/teamlogos/nfl/500/{espn_abbr}.png"
        else:
            logo_url = ""

    if score is None or (isinstance(score, float) and pd.isna(score)):
        score_str = "—"
        pct_str = "—"
    else:
        sign = "+" if score >= 0 else ""
        score_str = f"{sign}{score:.2f}"
        pct_val = float(norm.cdf(score) * 100)
        pct_str = f"{int(pct_val)}th"

    sum_cols = sum_cols or set()
    tile_blocks = []
    for col, fmt, label in stat_specs:
        if (is_career_view and col in sum_cols
                and player_career is not None
                and col in player_career.columns
                and player_career[col].notna().any()):
            v = player_career[col].sum()
        else:
            v = view_row.get(col) if view_row is not None else None
        if v is None or (isinstance(v, float) and pd.isna(v)):
            v_str = "—"
        else:
            try:
                v_str = fmt.format(v)
            except (ValueError, TypeError):
                v_str = str(v)
        tile_blocks.append(
            f"<div style='flex:1;min-width:78px;background:rgba(255,255,255,0.18);"
            f"border-radius:10px;padding:8px 4px;text-align:center;"
            f"backdrop-filter:blur(4px);border:1px solid rgba(255,255,255,0.1);'>"
            f"<div style='font-size:0.62rem;color:{secondary};letter-spacing:1.2px;"
            f"font-weight:700;text-transform:uppercase;'>{label}</div>"
            f"<div style='font-size:1.45rem;font-weight:900;color:white;line-height:1.0;"
            f"margin-top:3px;text-shadow:1px 1px 3px rgba(0,0,0,0.4);'>{v_str}</div>"
            f"</div>"
        )
    tiles_html = "".join(tile_blocks)

    parts = (player_name or "").split()
    first = parts[0] if parts else (player_name or "")
    last = " ".join(parts[1:]) if len(parts) > 1 else ""

    logo_html = (
        f"<img src='{logo_url}' alt='{team_abbr} logo' "
        f"style='height:88px;margin-top:0;opacity:0.95;'/>"
        if logo_url else ""
    )
    # Name's negative margin floats it UP into the empty space left of
    # the 88px-tall logo. Without a logo, that empty space doesn't exist
    # — the negative margin would clip the name behind the top row (the
    # card has overflow:hidden). Use a normal top margin in that case.
    _name_margin_top = "-72px" if logo_url else "10px"

    st.markdown(
        f"<div style='background:linear-gradient(135deg, {primary} 0%, "
        f"{primary}cc 45%, #0a1929 100%);"
        f"border-radius:18px;padding:0 26px 18px 26px;margin:6px 0 18px 0;"
        f"color:white;box-shadow:0 10px 28px rgba(0,0,0,0.30),"
        f"inset 0 1px 0 rgba(255,255,255,0.15);position:relative;overflow:hidden;"
        f"border-top:5px solid {secondary};'>"
        # Top row: position badge (left) + team text + logo stacked (right)
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
        f"margin-bottom:0;position:relative;z-index:1;'>"
        f"<div style='background:rgba(255,255,255,0.22);border-radius:6px;"
        f"padding:4px 12px;font-size:0.82rem;font-weight:800;letter-spacing:1.5px;'>"
        f"{position_label}</div>"
        f"<div style='display:flex;flex-direction:column;align-items:center;'>"
        f"<div style='font-size:0.7rem;letter-spacing:2px;color:{secondary};"
        f"font-weight:700;text-transform:uppercase;'>"
        f"{team_name} · {season_str}</div>"
        f"{logo_html}"
        f"</div>"
        f"</div>"
        # Player name — first (silver) + last (white, huge)
        f"<div style='font-size:1.35rem;font-weight:700;line-height:1.0;"
        f"color:{secondary};letter-spacing:1.5px;margin-top:{_name_margin_top};"
        f"margin-bottom:0;text-transform:uppercase;position:relative;z-index:1;'>{first}</div>"
        f"<div style='font-size:2.6rem;font-weight:900;line-height:1.0;"
        f"text-shadow:2px 2px 6px rgba(0,0,0,0.45);letter-spacing:-1px;"
        f"margin-bottom:8px;text-transform:uppercase;position:relative;z-index:1;'>{last}</div>"
        # Score + percentile banner
        f"<div style='background:rgba(0,0,0,0.32);border-left:5px solid {secondary};"
        f"padding:9px 14px;margin:10px 0 14px 0;border-radius:6px;"
        f"display:flex;justify-content:space-between;align-items:center;"
        f"position:relative;z-index:1;'>"
        f"<div>"
        f"<div style='font-size:0.62rem;color:{secondary};letter-spacing:1.6px;"
        f"font-weight:700;text-transform:uppercase;'>Your Score</div>"
        f"<div style='font-size:1.75rem;font-weight:900;line-height:1.1;'>{score_str}</div>"
        f"</div>"
        f"<div style='text-align:right;'>"
        f"<div style='font-size:0.62rem;color:{secondary};letter-spacing:1.6px;"
        f"font-weight:700;text-transform:uppercase;'>Percentile</div>"
        f"<div style='font-size:1.75rem;font-weight:900;line-height:1.1;'>{pct_str}</div>"
        f"</div>"
        f"</div>"
        # Stat tiles
        f"<div style='display:flex;flex-wrap:wrap;gap:6px;"
        f"position:relative;z-index:1;'>{tiles_html}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# Combine workout chart
# ============================================================
# Lower-is-better metrics — z-score gets flipped so positive always means
# "above average" on the rendered bar.
_COMBINE_LOWER_BETTER = {"forty", "cone", "shuttle"}
_COMBINE_LABELS = {
    "forty": "40-yard",
    "bench": "Bench (reps)",
    "vertical": "Vertical (in)",
    "broad_jump": "Broad jump (in)",
    "cone": "3-cone",
    "shuttle": "Shuttle",
}
_COMBINE_RAW_FMT = {
    "forty": "{:.2f}s",
    "bench": "{:.0f}",
    "vertical": "{:.1f}\"",
    "broad_jump": "{:.0f}\"",
    "cone": "{:.2f}s",
    "shuttle": "{:.2f}s",
}


@st.cache_data
def _load_workouts_parquet(path_str: str):
    from pathlib import Path
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data
def _position_workout_means(path_str: str, positions: tuple):
    """Mean/std/n per measurable for a given position pool (all-time).
    `positions` is a tuple so the result is hashable for caching.
    Drops 0.0 sentinels on timed events (they mean 'didn't run')."""
    df = _load_workouts_parquet(path_str)
    if df.empty or "pos" not in df.columns:
        return {}
    pool = df[df["pos"].isin(positions)]
    out = {}
    for c in ("forty", "bench", "vertical", "broad_jump", "cone", "shuttle"):
        if c not in pool.columns:
            continue
        s = pool[c].dropna()
        if c in _COMBINE_LOWER_BETTER:
            s = s[s > 0]
        if len(s) < 10:
            continue
        out[c] = (float(s.mean()), float(s.std()), int(len(s)))
    return out


def render_combine_chart(*, player_name, position, workouts_path, key,
                         pool_positions=None,
                         above_color=None, below_color=None):
    """Render a horizontal percentile bar chart of the player's combine
    workout measurables, vs. the all-time pool for `position`.

    Each bar is colored by its percentile via a smooth red→yellow→green
    heatmap (vivid green at 90th+, red at 10th-) — color carries data
    signal, not team identity, because this is an *information* chart.

    `position` is the display label (e.g., 'WR', 'RB', 'OL', 'LB', 'S').
    `pool_positions`, if given, is the list of pos codes used to build
    the all-time pool (e.g., ['OT','OG','C'] for OL, ['OLB','ILB'] for
    LB). Defaults to [position].

    Z-scores for lower-is-better metrics (40, 3-cone, shuttle) are flipped
    so positive always means 'above the position mean'. Skips silently if
    the player has no usable combine data.

    `above_color` / `below_color` are deprecated and ignored — kept on
    the signature for backwards compatibility with existing callers.
    """
    import streamlit as st
    import plotly.graph_objects as go
    from scipy.stats import norm

    workouts = _load_workouts_parquet(str(workouts_path))
    if workouts.empty or not player_name:
        return

    parts = player_name.split()
    if not parts:
        return
    last, first = parts[-1], parts[0]
    matches = workouts[
        workouts["player_name"].str.contains(last, na=False, case=False)
        & workouts["player_name"].str.contains(first, na=False, case=False)
    ]
    if len(matches) == 0:
        return
    player_combine = matches.iloc[0]

    pool_pos = tuple(pool_positions) if pool_positions else (position,)
    means = _position_workout_means(str(workouts_path), pool_pos)
    bars = []
    for col, lbl in _COMBINE_LABELS.items():
        v = player_combine.get(col)
        if pd.isna(v):
            continue
        if col in _COMBINE_LOWER_BETTER and (v is None or v <= 0):
            continue
        if col not in means:
            continue
        mu, sigma, n = means[col]
        if sigma == 0:
            continue
        z = (v - mu) / sigma
        if col in _COMBINE_LOWER_BETTER:
            z = -z
        bars.append((lbl, z, v, mu, n, col))

    if not bars:
        return

    st.markdown(f"**🏋️ Combine workout — vs. all-time {position} pool**")
    fig = go.Figure()
    # Numeric y so we can draw per-row dashed rectangles whose top/bottom
    # line up with each bar — categorical y wouldn't allow numeric offsets.
    y_idx = list(range(len(bars)))
    labels, pcts, hover, colors, texts = [], [], [], [], []
    for lbl, z, raw, mu, n, col in bars:
        pct = float(norm.cdf(z) * 100)
        labels.append(lbl)
        pcts.append(pct)
        raw_str = _COMBINE_RAW_FMT[col].format(raw)
        mean_str = _COMBINE_RAW_FMT[col].format(mu)
        hover.append(
            f"<b>{lbl}</b><br>"
            f"Player: {raw_str}<br>"
            f"{position} mean: {mean_str} (n={n})<br>"
            f"Percentile: {int(pct)}th"
        )
        # Bar end label: player's value, then the historical mean in parens.
        texts.append(f"{raw_str} ({mean_str})")
        # Smooth percentile heatmap — red at 10th-, green at 90th+.
        colors.append(heatmap_color(pct, lo=0.0, hi=100.0))

    fig.add_trace(go.Bar(
        x=pcts, y=y_idx, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.3)", width=0.5)),
        text=texts, textposition="outside",
        textfont=dict(size=11, color="#1a1a2e"),
        hovertext=hover, hoverinfo="text",
        cliponaxis=False,
        showlegend=False,
    ))

    # Per-row dashed rectangle from x=0 to the historical mean (50th
    # percentile by definition). Top/bottom match each bar's height
    # (bargap=0.35 → bar half-height ≈ 0.325). Drawn AFTER the bar
    # trace with a transparent fill, so the dashed border is always
    # visible — even when the player's bar extends past the mean.
    bar_half = 0.325
    for i in y_idx:
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=0, x1=50,
            y0=i - bar_half, y1=i + bar_half,
            line=dict(color="#444", width=1.6, dash="dash"),
            fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    fig.update_layout(
        xaxis=dict(title=f"Percentile vs. all-time {position}s →",
                   zeroline=True, zerolinecolor="#bbb", zerolinewidth=1,
                   gridcolor="#eee",
                   range=[0, 130],
                   tickvals=[0, 25, 50, 75, 100],
                   ticktext=["0", "25th", "50th", "75th", "100th"]),
        yaxis=dict(autorange="reversed",
                   tickvals=y_idx, ticktext=labels),
        height=260, margin=dict(l=10, r=60, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.caption(
        f"_Bars show this player's percentile vs. the all-time {position} "
        f"combine pool — longer = better (lower-is-better metrics like the "
        f"40 are inverted). Dashed box = 0 → {position} historical average "
        f"(the 50th percentile). Bar past the box = above average._"
    )

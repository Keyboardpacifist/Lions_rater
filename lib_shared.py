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

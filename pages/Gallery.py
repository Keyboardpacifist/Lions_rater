"""
Community Gallery — public feed of saved trading cards.

Each row shows a regenerated 4:5 trading card built from the saved
(player, season, preset) tuple. Filter by team / position, sort by
recent / upvotes / score, click through to load the preset on the
player's own page.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from lib_shared import (
    inject_css,
    list_cards,
    upvote_card,
    team_theme,
)

st.set_page_config(
    page_title="Lions Rater — Community Gallery",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# ── Position → data file + page slug for click-through ────────────
_POSITION_CONFIG = {
    "qb":     {"label": "QB",     "data": "league_qb_all_seasons.parquet",
                "page": "QB"},
    "rb":     {"label": "RB",     "data": "master_lions_rbs_with_z.parquet",
                "page": "Running_backs"},
    "wr":     {"label": "WR",     "data": "master_lions_with_z.parquet",
                "page": "WR"},
    "te":     {"label": "TE",     "data": "league_te_all_seasons.parquet",
                "page": "TE"},
    "ol":     {"label": "OL",     "data": "league_ol_all_seasons.parquet",
                "page": "Offensive_Line"},
    "de":     {"label": "EDGE",   "data": "league_de_all_seasons.parquet",
                "page": "DE"},
    "dt":     {"label": "DT",     "data": "league_dt_all_seasons.parquet",
                "page": "DT"},
    "lb":     {"label": "LB",     "data": "league_lb_all_seasons.parquet",
                "page": "LB"},
    "cb":     {"label": "CB",     "data": "league_cb_all_seasons.parquet",
                "page": "CB"},
    "safety": {"label": "Safety", "data": "league_s_all_seasons.parquet",
                "page": "Safety."},
    "kicker": {"label": "K",      "data": "league_k_all_seasons.parquet",
                "page": "Kicker"},
    "punter": {"label": "P",      "data": "league_p_all_seasons.parquet",
                "page": "Punter"},
}

# Common NFL teams — keep aligned with team_selector.NFL_TEAMS
_NFL_TEAMS = sorted([
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN",
    "DET","GB","HOU","IND","JAX","KC","LA","LAC","LV","MIA",
    "MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS",
])


@st.cache_data
def _load_position_data(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pl.read_parquet(path).to_pandas()
    except Exception:
        return pd.DataFrame()


def _render_card_thumb(card: dict) -> None:
    """Render one gallery card as a compact preview tile."""
    pos = card.get("position_group", "")
    cfg = _POSITION_CONFIG.get(pos)
    theme = team_theme(card.get("team_abbr"))
    primary = theme.get("primary", "#1F2A44")
    secondary = theme.get("secondary", "#0B1730")

    # Score formatting
    score = card.get("score")
    if score is not None:
        sign = "+" if score >= 0 else ""
        score_str = f"{sign}{score:.2f}"
    else:
        score_str = "—"

    season_label = card.get("season_label") or "—"
    name = card.get("player_name", "Unknown")
    team_abbr = card.get("team_abbr") or ""
    pos_label = (cfg or {}).get("label", pos.upper())
    caption = card.get("caption") or ""
    author = card.get("author") or "Anonymous"
    upvotes = card.get("upvotes", 0) or 0

    # Stylized HTML card preview — same gradient frame language as the
    # in-app trading-card hero, scaled to a thumbnail.
    st.markdown(
        f"""
<div style="
    background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 4px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    color: white;
    height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
">
    <div>
        <div style="font-size: 11px; opacity: 0.7; letter-spacing: 1px;">
            {pos_label} · {team_abbr} · {season_label}
        </div>
        <div style="font-size: 22px; font-weight: 800; margin-top: 6px;
                     line-height: 1.1;">
            {name}
        </div>
    </div>
    <div style="font-size: 38px; font-weight: 900; letter-spacing: -1px;">
        {score_str}
    </div>
    <div style="font-size: 12px; opacity: 0.85; font-style: italic;
                 line-height: 1.3; min-height: 30px;">
        {caption}
    </div>
    <div style="display: flex; justify-content: space-between;
                 align-items: center; font-size: 11px; opacity: 0.75;">
        <span>by {author}</span>
        <span>▲ {upvotes}</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Action row
    c1, c2 = st.columns(2)
    with c1:
        if cfg and st.button("Open on player page",
                              key=f"open_{card.get('id')}",
                              use_container_width=True,
                              help="Loads this preset on the player's page."):
            # Stash the loaded preset in session state, then nav.
            algo_payload = {
                "name": caption or f"{author}'s preset",
                "bundle_weights": (
                    card["bundle_weights"]
                    if isinstance(card.get("bundle_weights"), dict)
                    else json.loads(card.get("bundle_weights") or "{}")
                ),
                "author": author,
            }
            st.session_state[f"{pos}_loaded_algo"] = algo_payload
            st.switch_page(f"pages/{cfg['page']}.py")
    with c2:
        if st.button(f"▲ Upvote ({upvotes})",
                      key=f"up_{card.get('id')}",
                      use_container_width=True,
                      help="Show love for this card."):
            ok = upvote_card(card["id"], upvotes)
            if ok:
                st.toast("Upvoted!")
                st.rerun()


# ── Page header ──────────────────────────────────────────────────
st.title("🃏 Community Gallery")
st.caption(
    "Every card is a saved (player × season × slider preset) the community "
    "built. Sort, filter, and click through to load the preset on the "
    "player's page."
)

# ── Filters ──────────────────────────────────────────────────────
f1, f2, f3 = st.columns([2, 2, 2])
with f1:
    pos_pretty = (
        ["All positions"]
        + [f"{cfg['label']} ({k})" for k, cfg in _POSITION_CONFIG.items()]
    )
    pos_pick = st.selectbox("Position", options=pos_pretty, index=0)
    if pos_pick == "All positions":
        pos_filter = None
    else:
        pos_filter = pos_pick.split("(")[1].rstrip(")")
with f2:
    team_pick = st.selectbox(
        "Team",
        options=["All teams"] + _NFL_TEAMS,
        index=0,
    )
    team_filter = None if team_pick == "All teams" else team_pick
with f3:
    sort_pick = st.selectbox(
        "Sort by",
        options=["Most recent", "Most upvoted", "Highest score"],
        index=0,
    )
    sort_axis = {
        "Most recent": "created_at",
        "Most upvoted": "upvotes",
        "Highest score": "score",
    }[sort_pick]

# ── Fetch + render ───────────────────────────────────────────────
with st.spinner("Loading cards…"):
    cards = list_cards(
        position_group=pos_filter,
        team_abbr=team_filter,
        order_by=sort_axis,
        limit=60,
    )

if not cards:
    st.info(
        "No cards yet for this filter — be the first! Build a preset on "
        "any player page and click **📌 Save to gallery** under the "
        "trading card."
    )
else:
    st.caption(f"Showing {len(cards)} card(s).")

    # 3-column grid
    n_per_row = 3
    rows = [cards[i:i + n_per_row] for i in range(0, len(cards), n_per_row)]
    for row in rows:
        cols = st.columns(n_per_row)
        for col, card in zip(cols, row):
            with col:
                _render_card_thumb(card)
        # Spacer between rows
        st.markdown(
            '<div style="height: 12px;"></div>',
            unsafe_allow_html=True,
        )

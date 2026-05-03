"""Top-of-page navigation — replaces the Streamlit sidebar nav.

Renders a row of top-level tabs (Gambling / Fantasy / Draft / Scheme)
and a position dropdown that jumps to the corresponding position page.
Designed to be called from any page via:

    from lib_top_nav import render_top_nav
    render_top_nav(active="landing")  # or "gambling", "fantasy", etc.

The Streamlit sidebar is hidden globally via lib_shared.SHARED_CSS, so
this top nav is the user's sole navigation surface. All position-page
sidebar widgets (sliders, etc.) still instantiate behind the hidden
sidebar — their session_state persists with default values until we
resurface them in main-body expanders.

Why a shared lib rather than hand-rolling per page: 20+ pages need
identical nav. Centralizing here means a single edit propagates.
"""
from __future__ import annotations

import streamlit as st


# Top-tab definitions — (label, icon, target_page, key)
TOP_TABS = [
    ("Gambling", "🎰", "pages/Gambling.py", "gambling"),
    ("Fantasy",  "🏆", "pages/Fantasy.py",  "fantasy"),
    ("Draft",    "📋", "pages/Draft.py",    "draft"),
    ("Scheme",   "🧠", "pages/Schemes.py",  "scheme"),
]

# Position dropdown — (display_label, page_path)
# Order is: skill → front 7 → secondary → specialists → front office.
# Front-office roles (Coaches/OC/DC/GM) live in this dropdown per
# Brett's instruction; we may break them out later.
POSITION_OPTIONS = [
    ("— Jump to a position —", None),
    ("QB",                     "pages/QB.py"),
    ("RB",                     "pages/2_Running_backs.py"),
    ("WR",                     "pages/WR.py"),
    ("TE",                     "pages/TE.py"),
    ("OL",                     "pages/3_Offensive_Line.py"),
    ("DE",                     "pages/DE.py"),
    ("DT",                     "pages/DT.py"),
    ("LB",                     "pages/LB.py"),
    ("CB",                     "pages/CB.py"),
    ("S (Safety)",             "pages/Safety..py"),
    ("K (Kicker)",             "pages/Kicker.py"),
    ("P (Punter)",             "pages/Punter.py"),
    ("Coaches",                "pages/4 coaches.py"),
    ("OC (Off. Coordinator)",  "pages/OC.py"),
    ("DC (Def. Coordinator)",  "pages/DC_coord.py"),
    ("GM",                     "pages/GM.py"),
]


def render_top_tabs(*, active: str | None = None) -> None:
    """Render just the 4 top-tab buttons (Gambling/Fantasy/Draft/Scheme)
    in a single row. Each button switch_page's to its target.

    `active` — key of the currently-active top tab (or "landing").
    """
    cols = st.columns(len(TOP_TABS))
    for col, (label, icon, page_path, key) in zip(cols, TOP_TABS):
        with col:
            is_active = (active == key)
            btn_label = f"{icon} **{label}**" if is_active else f"{icon} {label}"
            if st.button(
                btn_label,
                key=f"topnav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.switch_page(page_path)


def render_position_dropdown(*, key: str = "topnav_position_pick",
                                label: str = "Jump to a position",
                                label_visibility: str = "collapsed",
                                placeholder_idx: int = 0) -> None:
    """Render JUST the position dropdown — the 16-entry selectbox that
    jumps to the chosen position page. Designed to be dropped inside
    an existing column. No labels by default so it nests cleanly.

    Uses a nonce-suffixed widget key so each successful pick gets a
    fresh widget on the next render. Streamlit forbids setting
    session_state[widget_key] AFTER the widget instantiates — the
    nonce trick sidesteps that by giving the next render a brand-new
    key that defaults back to the placeholder.
    """
    nonce_key = f"{key}_nonce"
    if nonce_key not in st.session_state:
        st.session_state[nonce_key] = 0
    full_key = f"{key}_{st.session_state[nonce_key]}"

    labels = [opt[0] for opt in POSITION_OPTIONS]
    pick_label = st.selectbox(
        label,
        options=labels,
        index=placeholder_idx,
        key=full_key,
        label_visibility=label_visibility,
    )
    page_path = dict(POSITION_OPTIONS).get(pick_label)
    if page_path:
        # Bump nonce so next render gets a fresh widget at index=0.
        # Don't touch session_state[full_key] — Streamlit forbids it
        # after the widget already instantiated this render.
        st.session_state[nonce_key] += 1
        st.switch_page(page_path)


def render_top_nav(*, active: str | None = None,
                     show_position_dropdown: bool = True) -> None:
    """Convenience: render top tabs + position dropdown stacked.
    Kept for any pages that want both in one call."""
    render_top_tabs(active=active)
    if show_position_dropdown:
        render_position_dropdown()


__all__ = [
    "render_top_nav",
    "render_top_tabs",
    "render_position_dropdown",
    "TOP_TABS",
    "POSITION_OPTIONS",
]

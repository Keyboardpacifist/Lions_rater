"""Stylized "alpha card" component — the trading-card-style hero
tile used in the Today's-5 hero strip on Fantasy / Gambling pages.

Each card renders a single alpha record with:
  • Team-color gradient background
  • Player name (large, bold) + position chip
  • Big verdict tag (🚀 BUY, 💎 LOW-OWN, 👀 WATCH, ⚠️ FADE)
  • Headline number (total alpha tailwind FP)
  • One-line rationale (plain-English "why")

The lib is data-shape-agnostic — give it any record dict + a theme
dict and it'll render. Built so Gambling can reuse the same
visual primitive with sport / book / line numbers later.
"""
from __future__ import annotations

import streamlit as st


def _readable_text(hex_color: str) -> str:
    """Return 'white' or near-black depending on background luminance,
    so card text is always legible regardless of team primary color."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return "white"
    try:
        r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return "white"
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if lum < 0.6 else "#0d1530"


def render_alpha_card(*, player: str, position: str, team: str,
                         theme: dict, total_fp: float, verdict: str,
                         reason: str,
                         breakdown: list[tuple[str, float]] | None = None,
                         logo_url: str | None = None,
                         unit_label: str = "PROJECTED PPR TAILWIND",
                         number_format: str = "{sign}{value:.1f}",
                         ) -> None:
    """Render one alpha card. `theme` is the dict returned by
    lib_shared.team_theme(team_abbr). `breakdown` is an optional
    list of (factor_label, fp) tuples shown as small chips at the
    bottom of the card (e.g., ("Lamar bounce", 64.0)).

    `unit_label` is the small caption under the headline number —
    fantasy uses "PROJECTED PPR TAILWIND", gambling overrides to
    "% VS BASELINE" or similar.

    `number_format` is the format string for the headline number.
    Default "{sign}{value:.1f}" works for FP. Use "{sign}{value:.0f}%"
    for percentage edges."""
    primary = theme.get("primary", "#1F2A44")
    secondary = theme.get("secondary", "#0B1730")
    text_color = _readable_text(primary)
    sub_color = ("rgba(255,255,255,0.85)"
                 if text_color == "white" else "#3a4a5e")
    chip_bg = ("rgba(255,255,255,0.18)"
               if text_color == "white" else "rgba(0,0,0,0.08)")
    chip_text = text_color

    sign = "+" if total_fp >= 0 else ""
    fp_str = number_format.format(sign=sign, value=total_fp)

    breakdown_html = ""
    if breakdown:
        chips = []
        for label, val in breakdown:
            if abs(val) < 0.5:
                continue
            s = "+" if val >= 0 else ""
            chips.append(
                f"<span style='display:inline-block;background:{chip_bg};"
                f"color:{chip_text};padding:3px 8px;margin:2px 4px 2px 0;"
                f"border-radius:10px;font-size:10px;font-weight:600;"
                f"letter-spacing:0.2px;'>{label} {s}{val:.0f}</span>"
            )
        if chips:
            breakdown_html = (
                f"<div style='margin-top:10px;line-height:1.7;'>"
                f"{''.join(chips)}</div>"
            )

    logo_html = ""
    if logo_url:
        logo_html = (
            f'<img src="{logo_url}" style="height:40px;width:40px;'
            'object-fit:contain;opacity:0.95;'
            'filter:drop-shadow(0 2px 4px rgba(0,0,0,0.3));"/>'
        )

    card_html = (
        f'<div style="background:linear-gradient(135deg,{primary} 0%,'
        f'{secondary} 100%);border-radius:14px;padding:18px 16px;'
        f'color:{text_color};box-shadow:0 4px 12px rgba(0,0,0,0.16);'
        f'min-height:200px;display:flex;flex-direction:column;'
        f'justify-content:space-between;">'
        # Top row — verdict + logo
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;">'
        f'<div style="font-size:13px;font-weight:800;'
        f'letter-spacing:0.4px;background:{chip_bg};color:{chip_text};'
        f'padding:5px 10px;border-radius:8px;">{verdict}</div>'
        f'{logo_html}'
        f'</div>'
        # Player block
        f'<div style="margin-top:8px;">'
        f'<div style="font-size:18px;font-weight:900;line-height:1.1;'
        f'letter-spacing:-0.3px;">{player}</div>'
        f'<div style="font-size:11px;font-weight:600;opacity:0.85;'
        f'margin-top:3px;letter-spacing:0.6px;">'
        f'{position} · {team}</div>'
        f'</div>'
        # Big number
        f'<div style="margin-top:10px;">'
        f'<div style="font-size:32px;font-weight:900;line-height:1;'
        f'letter-spacing:-1px;">{fp_str}</div>'
        f'<div style="font-size:10px;font-weight:600;opacity:0.85;'
        f'letter-spacing:0.5px;margin-top:2px;">'
        f'{unit_label}</div>'
        f'</div>'
        # Reason
        f'<div style="margin-top:10px;font-size:12px;line-height:1.4;'
        f'color:{sub_color};">{reason}</div>'
        f'{breakdown_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def verdict_for(total_fp: float, share_2025: float | None = None) -> str:
    """Heuristic verdict tag from a total alpha number.

    Used by the Today's-5 builder so callers don't have to reinvent
    the cutoffs."""
    if total_fp >= 30:
        return "🚀 BIG BUY"
    if total_fp >= 15:
        return "🚀 BUY"
    if total_fp >= 5 and (share_2025 is not None and share_2025 < 0.10):
        return "💎 LOW-OWN BUY"
    if total_fp >= 3:
        return "👀 WATCH"
    if total_fp <= -5:
        return "⚠️ FADE"
    return "➡️ NEUTRAL"


__all__ = ["render_alpha_card", "verdict_for"]

"""
Trading-card export — produces a single-image PNG suitable for sharing
on Twitter, Reddit, Instagram. 4:5 portrait, 1080×1350.

Layout (top → bottom):
    ┌────────────────────────────────────┐  120px header strip
    │  [helmet]  PLAYER NAME             │   (team primary gradient)
    │            POSITION · SEASON       │
    ├────────────────────────────────────┤
    │  [head-                            │  hero zone (550px)
    │   shot]      SCORE: +1.23          │  half headshot, half score
    │              87th percentile        │
    ├────────────────────────────────────┤
    │  📖 STORY                          │  narrative band (200px)
    │  Signature: A-gap demon — 3rd of   │
    │  47 RBs at +0.19 EPA per carry.    │
    ├────────────────────────────────────┤
    │  ┌─stat─┐ ┌─stat─┐ ┌─stat─┐ ┌─stat─┐│  stats row (220px)
    │  │ ...  │ │ ...  │ │ ...  │ │ ...  ││
    │  └──────┘ └──────┘ └──────┘ └──────┘│
    ├────────────────────────────────────┤
    │  Built by [preset] · LR.streamlit  │  footer (60px)
    └────────────────────────────────────┘

Public API:
    build_player_card_png(...) -> bytes   (PNG)
"""
from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Canvas dimensions ───────────────────────────────────────────
CARD_W = 1080
CARD_H = 1350

# ── Layout zones (y bounds) ─────────────────────────────────────
HEADER_BOTTOM = 200
HERO_BOTTOM = 770
NARRATIVE_BOTTOM = 990
STATS_BOTTOM = 1280
FOOTER_BOTTOM = CARD_H

# ── Padding ─────────────────────────────────────────────────────
SIDE_PAD = 50

# ── Image cache (avoid re-downloading every export) ─────────────
_IMAGE_CACHE_DIR = Path("/tmp/lions_rater_card_assets")
_IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# Fonts — prefer macOS system fonts locally, fall back gracefully
# on Linux (Streamlit Cloud).
# ──────────────────────────────────────────────────────────────────
_FONT_CANDIDATES_BOLD = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
_FONT_CANDIDATES_REGULAR = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Best-available system font at the requested size."""
    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default(size=size)


# ──────────────────────────────────────────────────────────────────
# Color helpers
# ──────────────────────────────────────────────────────────────────
def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _readable_text_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return white or near-black depending on background luminance."""
    # WCAG-style relative luminance approximation
    r, g, b = bg_rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if luminance < 140 else (26, 26, 46)


# ──────────────────────────────────────────────────────────────────
# Image fetch (helmet logos + headshots) — cached on /tmp
# ──────────────────────────────────────────────────────────────────
def _fetch_image(url: str, cache_key: str) -> Image.Image | None:
    """Fetch a remote image; cache to /tmp. Returns PIL Image or None."""
    if not url:
        return None
    cache_path = _IMAGE_CACHE_DIR / f"{cache_key}.bin"
    if not cache_path.exists():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                cache_path.write_bytes(resp.read())
        except Exception:
            return None
    try:
        return Image.open(cache_path).convert("RGBA")
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────
def _draw_gradient_band(img: Image.Image, x0: int, y0: int, x1: int, y1: int,
                          color_left: tuple[int, int, int],
                          color_right: tuple[int, int, int]) -> None:
    """Draw a horizontal linear gradient between two RGB colors."""
    width = x1 - x0
    if width <= 0:
        return
    # Build a 1-pixel-tall gradient strip then resize vertically.
    strip = Image.new("RGB", (width, 1))
    px = strip.load()
    for x in range(width):
        t = x / max(width - 1, 1)
        r = int(color_left[0] * (1 - t) + color_right[0] * t)
        g = int(color_left[1] * (1 - t) + color_right[1] * t)
        b = int(color_left[2] * (1 - t) + color_right[2] * t)
        px[x, 0] = (r, g, b)
    strip = strip.resize((width, y1 - y0))
    img.paste(strip, (x0, y0))


def _truncate(text: str, draw: ImageDraw.ImageDraw,
                 font: ImageFont.FreeTypeFont, max_w: int) -> str:
    """Truncate with ellipsis if rendered text exceeds max_w."""
    if not text:
        return text
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] <= max_w:
        return text
    # Binary search for fit
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + "…"
        bb = draw.textbbox((0, 0), candidate, font=font)
        if bb[2] <= max_w:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _wrap(text: str, draw: ImageDraw.ImageDraw,
            font: ImageFont.FreeTypeFont, max_w: int,
            max_lines: int = 4) -> list[str]:
    """Word-wrap text to fit max_w, capped at max_lines (last line truncated)."""
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        candidate = (cur + " " + word).strip() if cur else word
        bb = draw.textbbox((0, 0), candidate, font=font)
        if bb[2] <= max_w:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
            if len(lines) >= max_lines - 1:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(words) > sum(len(line.split()) for line in lines):
        # Truncate last line if we ran out of room
        lines[-1] = _truncate(lines[-1] + " …", draw, font, max_w)
    return lines


# ──────────────────────────────────────────────────────────────────
# Section renderers
# ──────────────────────────────────────────────────────────────────
def _draw_header(img: Image.Image, draw: ImageDraw.ImageDraw, *,
                   player_name: str, position_label: str, season_str: str,
                   theme: dict) -> None:
    """Top band: team-color gradient with helmet, name, position·season."""
    primary = _hex_to_rgb(theme.get("primary", "#0076B6"))
    secondary = _hex_to_rgb(theme.get("secondary", "#B0B7BC"))
    text_color = _readable_text_color(primary)
    sub_color = tuple(int(c * 0.85 + 38) if text_color == (26, 26, 46)
                       else int(c * 0.75 + 64) for c in text_color)

    _draw_gradient_band(img, 0, 0, CARD_W, HEADER_BOTTOM, primary, secondary)

    # Helmet logo (left side)
    helmet_url = theme.get("logo")
    helmet = _fetch_image(helmet_url, theme.get("abbr", "x") + "_helmet")
    helmet_w = 0
    if helmet is not None:
        h_target = HEADER_BOTTOM - 50
        ratio = h_target / helmet.size[1]
        helmet_w = int(helmet.size[0] * ratio)
        helmet_resized = helmet.resize((helmet_w, h_target), Image.LANCZOS)
        img.paste(helmet_resized, (SIDE_PAD, 25), helmet_resized)
        helmet_w += 25  # add spacing after the logo

    # Player name + position·season (text right of helmet)
    name_x = SIDE_PAD + helmet_w
    name_max_w = CARD_W - name_x - SIDE_PAD
    name_font = _load_font(54, bold=True)
    sub_font = _load_font(24, bold=False)

    name_clipped = _truncate(player_name.upper(), draw, name_font, name_max_w)
    draw.text((name_x, 50), name_clipped, font=name_font, fill=text_color)

    sub_text = f"{position_label.upper()}  ·  {season_str.upper()}"
    draw.text((name_x, 122), sub_text, font=sub_font, fill=sub_color)


def _draw_hero(img: Image.Image, draw: ImageDraw.ImageDraw, *,
                 headshot_url: str | None, player_id_for_cache: str,
                 score_label: str, pct_label: str, theme: dict) -> None:
    """Middle hero zone: headshot on left, score on right."""
    primary = _hex_to_rgb(theme.get("primary", "#0076B6"))

    # Soft cream background for the hero zone
    draw.rectangle((0, HEADER_BOTTOM, CARD_W, HERO_BOTTOM),
                   fill=(248, 249, 251))

    # ── Headshot on left half ──
    headshot = _fetch_image(headshot_url, f"{player_id_for_cache}_headshot")
    headshot_box_x = SIDE_PAD
    headshot_box_y = HEADER_BOTTOM + 40
    headshot_box_w = 460
    headshot_box_h = HERO_BOTTOM - HEADER_BOTTOM - 80
    if headshot is not None:
        # Letterbox-fit the headshot in the box
        ratio = min(headshot_box_w / headshot.size[0],
                    headshot_box_h / headshot.size[1])
        new_w = int(headshot.size[0] * ratio)
        new_h = int(headshot.size[1] * ratio)
        resized = headshot.resize((new_w, new_h), Image.LANCZOS)
        paste_x = headshot_box_x + (headshot_box_w - new_w) // 2
        paste_y = headshot_box_y + (headshot_box_h - new_h) // 2
        img.paste(resized, (paste_x, paste_y), resized)
    else:
        # Placeholder dot
        cx = headshot_box_x + headshot_box_w // 2
        cy = headshot_box_y + headshot_box_h // 2
        draw.ellipse((cx - 80, cy - 80, cx + 80, cy + 80),
                      fill=(220, 224, 230))

    # ── Score banner on right half ──
    banner_x = headshot_box_x + headshot_box_w + 30
    banner_y = headshot_box_y + 40
    banner_w = CARD_W - banner_x - SIDE_PAD
    banner_h = headshot_box_h - 80

    # Background panel for score
    draw.rounded_rectangle(
        (banner_x, banner_y, banner_x + banner_w, banner_y + banner_h),
        radius=18, fill=primary,
    )
    # "YOUR SCORE" label
    score_label_font = _load_font(20, bold=True)
    draw.text((banner_x + 30, banner_y + 28),
              "YOUR SCORE", font=score_label_font,
              fill=_readable_text_color(primary))

    # Big score number
    score_font = _load_font(110, bold=True)
    text_w = draw.textbbox((0, 0), score_label, font=score_font)[2]
    cx_text = banner_x + (banner_w - text_w) // 2
    draw.text((cx_text, banner_y + 80),
              score_label, font=score_font,
              fill=_readable_text_color(primary))

    # Percentile sublabel
    pct_font = _load_font(28, bold=False)
    pct_text = f"{pct_label} percentile"
    bb = draw.textbbox((0, 0), pct_text, font=pct_font)
    cx_text = banner_x + (banner_w - bb[2]) // 2
    draw.text((cx_text, banner_y + 220),
              pct_text, font=pct_font,
              fill=_readable_text_color(primary))


def _draw_narrative(img: Image.Image, draw: ImageDraw.ImageDraw, *,
                      narrative: str | None, theme: dict) -> None:
    """Narrative band — signature/weakness blurb."""
    primary = _hex_to_rgb(theme.get("primary", "#0076B6"))

    # Light panel
    draw.rectangle((0, HERO_BOTTOM, CARD_W, NARRATIVE_BOTTOM),
                   fill=(255, 255, 255))
    # Left accent stripe
    draw.rectangle((SIDE_PAD, HERO_BOTTOM + 25,
                     SIDE_PAD + 6, NARRATIVE_BOTTOM - 25),
                    fill=primary)

    # Story label (no emoji — system fonts don't ship the glyph reliably)
    label_font = _load_font(18, bold=True)
    draw.text((SIDE_PAD + 25, HERO_BOTTOM + 28),
              "STORY", font=label_font,
              fill=(91, 107, 126))

    # Body text — wrap to fit
    body_font = _load_font(28, bold=False)
    body_x = SIDE_PAD + 25
    body_y = HERO_BOTTOM + 65
    body_max_w = CARD_W - body_x - SIDE_PAD
    body_text = (narrative or "").replace("**", "")
    lines = _wrap(body_text, draw, body_font, body_max_w, max_lines=4)
    for i, line in enumerate(lines):
        draw.text((body_x, body_y + i * 38),
                  line, font=body_font, fill=(42, 58, 77))


def _draw_stats_row(img: Image.Image, draw: ImageDraw.ImageDraw, *,
                       key_stats: list[tuple[str, str]],
                       theme: dict) -> None:
    """Bottom stats row — up to 4 stat tiles."""
    primary = _hex_to_rgb(theme.get("primary", "#0076B6"))

    # Light grey panel background
    draw.rectangle((0, NARRATIVE_BOTTOM, CARD_W, STATS_BOTTOM),
                   fill=(243, 246, 250))

    stats = key_stats[:4]
    if not stats:
        return
    n = len(stats)
    band_top = NARRATIVE_BOTTOM + 30
    band_bottom = STATS_BOTTOM - 30
    tile_h = band_bottom - band_top
    tile_gap = 18
    avail_w = CARD_W - 2 * SIDE_PAD - tile_gap * (n - 1)
    tile_w = avail_w // n

    label_font = _load_font(16, bold=True)
    value_font = _load_font(46, bold=True)
    sub_font = _load_font(16, bold=False)

    for i, (label, value, *rest) in enumerate(
            (s if len(s) >= 2 else (*s, "") for s in stats)):
        sub = rest[0] if rest else ""
        x0 = SIDE_PAD + i * (tile_w + tile_gap)
        x1 = x0 + tile_w
        y0 = band_top
        y1 = band_bottom
        # White card with team-color top border
        draw.rounded_rectangle((x0, y0, x1, y1), radius=14,
                                 fill=(255, 255, 255),
                                 outline=(214, 221, 230), width=1)
        draw.rectangle((x0, y0, x1, y0 + 4), fill=primary)

        # Label
        draw.text((x0 + 18, y0 + 18), label.upper(), font=label_font,
                  fill=(91, 107, 126))
        # Value (centered horizontally in tile)
        bb = draw.textbbox((0, 0), value, font=value_font)
        cx_v = x0 + (tile_w - bb[2]) // 2
        draw.text((cx_v, y0 + 50), value, font=value_font,
                  fill=(10, 61, 98))
        # Sub-label
        if sub:
            bb_s = draw.textbbox((0, 0), sub, font=sub_font)
            cx_s = x0 + (tile_w - bb_s[2]) // 2
            draw.text((cx_s, y0 + tile_h - 36),
                      sub, font=sub_font, fill=(91, 107, 126))


def _draw_footer(img: Image.Image, draw: ImageDraw.ImageDraw, *,
                    preset_name: str | None, theme: dict) -> None:
    """Footer: preset attribution + URL."""
    primary = _hex_to_rgb(theme.get("primary", "#0076B6"))

    draw.rectangle((0, STATS_BOTTOM, CARD_W, FOOTER_BOTTOM), fill=primary)

    fg = _readable_text_color(primary)
    foot_font = _load_font(20, bold=True)

    if preset_name and str(preset_name).strip():
        line1 = f"Built by «{preset_name}» preset"
    else:
        line1 = "lions-rater.streamlit.app  ·  build your football rater"

    bb = draw.textbbox((0, 0), line1, font=foot_font)
    cx = (CARD_W - bb[2]) // 2
    cy = STATS_BOTTOM + (FOOTER_BOTTOM - STATS_BOTTOM - 22) // 2
    draw.text((cx, cy), line1, font=foot_font, fill=fg)


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────
def render_card_download_button(*,
                                  player_name: str,
                                  position_label: str,
                                  season_str: str,
                                  score: float | None,
                                  narrative: str | None,
                                  key_stats: list[tuple[str, str]],
                                  player_id: str | int,
                                  team_abbr: str,
                                  theme: dict,
                                  preset_name: str | None = None,
                                  key_prefix: str = "card") -> None:
    """Streamlit-side wrapper — builds the PNG and renders the download
    button. Headshot lookup is cached. Imports streamlit lazily so
    lib_trading_card stays importable from non-streamlit contexts."""
    import streamlit as st
    import pandas as pd
    from scipy.stats import norm

    @st.cache_data
    def _player_meta_for(pid: str) -> dict:
        """One rosters round-trip → headshot URL + jersey number.
        Returns {} on any failure so callers can default cleanly."""
        try:
            import nflreadpy as nfl
        except Exception:
            return {}
        try:
            # Most-recent season first (team photos refresh annually).
            ros = nfl.load_rosters([2025]).to_pandas()
            matches = ros[ros.get("gsis_id") == pid]
            if matches.empty:
                ros = nfl.load_rosters([2024]).to_pandas()
                matches = ros[ros.get("gsis_id") == pid]
            if matches.empty:
                return {}
            row = matches.iloc[0]
            out = {}
            url = row.get("headshot_url")
            if url and not pd.isna(url):
                out["headshot"] = str(url)
            num = row.get("jersey_number")
            if num is not None and not pd.isna(num):
                try:
                    out["number"] = int(num)
                except (ValueError, TypeError):
                    pass
            return out
        except Exception:
            return {}

    _meta = _player_meta_for(str(player_id)) if player_id else {}
    headshot_url = _meta.get("headshot")
    jersey_number = _meta.get("number")

    if score is None or pd.isna(score):
        score_label = "—"
        pct_label = "—"
    else:
        sign = "+" if score >= 0 else ""
        score_label = f"{sign}{score:.2f}"
        pct_val = float(norm.cdf(score) * 100)
        pct_label = f"{int(pct_val)}th"

    png_bytes = build_player_card_png(
        player_name=player_name,
        position_label=position_label,
        season_str=season_str,
        score=score,
        score_label=score_label,
        pct_label=pct_label,
        narrative=narrative,
        key_stats=key_stats,
        headshot_url=headshot_url,
        player_id_for_cache=str(player_id),
        theme=theme,
        preset_name=preset_name,
    )

    safe_name = "".join(c for c in player_name if c.isalnum() or c in " -_").strip()
    safe_name = safe_name.replace(" ", "_") or "player"
    filename = f"{safe_name}_card.png"

    # Render the card mounted on a full-width team banner — wings flank
    # the card with team logo (left) and jersey number (right). Wings
    # stretch to match the card's height via flexbox align-items: stretch.
    # Card stays clean (4:5 share-ready PNG); wings are HTML-only chrome.
    import base64
    primary = theme.get("primary", "#1F2A44")
    secondary = theme.get("secondary", "#0B1730")
    logo_url = theme.get("logo", "")
    b64 = base64.b64encode(png_bytes).decode("ascii")

    # Card centered inside the team-color gradient banner. Jersey number
    # is overlaid on the card itself, positioned to align with the "6" of
    # the score (~74% from card left) and just below the "percentile"
    # text (~41% from card top). HTML-only — the share PNG stays clean.
    jersey_overlay = (
        f'<div class="lr-jersey-num">#{jersey_number}</div>'
        if jersey_number is not None else ""
    )

    st.markdown(
        f"""
<style>
.lr-card-banner {{
    background: linear-gradient(135deg, {primary} 0%, {secondary} 100%);
    border-radius: 18px;
    margin: 0 0 12px 0;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    padding: 28px 24px;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.lr-card-center {{
    position: relative;
    flex: 0 0 auto;
    max-width: 480px;
    width: 100%;
}}
.lr-card-img {{
    width: 100%;
    border-radius: 12px;
    box-shadow: 0 12px 32px rgba(0,0,0,0.45);
    display: block;
}}
.lr-card-caption {{
    color: rgba(255,255,255,0.85);
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.3px;
    text-align: center;
    margin-top: 10px;
}}
.lr-jersey-num {{
    position: absolute;
    /* x: aligns with the "6" of the score on the card (~74% from left).
       y: just under the "percentile" sublabel (~41% from top). */
    left: 74%;
    top: 41%;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: clamp(22px, 4.4vw, 48px);
    font-weight: 900;
    color: rgba(255, 255, 255, 0.96);
    line-height: 1;
    letter-spacing: -1.5px;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.45);
    white-space: nowrap;
    pointer-events: none;
}}
</style>
<div class="lr-card-banner">
    <div class="lr-card-center">
        <img src="data:image/png;base64,{b64}" class="lr-card-img" alt="{player_name} trading card"/>
        {jersey_overlay}
        <div class="lr-card-caption">{player_name} — {season_str}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    _spacer_l, _btn_col, _spacer_r = st.columns([1, 2, 1])
    with _btn_col:
        st.download_button(
            label="🃏  Download trading card",
            data=png_bytes,
            file_name=filename,
            mime="image/png",
            key=f"{key_prefix}_card_dl",
            help="4:5 trading-card export — sized for Twitter / "
                 "Reddit / Instagram. Built from your slider preset.",
            use_container_width=True,
        )


def build_player_card_png(*,
                            player_name: str,
                            position_label: str,
                            season_str: str,
                            score: float | None,
                            score_label: str,
                            pct_label: str,
                            narrative: str | None,
                            key_stats: list[tuple[str, str]],
                            headshot_url: str | None,
                            player_id_for_cache: str,
                            theme: dict,
                            preset_name: str | None = None) -> bytes:
    """Compose a 4:5 portrait trading card and return PNG bytes."""
    img = Image.new("RGB", (CARD_W, CARD_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    _draw_header(img, draw,
                  player_name=player_name,
                  position_label=position_label,
                  season_str=season_str,
                  theme=theme)
    _draw_hero(img, draw,
                headshot_url=headshot_url,
                player_id_for_cache=player_id_for_cache,
                score_label=score_label,
                pct_label=pct_label,
                theme=theme)
    _draw_narrative(img, draw, narrative=narrative, theme=theme)
    _draw_stats_row(img, draw, key_stats=key_stats, theme=theme)
    _draw_footer(img, draw, preset_name=preset_name, theme=theme)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

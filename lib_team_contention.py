"""Team contention-state classifier.

Maps a (team, season) row from team_context.parquet (joined to
team_seasons.parquet) to one of five fan-vocabulary states:

    Rebuild · Ascending · Playoff Contender · Super Bowl Contender · Fading

The classifier reads team rating, year-over-year trajectory, roster
age, cap concentration, and window-remaining indicators, then
applies a rule-based decision tree intended to match how fans
actually perceive these archetypes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_team_context() -> pd.DataFrame:
    path = _DATA / "team_context.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# Visual treatment per state — color used for the badge in the hero.
STATE_STYLES = {
    "Super Bowl Contender": {"bg": "#FFD700", "fg": "#1A1A1A",
                                "icon": "🏆"},
    "Playoff Contender":    {"bg": "#34A853", "fg": "#FFFFFF",
                                "icon": "✅"},
    "Ascending":            {"bg": "#1E90FF", "fg": "#FFFFFF",
                                "icon": "📈"},
    "Fading":               {"bg": "#E67E22", "fg": "#FFFFFF",
                                "icon": "🌅"},
    "Rebuild":              {"bg": "#888888", "fg": "#FFFFFF",
                                "icon": "🔨"},
}


def classify_team(team: str, season: int) -> dict:
    """Classify a team-season into one of the five contention states.
    Returns: {state, rationale, signals}."""
    ctx = load_team_context()
    row = ctx[(ctx["team"] == team) & (ctx["season"] == season)]
    if row.empty:
        return {"state": "Unknown", "rationale": "", "signals": {}}
    r = row.iloc[0]

    rating = float(r.get("team_rating", 0) or 0)
    traj = float(r.get("trajectory", 0) or 0)
    age = float(r.get("avg_age", 27) or 27)
    window = float(r.get("avg_years_remaining_top5", 3) or 3)
    cap_top5 = float(r.get("top5_apy_pct_of_cap", 0) or 0)

    signals = {
        "team_rating": rating,
        "trajectory": traj,
        "avg_age": age,
        "window_years_remaining": window,
        "top5_cap_pct": cap_top5,
    }

    # ── Decision tree (intentionally readable, not optimized) ──
    # SUPER BOWL CONTENDER: elite rating, prime-age core, window open
    if rating >= 0.08 and age < 28 and window >= 2.5:
        return {
            "state": "Super Bowl Contender",
            "rationale": (
                f"Top-tier rating ({rating:+.2f}), young core "
                f"({age:.1f} yrs avg), {window:.1f} yrs of window left."
            ),
            "signals": signals,
        }

    # SUPER BOWL CONTENDER (alt path): elite rating + experienced but
    # not old core — the "ready now" champion archetype
    if rating >= 0.12 and age < 29:
        return {
            "state": "Super Bowl Contender",
            "rationale": (
                f"Elite rating ({rating:+.2f}) with a roster in its prime "
                f"({age:.1f} avg). The window is now."
            ),
            "signals": signals,
        }

    # FADING: above-water rating but aging core + trajectory flat or down
    if age >= 29 and rating >= 0.0 and traj <= 0.02:
        return {
            "state": "Fading",
            "rationale": (
                f"Aging core ({age:.1f} avg) with the window closing — "
                f"trajectory {traj:+.2f} from last year."
            ),
            "signals": signals,
        }

    # PLAYOFF CONTENDER: solid rating, balanced roster
    if rating >= 0.03 and traj >= -0.05:
        return {
            "state": "Playoff Contender",
            "rationale": (
                f"Above-average rating ({rating:+.2f}), balanced roster, "
                f"trajectory {traj:+.2f}."
            ),
            "signals": signals,
        }

    # ASCENDING: rising trajectory + young + cap flexibility
    if traj >= 0.05 and age < 28:
        return {
            "state": "Ascending",
            "rationale": (
                f"Year-over-year improvement ({traj:+.2f}) with a young "
                f"core ({age:.1f} avg). Window is opening."
            ),
            "signals": signals,
        }

    # FADING fallback: weak rating + old
    if age >= 28.5 and rating < 0.0:
        return {
            "state": "Fading",
            "rationale": (
                f"Old roster ({age:.1f} avg) without the production to "
                f"justify it ({rating:+.2f} rating)."
            ),
            "signals": signals,
        }

    # ASCENDING fallback: any positive trajectory + youngish
    if traj >= 0.03 and age < 28.5:
        return {
            "state": "Ascending",
            "rationale": (
                f"Rising trajectory ({traj:+.2f}) and a young roster "
                f"({age:.1f} avg) — building momentum."
            ),
            "signals": signals,
        }

    # PLAYOFF CONTENDER fallback: barely-positive rating
    if rating > -0.02:
        return {
            "state": "Playoff Contender",
            "rationale": (
                f"Borderline rating ({rating:+.2f}) — capable of "
                f"making the playoffs but unproven beyond that."
            ),
            "signals": signals,
        }

    # REBUILD: default for everything else (low rating, no clear upward signal)
    return {
        "state": "Rebuild",
        "rationale": (
            f"Below-average rating ({rating:+.2f}), trajectory "
            f"{traj:+.2f} — building from the ground up."
        ),
        "signals": signals,
    }


def render_contention_badge(state: str, rationale: str) -> str:
    """Return styled HTML for the contention badge — drop into the
    team-page hero with unsafe_allow_html=True."""
    style = STATE_STYLES.get(state, STATE_STYLES["Rebuild"])
    return f"""
<div style="
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: {style['bg']};
    color: {style['fg']};
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
">
    <span style="font-size: 18px;">{style['icon']}</span>
    <span>{state.upper()}</span>
</div>
<div style="margin-top: 8px; font-size: 13px; opacity: 0.85;
             font-style: italic; max-width: 600px;">
    {rationale}
</div>
"""

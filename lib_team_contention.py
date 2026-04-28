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
    """Return styled HTML for the contention badge. Single-line / no
    leading whitespace per line — Streamlit's markdown parser treats
    4+ space indentation as code blocks and breaks HTML structure."""
    style = STATE_STYLES.get(state, STATE_STYLES["Rebuild"])
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;'
        f'background:{style["bg"]};color:{style["fg"]};padding:8px 16px;'
        f'border-radius:999px;font-size:14px;font-weight:800;'
        f'letter-spacing:0.5px;box-shadow:0 2px 6px rgba(0,0,0,0.2);">'
        f'<span style="font-size:18px;">{style["icon"]}</span>'
        f'<span>{state.upper()}</span>'
        f'</div>'
        f'<div style="margin-top:8px;font-size:13px;opacity:0.85;'
        f'font-style:italic;max-width:600px;">{rationale}</div>'
    )


# ── Gap analysis — "what's keeping them from the next stage" ──
# Per stat: (label, column, ascending=True if lower-is-better, fan-friendly
# remediation phrase). Used to identify the team's biggest weaknesses
# and frame what they need to fix.
_GAP_STATS = [
    ("offensive efficiency",       "off_epa_per_play",         False,
     "the offense isn't pulling its weight per play"),
    ("passing offense",            "off_pass_epa_per_play",    False,
     "the passing game needs more juice"),
    ("rushing offense",            "off_rush_epa_per_play",    False,
     "the run game has to be more dangerous"),
    ("red zone TD rate",           "off_red_zone_td_rate",     False,
     "settling for field goals where they need touchdowns"),
    ("3rd down conversion",        "off_third_down_conv_rate", False,
     "drives die on 3rd down too often"),
    ("ball security",              "off_giveaway_rate",        True,
     "giving the ball away too much"),
    ("defensive efficiency",       "def_epa_per_play",         True,
     "the defense gives up too much per play"),
    ("pass defense",               "def_pass_epa_allowed",     True,
     "the secondary is leaking yards"),
    ("run defense",                "def_rush_epa_allowed",     True,
     "the front seven gets pushed around"),
    ("takeaway production",        "def_takeaway_rate",        False,
     "the defense doesn't generate enough turnovers"),
    ("pass rush",                  "def_pressure_rate",        False,
     "the pass rush isn't getting home"),
    ("4th-quarter offense",        "fourth_q_off_epa",         False,
     "they fade in the 4th quarter offensively"),
    ("4th-quarter defense",        "fourth_q_def_epa",         True,
     "the defense gives up too much in the 4th quarter"),
    ("discipline",                 "penalty_yards_per_game",   True,
     "they shoot themselves in the foot with penalties"),
]


def _rank_in_season(team_seasons, team, season, stat, ascending):
    """Returns rank (1 = best, N = worst) within season."""
    pool = team_seasons[team_seasons["season"] == season].copy()
    pool = pool.dropna(subset=[stat])
    if pool.empty:
        return None, 0
    pool = pool.sort_values(stat, ascending=ascending).reset_index(drop=True)
    match = pool.index[pool["team"] == team].tolist()
    if not match:
        return None, len(pool)
    return match[0] + 1, len(pool)


def compute_gap_analysis(team_seasons, team: str, season: int,
                            n_gaps: int = 3) -> list[dict]:
    """Identify the team's `n_gaps` biggest weaknesses by league rank.
    Returns: list of {label, rank, total, phrase} sorted worst-first."""
    gaps = []
    for label, col, ascending, phrase in _GAP_STATS:
        if col not in team_seasons.columns:
            continue
        rank, total = _rank_in_season(team_seasons, team, season,
                                          col, ascending)
        if rank is None or total == 0:
            continue
        # Only consider ranks below the median (i.e. weaknesses)
        if rank <= total // 2:
            continue
        gaps.append({
            "label": label,
            "rank": rank,
            "total": total,
            "phrase": phrase,
            "rank_pct": rank / total,
        })
    gaps.sort(key=lambda g: g["rank_pct"], reverse=True)
    return gaps[:n_gaps]


_GAP_TITLES = {
    "Super Bowl Contender": "What's keeping them from a championship",
    "Playoff Contender":    "What's keeping them out of true contention",
    "Ascending":            "What still needs to come together",
    "Fading":               "What's slipping fastest",
    "Rebuild":              "Where the climb has to start",
}


def render_gap_analysis_html(state: str, gaps: list[dict]) -> str:
    """Format the gap-analysis block as styled HTML. Single-line per
    HTML element — Streamlit's markdown treats 4+ space indentation
    as code blocks."""
    if not gaps:
        return ""
    title = _GAP_TITLES.get(state, "Biggest gaps")
    items = []
    for g in gaps:
        items.append(
            '<div style="display:flex;gap:12px;align-items:center;'
            'padding:10px 14px;background:rgba(255,255,255,0.05);'
            'border-left:3px solid #E67E22;border-radius:0 8px 8px 0;'
            'margin-bottom:8px;">'
            '<div style="font-size:11px;font-weight:800;'
            'letter-spacing:1.5px;background:rgba(230,126,34,0.85);'
            'color:white;padding:4px 8px;border-radius:6px;min-width:70px;'
            f'text-align:center;">{g["rank"]} of {g["total"]}</div>'
            '<div style="font-size:14px;line-height:1.4;">'
            f'<span style="font-weight:700;text-transform:capitalize;">{g["label"]}</span>'
            f'<span style="opacity:0.85;"> — {g["phrase"]}.</span>'
            '</div>'
            '</div>'
        )
    return (
        '<div style="margin-top:18px;padding:16px 18px;'
        'background:rgba(0,0,0,0.18);border-radius:12px;color:white;">'
        '<div style="font-size:12px;font-weight:800;letter-spacing:1.5px;'
        f'opacity:0.85;margin-bottom:10px;">🎯  {title.upper()}</div>'
        + "".join(items)
        + '</div>'
    )

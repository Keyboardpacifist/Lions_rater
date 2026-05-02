"""Smart Alerts fusion engine — Feature 4.4.

Takes a single news event (player_name, team, status, body_part) and
produces a multi-paragraph alert that fuses every other engine's output:

  • Feature 4.1 — Pr(plays Sunday) + usage retention from cohort
  • Feature 4.2 — game-script shift (pass rate, points, plays/game)
  • Feature 4.3 — book behavior context (over/under-react history)
  • Feature 4.5 — weather context if relevant

The output is a structured `AlertBundle` that the UI can render. The
news ingestion (RSS scrapers, headline detection) is a separate
concern and not built here — this is the pure fusion engine that
runs on top of any source.

Public entry point
------------------
    fuse_alert(player_name, team, position, status, body_part,
               opponent=None, season=None, week=None) -> AlertBundle
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

from lib_injury_cohort import predict as cohort_predict


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
GAME_SCRIPT = DATA / "game_script_deltas.parquet"
BOOKS_VS_MODEL = DATA / "books_vs_model.parquet"
SCHEDULES = DATA / "nfl_schedules.parquet"


@st.cache_data(show_spinner=False)
def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@dataclass
class AlertBundle:
    headline: str
    cohort_line: str        # Pr(plays) summary
    game_script_line: str   # Team scoring/pass-rate impact
    book_behavior_line: str # Historical book over/under-react
    weather_line: str       # Weather context (if applicable)
    bullet_points: list[str] = field(default_factory=list)


def _role_for(position: str) -> str:
    """Map raw position to the role bucket used in game-script deltas."""
    p = (position or "").upper()
    if p == "QB":
        return "QB1"
    if p in ("RB", "FB", "HB"):
        return "RB1"
    if p == "WR":
        return "WR1"
    if p == "TE":
        return "TE1"
    return "OTHER"


def _game_script_clip(role: str) -> dict | None:
    df = _load(GAME_SCRIPT)
    if df.empty:
        return None
    sub = df[df["scenario"] == role]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def _books_clip(role: str, status: str, body_part: str) -> dict | None:
    """Find the (role, status, body_part) cohort in books-vs-model.
    Falls back through (role, status) and finally (role) if specific
    body part is too thin."""
    df = _load(BOOKS_VS_MODEL)
    if df.empty:
        return None
    # role_lost in the table uses 'QB' not 'QB1'
    role_key = role.replace("1", "")
    sub = df[(df["position_lost"] == role_key)
             & (df["status"] == status)
             & (df["body_part"] == body_part)]
    if sub.empty or sub.iloc[0]["n_games"] < 8:
        sub = df[(df["position_lost"] == role_key)
                 & (df["status"] == status)]
    if sub.empty or sub.iloc[0]["n_games"] < 20:
        sub = df[(df["position_lost"] == role_key)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def _weather_for_game(team: str, season: int, week: int) -> dict | None:
    df = _load(SCHEDULES)
    if df.empty:
        return None
    sub = df[(df["season"] == season)
             & (df["week"] == week)
             & ((df["home_team"] == team) | (df["away_team"] == team))]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return {
        "temp": row.get("temp"),
        "wind": row.get("wind"),
        "roof": row.get("roof"),
        "surface": row.get("surface"),
        "stadium": row.get("stadium"),
    }


def _format_pct(x, sign: bool = False) -> str:
    if x is None or pd.isna(x):
        return "—"
    if sign:
        return f"{x:+.1%}"
    return f"{x:.1%}"


def _format_num(x, decimals: int = 1, sign: bool = False) -> str:
    if x is None or pd.isna(x):
        return "—"
    if sign:
        return f"{x:+.{decimals}f}"
    return f"{x:.{decimals}f}"


def fuse_alert(player_name: str, team: str, position: str,
               status: str, body_part: str = "unknown",
               practice_status: str = "DNP",
               opponent: str | None = None,
               season: int | None = None,
               week: int | None = None) -> AlertBundle:
    """Build the fused alert bundle for one news event."""
    role = _role_for(position)
    bullets: list[str] = []

    # ── 4.1 Cohort
    cohort = cohort_predict(
        position=position, body_part=body_part,
        report_status=status, practice_status=practice_status,
    )
    cohort_line = (
        f"Cohort: {cohort.n} comparable cases ({cohort.cohort_level}). "
        f"Pr(plays Sunday) = {_format_pct(cohort.p_played)}. "
        f"If active, snap share retention "
        f"= {_format_pct(cohort.snap_share_if_played)}."
    )
    bullets.append(
        f"**Play probability:** {_format_pct(cohort.p_played)} "
        f"({cohort.cohort_level} cohort, n={cohort.n})"
    )

    # ── 4.2 Game script
    gs = _game_script_clip(role) if role != "OTHER" else None
    if gs:
        gs_line = (
            f"League-wide when a team's {role} is out: "
            f"points/game {_format_num(gs['points_per_game_delta'], 1, sign=True)}, "
            f"pass rate {_format_pct(gs['pass_rate_delta'], sign=True)}, "
            f"plays/game {_format_num(gs['plays_per_game_delta'], 1, sign=True)} "
            f"(n={int(gs['n_games'])} historical games)."
        )
        bullets.append(
            f"**Team scoring shift (league-wide):** "
            f"{_format_num(gs['points_per_game_delta'], 1, sign=True)} pts/game"
        )
    else:
        gs_line = "Game-script delta unavailable for this role."

    # ── 4.3 Book behavior
    bv = _books_clip(role, status, body_part) if role != "OTHER" else None
    if bv:
        miss = bv["mean_line_miss"]
        cov = bv["cover_rate"]
        # line_miss > 0  →  affected team outperformed the close → books
        # OVER-reacted to the news (set the line too negatively for this
        # team) → contrarian play is to bet ON the affected team.
        # line_miss < 0  →  team underperformed the close → books
        # UNDER-reacted (didn't drop the line enough) → fade the team.
        if miss > 1.5:
            book_verdict = ("Books historically OVER-react to this kind of "
                            "news — contrarian bet ON the affected team")
        elif miss < -1.5:
            book_verdict = ("Books historically UNDER-react — fade the "
                            "affected team")
        else:
            book_verdict = "Books price this fairly on average"
        bv_line = (
            f"Book behavior: in {int(bv['n_games'])} comparable historical "
            f"games (role={role.replace('1','')}, status={status}, "
            f"body={body_part}), the affected team's mean line miss was "
            f"{_format_num(miss, 1, sign=True)} pts and cover rate was "
            f"{_format_pct(cov)}. {book_verdict}."
        )
        bullets.append(
            f"**Book behavior:** {_format_num(miss, 1, sign=True)} pts vs. close, "
            f"{_format_pct(cov)} cover rate"
        )
    else:
        bv_line = "Book behavior data unavailable for this scenario."

    # ── 4.5 Weather
    if season is not None and week is not None:
        w = _weather_for_game(team, int(season), int(week))
        if w and pd.notna(w.get("temp")):
            w_line = (
                f"Weather: {w['stadium']} "
                f"({_format_num(w['temp'], 0)}°F, "
                f"{_format_num(w['wind'], 0)} mph wind, "
                f"roof: {w['roof']}, surface: {w['surface']})."
            )
        elif w and w.get("roof") in ("dome", "closed"):
            w_line = f"Weather: indoor — {w['stadium']} ({w['roof']})."
        else:
            w_line = "Weather: not yet posted."
    else:
        w_line = ""

    headline = (
        f"⚠️ {player_name} ({position}, {team}) "
        f"— {status.title()} ({body_part})"
    )

    return AlertBundle(
        headline=headline,
        cohort_line=cohort_line,
        game_script_line=gs_line,
        book_behavior_line=bv_line,
        weather_line=w_line,
        bullet_points=bullets,
    )

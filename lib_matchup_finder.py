"""Matchup Edge Finder — scans the slate for game-bet inefficiencies.

For a given (season, week), runs every game on the slate through the
matchup-report engine and surfaces the games where the narrative
flagged a directional lean with at least a "SIGNAL DETECTED" rating.

Each row in the output is one game with:
  - Teams + line + total
  - The directional lean ("Take HOU -4.5", "UNDER 47.5", or "PASS")
  - Confidence rating (1-5 stars)
  - Top 1-2 signals driving the lean
  - Risk flags

Public entry point
------------------
    generate_matchup_findings(season, week, min_confidence=2)
        → DataFrame ranked by confidence × signal-count
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from lib_matchup_report import generate_matchup_report


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
SCHEDULES = DATA / "nfl_schedules.parquet"


@st.cache_data(show_spinner=False)
def _load_schedules() -> pd.DataFrame:
    return pd.read_parquet(SCHEDULES)


@dataclass
class MatchupFinding:
    home_team: str
    away_team: str
    season: int
    week: int
    spread_line: float | None
    total_line: float | None
    primary_lean: str
    confidence: int               # 1-5 stars
    confidence_label: str
    secondary_lean: str | None
    why_top: list[str]            # top 1-2 driver bullets
    risk_flags: list[str]
    n_starter_absences: int       # count of QB1/RB1/WR1/TE1 OUT
    has_books_signal: bool        # ≥1 historical book-behavior cohort matched


@st.cache_data(show_spinner=False, ttl=3600)
def generate_matchup_findings(season: int, week: int,
                                 min_confidence: int = 2
                                 ) -> pd.DataFrame:
    """Top-level entry point. Returns DataFrame ranked by
    confidence × |spread_score|. Cached 1 hour."""
    sch = _load_schedules()
    games = sch[(sch["season"] == int(season))
                & (sch["week"] == int(week))].copy()
    if games.empty:
        return pd.DataFrame()

    findings: list[MatchupFinding] = []
    for _, g in games.iterrows():
        home = str(g["home_team"])
        away = str(g["away_team"])
        try:
            r = generate_matchup_report(home, away, int(season),
                                          int(week))
        except Exception:
            continue
        if r.headline.get("error"):
            continue
        if not r.narrative:
            continue
        n = r.narrative
        if n.primary_confidence < min_confidence:
            continue   # below threshold — keep it as PASS

        # Extract top 1-2 why bullets
        why_top = n.why_bullets[:2] if n.why_bullets else []

        n_absences = sum(
            1 for inj in (r.home_injuries + r.away_injuries)
            if inj.role in ("QB1", "RB1", "WR1", "TE1")
            and inj.status in ("OUT", "DOUBTFUL")
        )

        findings.append(MatchupFinding(
            home_team=home, away_team=away,
            season=int(season), week=int(week),
            spread_line=(float(g["spread_line"])
                          if pd.notna(g.get("spread_line")) else None),
            total_line=(float(g["total_line"])
                         if pd.notna(g.get("total_line")) else None),
            primary_lean=n.primary_lean,
            confidence=n.primary_confidence,
            confidence_label=n.primary_confidence_label,
            secondary_lean=n.secondary_lean,
            why_top=why_top,
            risk_flags=n.risk_flags,
            n_starter_absences=n_absences,
            has_books_signal=bool(r.books_signals),
        ))

    if not findings:
        return pd.DataFrame()
    df = pd.DataFrame([f.__dict__ for f in findings])
    df = df.sort_values(
        ["confidence", "n_starter_absences"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return df


def latest_week_with_games(season: int) -> int:
    sch = _load_schedules()
    sub = sch[(sch["season"] == int(season))
              & sch["spread_line"].notna()]
    if sub.empty:
        return 1
    return int(sub["week"].max())

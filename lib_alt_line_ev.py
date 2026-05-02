"""Alt-Line EV Finder — Feature 5.3.

Models a player's empirical stat distribution and converts any
(threshold, american_odds) input into a model probability and
expected value. Sorts a ladder of alt-line rungs by EV so the
sharpest rung surfaces first.

Why this matters
----------------
Most bettors reflexively bet main lines and leave 5–15% of EV on
the table. Books price the main line tight; alt-lines often have
mis-pricing because they're priced off a smooth model that doesn't
match the player's empirical distribution.

Public entry points
-------------------
    american_to_decimal(odds)
    decimal_to_implied_prob(decimal)
    p_over_threshold(player_id, stat, threshold, lookback_games=None)
    rank_ladder(player_id, stat, ladder_rungs)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


REPO = Path(__file__).resolve().parent
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"


@st.cache_data(show_spinner=False)
def _load_stats() -> pd.DataFrame:
    if not PLAYER_STATS.exists():
        return pd.DataFrame()
    return pd.read_parquet(PLAYER_STATS)


# ── Odds-conversion helpers ──────────────────────────────────────

def american_to_decimal(odds: int | float) -> float:
    """Convert American odds (-110, +180) to decimal (1.909, 2.80)."""
    o = float(odds)
    if o > 0:
        return 1.0 + o / 100.0
    return 1.0 + 100.0 / abs(o)


def decimal_to_implied_prob(decimal: float) -> float:
    """Implied probability of a decimal-odds price (no juice removed)."""
    return 1.0 / decimal if decimal > 0 else float("nan")


def expected_value(p_win: float, decimal_odds: float) -> float:
    """EV per 1 unit risked. Positive = +EV bet."""
    return p_win * decimal_odds - 1.0


# ── Empirical distribution lookup ────────────────────────────────

@dataclass
class RungEV:
    threshold: float
    side: str           # "over" / "under"
    american_odds: int
    decimal_odds: float
    p_model: float
    p_implied: float
    edge: float         # p_model - p_implied
    ev: float           # EV per unit risked
    n_games: int


def p_over_threshold(player_id: str, stat: str, threshold: float,
                      lookback_games: int | None = None
                      ) -> tuple[float, int]:
    """Empirical probability that the player's stat exceeds threshold
    in the next game. Returns (p_model, n_games_in_sample).

    `lookback_games` (optional) restricts to the player's most-recent
    N games, useful when role has changed."""
    df = _load_stats()
    if df.empty or stat not in df.columns:
        return float("nan"), 0
    sub = df[(df["player_id"] == player_id) & df[stat].notna()]
    if sub.empty:
        return float("nan"), 0
    sub = sub.sort_values(["season", "week"], ascending=[False, False])
    if lookback_games:
        sub = sub.head(int(lookback_games))
    vals = sub[stat].astype(float)
    if len(vals) == 0:
        return float("nan"), 0
    p = float((vals > float(threshold)).mean())
    return p, len(vals)


def rank_ladder(player_id: str, stat: str,
                ladder_rungs: list[tuple[float, str, int]],
                lookback_games: int | None = None
                ) -> pd.DataFrame:
    """Score each rung in a ladder and sort by EV.

    ladder_rungs: list of (threshold, "over"|"under", american_odds).
    """
    rows: list[RungEV] = []
    for threshold, side, odds in ladder_rungs:
        p_over, n = p_over_threshold(player_id, stat, threshold,
                                      lookback_games)
        if np.isnan(p_over):
            continue
        p_model = p_over if side.lower() == "over" else (1.0 - p_over)
        decimal = american_to_decimal(odds)
        p_imp = decimal_to_implied_prob(decimal)
        ev = expected_value(p_model, decimal)
        rows.append(RungEV(
            threshold=float(threshold),
            side=side.lower(),
            american_odds=int(odds),
            decimal_odds=decimal,
            p_model=p_model,
            p_implied=p_imp,
            edge=p_model - p_imp,
            ev=ev,
            n_games=n,
        ))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame([r.__dict__ for r in rows])
    return out.sort_values("ev", ascending=False).reset_index(drop=True)


def player_distribution(player_id: str, stat: str,
                        lookback_games: int | None = None
                        ) -> pd.DataFrame:
    """Return the player's per-game stat values for plotting."""
    df = _load_stats()
    if df.empty or stat not in df.columns:
        return pd.DataFrame()
    sub = df[(df["player_id"] == player_id) & df[stat].notna()]
    sub = sub.sort_values(["season", "week"], ascending=[False, False])
    if lookback_games:
        sub = sub.head(int(lookback_games))
    return sub[["season", "week", "team", "opponent_team",
                stat]].rename(columns={stat: "value"}).reset_index(drop=True)

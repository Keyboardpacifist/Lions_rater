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


def wilson_interval(k: int, n: int,
                     z: float = 1.96
                     ) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion.

    Recommended over the naïve Wald interval for small n — handles
    boundary cases (k=0 or k=n) gracefully and gives the correct
    coverage at small samples.

    Returns (low, high). When n=0 returns (0.0, 1.0).
    """
    if n <= 0:
        return 0.0, 1.0
    phat = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = z * (((phat * (1.0 - phat)) / n
                 + z2 / (4.0 * n * n)) ** 0.5) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


# ── Empirical distribution lookup ────────────────────────────────

@dataclass
class RungEV:
    threshold: float
    side: str           # "over" / "under"
    american_odds: int
    decimal_odds: float
    p_model: float
    p_model_ci_low: float    # Wilson 95% CI lower bound
    p_model_ci_high: float   # Wilson 95% CI upper bound
    p_implied: float
    edge: float         # p_model - p_implied
    ev: float           # EV per unit risked at p_model
    ev_low: float       # EV at lower CI bound (conservative)
    n_games: int


def p_over_threshold(player_id: str, stat: str, threshold: float,
                      lookback_games: int | None = None
                      ) -> tuple[float, int, int]:
    """Empirical probability that the player's stat exceeds threshold
    in the next game. Returns (p_model, k_successes, n_games_in_sample)
    so callers can compute their own confidence intervals.

    `lookback_games` (optional) restricts to the player's most-recent
    N games, useful when role has changed."""
    df = _load_stats()
    if df.empty or stat not in df.columns:
        return float("nan"), 0, 0
    sub = df[(df["player_id"] == player_id) & df[stat].notna()]
    if sub.empty:
        return float("nan"), 0, 0
    sub = sub.sort_values(["season", "week"], ascending=[False, False])
    if lookback_games:
        sub = sub.head(int(lookback_games))
    vals = sub[stat].astype(float)
    if len(vals) == 0:
        return float("nan"), 0, 0
    k = int((vals > float(threshold)).sum())
    n = len(vals)
    p = float(k) / float(n)
    return p, k, n


def rank_ladder(player_id: str, stat: str,
                ladder_rungs: list[tuple[float, str, int]],
                lookback_games: int | None = None
                ) -> pd.DataFrame:
    """Score each rung in a ladder and sort by EV.

    ladder_rungs: list of (threshold, "over"|"under", american_odds).

    Each row also carries a Wilson 95% CI on the model probability and
    a conservative `ev_low` computed at the lower CI bound — so callers
    can see "the bet might still be -EV at the lower bound" rather than
    treating the point estimate as deterministic.
    """
    rows: list[RungEV] = []
    for threshold, side, odds in ladder_rungs:
        p_over, k_over, n = p_over_threshold(player_id, stat, threshold,
                                               lookback_games)
        if np.isnan(p_over):
            continue
        if side.lower() == "over":
            p_model = p_over
            ci_lo, ci_hi = wilson_interval(k_over, n)
        else:
            p_model = 1.0 - p_over
            # For "under," successes are n - k_over
            ci_lo, ci_hi = wilson_interval(n - k_over, n)
        decimal = american_to_decimal(odds)
        p_imp = decimal_to_implied_prob(decimal)
        ev = expected_value(p_model, decimal)
        ev_low = expected_value(ci_lo, decimal)
        rows.append(RungEV(
            threshold=float(threshold),
            side=side.lower(),
            american_odds=int(odds),
            decimal_odds=decimal,
            p_model=p_model,
            p_model_ci_low=ci_lo,
            p_model_ci_high=ci_hi,
            p_implied=p_imp,
            edge=p_model - p_imp,
            ev=ev,
            ev_low=ev_low,
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

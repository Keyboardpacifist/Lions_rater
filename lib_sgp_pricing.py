"""SGP correlation pricing — Feature 5.2 upgrade.

Books price same-game parlays as if legs are independent. They aren't.
This module takes a user-built SGP (list of player-stat-threshold legs)
and computes:

  • independent probability — what the book assumes (product of marginals)
  • correlated probability  — empirical joint from same-team-same-game
                              historical data
  • implied vs. fair odds   — if the user pastes the book's parlay odds
  • lift / EV gap           — where the edge lives

For 2-leg parlays involving a QB and a teammate, this is the natural
upgrade to the precomputed correlations table. For any combination,
it falls back to bootstrapping from the empirical joint distribution.

Public entry points
-------------------
    Leg(player_id, stat, threshold, side)
    sgp_price(legs, lookback_games=None)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lib_alt_line_ev import (
    american_to_decimal,
    decimal_to_implied_prob,
    p_over_threshold,
    wilson_interval,
)


REPO = Path(__file__).resolve().parent
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"


@st.cache_data(show_spinner=False)
def _load_stats() -> pd.DataFrame:
    return pd.read_parquet(PLAYER_STATS)


@dataclass
class Leg:
    player_id: str
    player_display_name: str  # for readability only
    stat: str                 # column name in player_stats_weekly
    threshold: float
    side: str                 # "over" / "under"


@dataclass
class SGPPriceResult:
    n_legs: int
    n_games_joint: int          # historical games where ALL players played
    p_independent: float        # product of marginals
    p_correlated: float         # empirical joint
    p_correlated_ci_low: float  # Wilson 95% CI lower bound on joint
    p_correlated_ci_high: float # Wilson 95% CI upper bound on joint
    correlation_lift: float     # p_correlated - p_independent
    fair_decimal_independent: float
    fair_decimal_correlated: float
    fair_american_independent: int
    fair_american_correlated: int
    book_decimal: float | None
    book_american: int | None
    edge_vs_book: float | None  # p_correlated - book_implied
    ev_vs_book: float | None    # EV per 1 unit on book's price (point est)
    ev_vs_book_low: float | None  # EV at lower CI bound (conservative)


def _decimal_to_american(decimal: float) -> int:
    """Round-trip decimal odds → American odds."""
    if decimal <= 1:
        return 0
    if decimal >= 2:
        return int(round((decimal - 1) * 100))
    return int(round(-100 / (decimal - 1)))


def sgp_price(legs: list[Leg],
               lookback_games: int | None = None,
               book_american_odds: int | None = None
               ) -> SGPPriceResult:
    """Price a multi-leg same-game parlay using empirical joint
    distribution. Returns the comparison vs. independent assumption."""
    if len(legs) < 2:
        raise ValueError("SGP needs ≥2 legs")

    df = _load_stats()
    if df.empty:
        return SGPPriceResult(
            n_legs=len(legs), n_games_joint=0,
            p_independent=float("nan"), p_correlated=float("nan"),
            p_correlated_ci_low=float("nan"),
            p_correlated_ci_high=float("nan"),
            correlation_lift=float("nan"),
            fair_decimal_independent=float("nan"),
            fair_decimal_correlated=float("nan"),
            fair_american_independent=0, fair_american_correlated=0,
            book_decimal=None, book_american=None,
            edge_vs_book=None, ev_vs_book=None, ev_vs_book_low=None,
        )

    # Marginals
    p_marginals: list[float] = []
    for leg in legs:
        p_over, _k, _n = p_over_threshold(leg.player_id, leg.stat,
                                            leg.threshold, lookback_games)
        if np.isnan(p_over):
            p_marginals.append(float("nan"))
            continue
        p = p_over if leg.side.lower() == "over" else (1.0 - p_over)
        p_marginals.append(p)

    if any(np.isnan(p) for p in p_marginals):
        p_independent = float("nan")
    else:
        p_independent = float(np.prod(p_marginals))

    # Joint — find games where all legs' players played, evaluate each
    # game against all leg conditions, count joint successes.
    per_leg_games = []
    for leg in legs:
        sub = df[(df["player_id"] == leg.player_id)
                 & df[leg.stat].notna()][["season", "week", leg.stat]]
        sub = sub.rename(columns={leg.stat: f"v_{id(leg)}"})
        per_leg_games.append((leg, sub))

    # Inner-join all leg dataframes on (season, week)
    joint = per_leg_games[0][1]
    for leg, sub in per_leg_games[1:]:
        joint = joint.merge(sub, on=["season", "week"], how="inner")
    joint = joint.sort_values(["season", "week"], ascending=[False, False])
    if lookback_games:
        joint = joint.head(int(lookback_games))

    if joint.empty:
        p_correlated = float("nan")
        n_joint = 0
        k_joint = 0
    else:
        n_joint = len(joint)
        # Evaluate each leg condition
        success = pd.Series(True, index=joint.index)
        for leg, _ in per_leg_games:
            col = f"v_{id(leg)}"
            if leg.side.lower() == "over":
                success &= (joint[col] > leg.threshold)
            else:
                success &= (joint[col] < leg.threshold)
        k_joint = int(success.sum())
        p_correlated = float(k_joint) / float(n_joint)

    # Wilson 95% CI on the joint probability
    if n_joint > 0:
        ci_low, ci_high = wilson_interval(k_joint, n_joint)
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    fair_decimal_indep = (1.0 / p_independent
                            if p_independent > 0 else float("nan"))
    fair_decimal_corr = (1.0 / p_correlated
                           if p_correlated > 0 else float("nan"))

    book_decimal: float | None = None
    book_american: int | None = None
    edge: float | None = None
    ev: float | None = None
    ev_low: float | None = None
    if book_american_odds is not None:
        book_american = int(book_american_odds)
        book_decimal = american_to_decimal(book_american_odds)
        if not (np.isnan(p_correlated)) and book_decimal is not None:
            book_implied = decimal_to_implied_prob(book_decimal)
            edge = float(p_correlated - book_implied)
            ev = float(p_correlated * book_decimal - 1.0)
            # Conservative EV at lower CI bound — if this is still
            # positive, the bet is +EV with high confidence
            ev_low = float(ci_low * book_decimal - 1.0) if not np.isnan(ci_low) else None

    return SGPPriceResult(
        n_legs=len(legs),
        n_games_joint=n_joint,
        p_independent=p_independent,
        p_correlated=p_correlated,
        p_correlated_ci_low=ci_low,
        p_correlated_ci_high=ci_high,
        correlation_lift=(p_correlated - p_independent
                            if not np.isnan(p_correlated) and not np.isnan(p_independent)
                            else float("nan")),
        fair_decimal_independent=fair_decimal_indep,
        fair_decimal_correlated=fair_decimal_corr,
        fair_american_independent=(_decimal_to_american(fair_decimal_indep)
                                     if not np.isnan(fair_decimal_indep) else 0),
        fair_american_correlated=(_decimal_to_american(fair_decimal_corr)
                                    if not np.isnan(fair_decimal_corr) else 0),
        book_decimal=book_decimal,
        book_american=book_american,
        edge_vs_book=edge,
        ev_vs_book=ev,
        ev_vs_book_low=ev_low,
    )

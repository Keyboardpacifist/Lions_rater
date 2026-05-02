"""Smart Parlay Builder — Feature 5.4.

Combines:
  • 5.3 Alt-Line EV — per-leg model probability
  • 5.2 SGP Pricing — correlated joint probability
  • 5.1 Decomposed Projection — for the daily mispriced-prop menu

Public entry points
-------------------
    score_parlay(legs, book_odds=None)
        → ParlayScoreResult
    suggest_improvements(legs, book_odds, candidate_pool)
        → ranked list of single-leg swaps that increase EV
    daily_menu(season, week, position=None, top_n=20)
        → ranked candidate legs sourced from divergence flags + recent
          form (placeholder for the production menu)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lib_alt_line_ev import (
    american_to_decimal,
    decimal_to_implied_prob,
    p_over_threshold,
)
from lib_sgp_pricing import Leg, sgp_price


@dataclass
class ParlayScoreResult:
    n_legs: int
    leg_marginals: list[tuple[str, float]]   # (label, p_model)
    p_independent: float
    p_correlated: float
    p_correlated_ci_low: float
    p_correlated_ci_high: float
    correlation_lift: float
    n_games_joint: int
    fair_decimal_correlated: float
    fair_american_correlated: int
    book_decimal: float | None
    book_american: int | None
    edge_vs_book: float | None
    ev_vs_book: float | None
    ev_vs_book_low: float | None
    verdict: str


def _verdict_from_ev(ev: float | None,
                       ev_low: float | None = None) -> str:
    """EV-based verdict. When `ev_low` (lower-CI EV) is also provided
    AND the lower bound is still positive, that's a high-confidence
    +EV bet. When the point EV is positive but the lower bound is
    negative, downgrade the conviction."""
    if ev is None:
        return "—"
    if ev >= 0.10:
        if ev_low is not None and ev_low >= 0:
            return "STRONG +EV — high-confidence +EV at lower CI bound"
        return "STRONG +EV (point estimate) — but uncertainty wide"
    if ev >= 0.03:
        if ev_low is not None and ev_low >= 0:
            return "Positive EV — confidence interval still positive"
        return "Positive EV (point estimate) — sample noise possible"
    if ev >= -0.03:
        return "Roughly fair — pass unless price improves"
    if ev >= -0.15:
        return "Negative EV — skip"
    return "Strong negative EV — avoid"


def score_parlay(legs: list[Leg],
                 book_odds: int | None = None,
                 lookback_games: int | None = None
                 ) -> ParlayScoreResult:
    """Score a multi-leg parlay. Returns marginals + joint comparison."""
    res = sgp_price(legs, lookback_games=lookback_games,
                    book_american_odds=book_odds)

    # Per-leg marginal probabilities (for the audit trail)
    marginals: list[tuple[str, float]] = []
    for leg in legs:
        p_over, _k, _n = p_over_threshold(leg.player_id, leg.stat,
                                            leg.threshold, lookback_games)
        if leg.side.lower() == "over":
            p = p_over
        else:
            p = (1.0 - p_over) if not np.isnan(p_over) else float("nan")
        label = (f"{leg.player_display_name} "
                 f"{leg.side} {leg.threshold} {leg.stat}")
        marginals.append((label, float(p)))

    return ParlayScoreResult(
        n_legs=res.n_legs,
        leg_marginals=marginals,
        p_independent=res.p_independent,
        p_correlated=res.p_correlated,
        p_correlated_ci_low=res.p_correlated_ci_low,
        p_correlated_ci_high=res.p_correlated_ci_high,
        correlation_lift=res.correlation_lift,
        n_games_joint=res.n_games_joint,
        fair_decimal_correlated=res.fair_decimal_correlated,
        fair_american_correlated=res.fair_american_correlated,
        book_decimal=res.book_decimal,
        book_american=res.book_american,
        edge_vs_book=res.edge_vs_book,
        ev_vs_book=res.ev_vs_book,
        ev_vs_book_low=res.ev_vs_book_low,
        verdict=_verdict_from_ev(res.ev_vs_book, res.ev_vs_book_low),
    )


def suggest_improvements(legs: list[Leg],
                          candidate_legs: list[Leg],
                          book_odds_per_leg: dict[int, int],
                          lookback_games: int | None = None,
                          top_n: int = 5) -> pd.DataFrame:
    """For each candidate leg, compute the parlay EV gained by SWAPPING
    it in for one of the existing legs. Returns top_n swaps ranked by
    EV improvement.

    `book_odds_per_leg` is a {leg_index: american_odds} mapping for the
    *original* parlay legs; new candidates are scored at -110 by
    default (caller can overwrite).
    """
    base_score = score_parlay(legs, lookback_games=lookback_games)
    base_ev = base_score.ev_vs_book if base_score.ev_vs_book is not None else float("-inf")

    suggestions = []
    for swap_idx in range(len(legs)):
        for cand in candidate_legs:
            if cand.player_id == legs[swap_idx].player_id:
                continue
            new_legs = list(legs)
            new_legs[swap_idx] = cand
            new_score = score_parlay(new_legs, lookback_games=lookback_games)
            new_ev = (new_score.ev_vs_book
                       if new_score.ev_vs_book is not None
                       else float("-inf"))
            suggestions.append({
                "drop_leg": (f"{legs[swap_idx].player_display_name} "
                              f"{legs[swap_idx].side} "
                              f"{legs[swap_idx].threshold} "
                              f"{legs[swap_idx].stat}"),
                "add_leg":  (f"{cand.player_display_name} "
                              f"{cand.side} {cand.threshold} {cand.stat}"),
                "ev_before": base_ev if base_ev != float("-inf") else float("nan"),
                "ev_after": new_ev if new_ev != float("-inf") else float("nan"),
                "ev_delta": ((new_ev - base_ev)
                              if new_ev != float("-inf") and base_ev != float("-inf")
                              else float("nan")),
                "p_correlated_after": new_score.p_correlated,
            })

    if not suggestions:
        return pd.DataFrame()
    df = pd.DataFrame(suggestions)
    return (df.sort_values("ev_delta", ascending=False)
              .head(top_n)
              .reset_index(drop=True))


def detect_anti_correlated(legs: list[Leg],
                            lookback_games: int | None = None,
                            threshold: float = -0.15
                            ) -> list[tuple[int, int, float]]:
    """Return (i, j, lift) pairs whose pairwise lift is below `threshold`
    — these are the parlays where book independence is over-pricing
    your parlay relative to reality (anti-correlated legs).

    `threshold = -0.15` ~= the leg pair has 15% LESS joint probability
    than the independence assumption.
    """
    out: list[tuple[int, int, float]] = []
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            r = sgp_price([legs[i], legs[j]],
                           lookback_games=lookback_games)
            lift = r.correlation_lift
            if not np.isnan(lift) and lift <= threshold:
                out.append((i, j, lift))
    return out

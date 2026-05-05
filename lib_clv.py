"""Closing Line Value (CLV) — the gold-standard measure of betting skill.

WHAT IS CLV?
============

When you place a bet at time T, you lock in the odds offered at that
moment. Between T and game start, the line moves as new information
arrives (injuries, weather, sharp action). The "closing line" — the
final pre-game odds — is empirically the sharpest market estimate
because it has absorbed everything the market knows.

A bettor who consistently beats the closing line has demonstrably
priced the market more accurately than the market itself. CLV is the
ONLY fast-feedback measurement of betting skill — bet outcomes are
high-variance and a 100-bet sample tells you almost nothing about
ROI, but CLV across 100 bets gives a clear signal of skill within
weeks instead of years.

A bettor with consistent +2% CLV is approximately a 0.5-1% ROI bettor
long-term. Sportsbooks limit accounts that show +CLV.

WHAT THIS LIB DOES
==================

Two flavors of CLV, both computable from this lib:

1. **TRUE CLV (book-vs-book)** — the standard. Requires the closing
   line as a separate input. Today the closing line comes from
   OddsAPI (already budgeted, not yet wired). When wired, every
   placed bet gets snapped against the closing line at game start.

2. **MODEL self-CLV** — bridge while OddsAPI isn't wired. Compares
   our own model's probability at bet time to its probability at
   game start. Doesn't measure sharpness vs the market, but does
   measure: "my model became more confident in this bet's
   direction" — useful in backtests of ranked findings.

KEY METRICS RETURNED
====================
- `clv_prob_points`  — (closing_fair_implied − bet_fair_implied) in
  the bet's direction. POSITIVE = beat the close. Mean across
  bets ≈ +0.02 (2 percentage points) is institutional-grade.
- `clv_pct_payout`   — (bet_decimal − closing_decimal) / closing_decimal.
  Useful for dollar-translation: "got 5% more upside than the
  market closed at."
- `beats_close`      — boolean shortcut: did this bet beat the close?

Public entry points
-------------------
    compute_clv(bet, closing_decimal=None, closing_other_side=None,
                model_prob_at_close=None) -> CLVRecord
    aggregate_clv(records) -> CLVSummary
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

from lib_alt_line_ev import (
    american_to_decimal,
    decimal_to_implied_prob,
    vig_free_implied,
)


@dataclass
class CLVRecord:
    """Per-bet CLV snapshot. Populated incrementally — `closing_*`
    fields are None until game start; `clv_*` fields populated only
    once closing data is in."""

    bet_id: str
    placed_at: datetime
    side: str                       # "over" / "under" / "home" / "away" etc.

    # User's bet — locked at placement
    bet_decimal_odds: float
    bet_american_odds: int
    bet_implied_raw: float          # 1 / decimal (vig included)
    bet_implied_fair: float         # vig-removed at placement
    model_prob_at_placement: float

    # Closing snapshot (populated at game start)
    closing_at: datetime | None = None
    closing_decimal_odds: float | None = None
    closing_american_odds: int | None = None
    closing_implied_raw: float | None = None
    closing_implied_fair: float | None = None
    model_prob_at_close: float | None = None

    # Computed metrics (populated when closing data is available)
    clv_prob_points: float | None = None    # primary CLV metric
    clv_pct_payout: float | None = None     # secondary
    model_self_clv: float | None = None     # model-only CLV
    beats_close: bool | None = None


def compute_clv(*, bet_id: str,
                placed_at: datetime,
                side: str,
                bet_american_odds: int,
                bet_other_side_american_odds: int | None = None,
                model_prob_at_placement: float,
                closing_american_odds: int | None = None,
                closing_other_side_american_odds: int | None = None,
                model_prob_at_close: float | None = None,
                closing_at: datetime | None = None,
                ) -> CLVRecord:
    """Compute CLV for a single bet. Closing data is optional —
    when omitted, only the placement snapshot is filled in (call
    again later once the close is known)."""

    bet_decimal = american_to_decimal(bet_american_odds)
    bet_implied_raw = decimal_to_implied_prob(bet_decimal)

    # Vig-free fair implied at placement
    if side.lower() == "over" or side.lower() == "home":
        p_o_fair, _ = vig_free_implied(
            bet_american_odds, bet_other_side_american_odds)
        bet_implied_fair = p_o_fair
    else:
        _, p_u_fair = vig_free_implied(
            bet_other_side_american_odds, bet_american_odds)
        bet_implied_fair = p_u_fair

    rec = CLVRecord(
        bet_id=bet_id,
        placed_at=placed_at,
        side=side,
        bet_decimal_odds=bet_decimal,
        bet_american_odds=int(bet_american_odds),
        bet_implied_raw=bet_implied_raw,
        bet_implied_fair=bet_implied_fair,
        model_prob_at_placement=float(model_prob_at_placement),
    )

    # If closing data is missing, return the partial record
    if closing_american_odds is None:
        return rec

    closing_decimal = american_to_decimal(closing_american_odds)
    closing_implied_raw = decimal_to_implied_prob(closing_decimal)
    if side.lower() == "over" or side.lower() == "home":
        p_o_fair_c, _ = vig_free_implied(
            closing_american_odds, closing_other_side_american_odds)
        closing_implied_fair = p_o_fair_c
    else:
        _, p_u_fair_c = vig_free_implied(
            closing_other_side_american_odds, closing_american_odds)
        closing_implied_fair = p_u_fair_c

    rec.closing_at = closing_at
    rec.closing_decimal_odds = closing_decimal
    rec.closing_american_odds = int(closing_american_odds)
    rec.closing_implied_raw = closing_implied_raw
    rec.closing_implied_fair = closing_implied_fair

    # Primary CLV: in the bet's direction, how many percentage points
    # did the closing FAIR implied probability exceed our bet's FAIR
    # implied probability? POSITIVE = market thinks the bet is more
    # likely than the price we got. We beat the close.
    rec.clv_prob_points = closing_implied_fair - bet_implied_fair

    # Secondary: dollar-payout-style CLV.
    # If our locked decimal is HIGHER than the closing decimal, we got
    # a better payout than the closer — positive CLV.
    rec.clv_pct_payout = (bet_decimal - closing_decimal) / closing_decimal

    rec.beats_close = rec.clv_prob_points > 0

    if model_prob_at_close is not None:
        rec.model_prob_at_close = float(model_prob_at_close)
        rec.model_self_clv = (model_prob_at_close
                                - model_prob_at_placement)

    return rec


@dataclass
class CLVSummary:
    n_bets: int
    n_with_closing: int
    n_beat_close: int
    pct_beat_close: float            # n_beat_close / n_with_closing
    mean_clv_prob_points: float      # mean CLV in pp
    median_clv_prob_points: float
    mean_clv_pct_payout: float
    by_finding_type: dict[str, float] = field(default_factory=dict)


def aggregate_clv(records: Iterable[CLVRecord],
                    finding_type_lookup: dict[str, str] | None = None
                    ) -> CLVSummary:
    """Aggregate a sequence of CLVRecords into a summary.

    `finding_type_lookup` is an optional `{bet_id: finding_type}`
    mapping — when provided, the summary breaks CLV down by type.
    """
    rec_list = list(records)
    n = len(rec_list)
    closed = [r for r in rec_list if r.clv_prob_points is not None]
    n_closed = len(closed)
    if n_closed == 0:
        return CLVSummary(
            n_bets=n, n_with_closing=0, n_beat_close=0,
            pct_beat_close=0.0,
            mean_clv_prob_points=0.0,
            median_clv_prob_points=0.0,
            mean_clv_pct_payout=0.0,
            by_finding_type={},
        )

    pp = sorted(r.clv_prob_points for r in closed)
    mid = pp[n_closed // 2] if n_closed % 2 else (
        (pp[n_closed // 2 - 1] + pp[n_closed // 2]) / 2)
    n_beat = sum(1 for r in closed if r.beats_close)

    summary = CLVSummary(
        n_bets=n,
        n_with_closing=n_closed,
        n_beat_close=n_beat,
        pct_beat_close=n_beat / n_closed,
        mean_clv_prob_points=sum(pp) / n_closed,
        median_clv_prob_points=mid,
        mean_clv_pct_payout=(
            sum(r.clv_pct_payout for r in closed) / n_closed),
    )

    if finding_type_lookup:
        by_type: dict[str, list[float]] = {}
        for r in closed:
            ft = finding_type_lookup.get(r.bet_id)
            if not ft:
                continue
            by_type.setdefault(ft, []).append(r.clv_prob_points)
        summary.by_finding_type = {
            ft: sum(vals) / len(vals)
            for ft, vals in by_type.items()
        }

    return summary


__all__ = [
    "CLVRecord",
    "CLVSummary",
    "compute_clv",
    "aggregate_clv",
]

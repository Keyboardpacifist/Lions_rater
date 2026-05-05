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
    """RAW implied probability of a decimal-odds price (vig included).

    NOTE: this is the price-implied probability — it does NOT remove the
    bookmaker's juice. For a -110/-110 prop both sides return 52.4% here,
    which sums to 104.8% (the vig). To compute the book's TRUE estimate,
    use `vig_free_implied()` with both sides' odds.

    Use this function only when you specifically want the price-implied
    figure (e.g., for EV calculation, which uses the offered price
    directly). For "how sharp is my model vs the book?" comparisons,
    use vig_free_implied().
    """
    return 1.0 / decimal if decimal > 0 else float("nan")


# Industry-standard prop-market vig when we only see one side of the
# market. Books typically run 4-5% vig on player props (-110/-110 ≈
# 4.8%; -115/-105 ≈ 4.5%). 4.5% is a defensible default.
DEFAULT_PROP_VIG = 0.045


def vig_free_implied(american_over: int | float | None,
                       american_under: int | float | None = None,
                       *, default_vig: float = DEFAULT_PROP_VIG,
                       ) -> tuple[float, float]:
    """Return the vig-removed (fair) probabilities for both sides of a
    two-way market. THIS is the book's true estimate of probability.

    Two paths:

    1. **Both sides known** — devig by dividing each raw implied by
       the sum (the standard "proportional devig" method, equivalent
       to assuming the book sets each side proportional to its true
       probability and then marks both up by the same multiplicative
       factor).

    2. **Only one side known** — deduct `default_vig / 2` from the raw
       implied probability of the known side. Less accurate than
       (1) but workable when callers only have one quote.

    Returns (p_over_fair, p_under_fair). When only one side is
    provided, the unknown side is computed as 1 - p_known_fair.
    """
    if american_over is None and american_under is None:
        return float("nan"), float("nan")

    if american_over is not None and american_under is not None:
        # Both sides — proportional devig
        dec_o = american_to_decimal(american_over)
        dec_u = american_to_decimal(american_under)
        p_o_raw = 1.0 / dec_o
        p_u_raw = 1.0 / dec_u
        total = p_o_raw + p_u_raw
        if total <= 0:
            return float("nan"), float("nan")
        return p_o_raw / total, p_u_raw / total

    # Only one side — single-side approximation
    side, opposite_label = (
        (american_over, "under") if american_over is not None
        else (american_under, "over")
    )
    dec = american_to_decimal(side)
    p_raw = 1.0 / dec
    p_fair = max(0.0, min(1.0, p_raw - default_vig / 2.0))
    p_other_fair = 1.0 - p_fair
    if american_over is not None:
        return p_fair, p_other_fair
    return p_other_fair, p_fair


def expected_value(p_win: float, decimal_odds: float) -> float:
    """EV per 1 unit risked. Positive = +EV bet.

    Note: EV uses the OFFERED decimal odds (not the fair / vig-free
    odds). The user is betting the offered price, so the offered price
    is what determines the payout. EV is correctly computed without
    vig adjustment — vig is already baked into the decimal odds."""
    return p_win * decimal_odds - 1.0


def beta_shrink(k: int, n: int,
                  alpha: float = 2.0, beta: float = 2.0) -> float:
    """Beta-Binomial posterior mean — shrinks raw k/n toward a prior.

    With alpha=beta=2 (the default), the prior is "2 imaginary games
    that came in at 50%". This is mild but does the right thing:
      • n=3, k=3 (raw 100%) → shrinks to 5/7 = 71% (sensible)
      • n=8, k=6 (raw 75%) → shrinks to 8/12 = 67%
      • n=50, k=37 (raw 74%) → shrinks to 39/54 = 72% (barely moves)

    Why this matters for prop betting: empirical hit rates from
    small samples are extremely high-variance. A WR who went over
    his target line in 3 of his last 5 games has empirical p=0.60,
    but the right Bayesian estimate is closer to 50% — and the
    book's price reflects that. Without shrinkage, hot-streak
    players get systematically over-projected.

    Adjust alpha/beta if you have a stronger prior (e.g., alpha=3,
    beta=2 for an "expected slightly favored" line).
    """
    if n < 0 or alpha <= 0 or beta <= 0:
        return float("nan")
    return (k + alpha) / (n + alpha + beta)


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
    p_model: float          # Beta-shrunk posterior — preferred for display
    p_model_raw: float      # Raw empirical k/n (legacy, high-variance)
    p_model_ci_low: float   # Wilson 95% CI lower bound (on raw)
    p_model_ci_high: float  # Wilson 95% CI upper bound (on raw)
    p_implied: float        # RAW (vig-included) implied — for reference
    p_implied_fair: float   # vig-removed implied — book's TRUE estimate
    edge: float             # p_model_shrunk - p_implied_fair (real edge)
    edge_raw: float         # p_model_raw - p_implied (legacy, biased)
    ev: float           # EV per unit risked at shrunk p_model
    ev_raw: float       # EV per unit risked at raw p_model
    ev_low: float       # EV at lower CI bound (conservative)
    n_games: int


def p_over_threshold(player_id: str, stat: str, threshold: float,
                      lookback_games: int | None = None
                      ) -> tuple[float, int, int]:
    """Empirical probability that the player's stat exceeds threshold
    in the next game. Returns (p_model, k_successes, n_decided)
    so callers can compute their own confidence intervals.

    PUSHES (stat exactly equal to threshold) are EXCLUDED from the
    denominator — real markets refund pushes, so they don't decide
    a winner or loser. The probability returned is conditional on
    "the bet is decided" (i.e., not a push). Without this exclusion
    a whole-number line like "carries OVER 12" would systematically
    under-estimate P(over) every time the player carries exactly 12.

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
    thr = float(threshold)
    # Push detection — exact equality. Practical tolerance for
    # floating-point: 1e-9 (yardage stats are integer-valued).
    push_mask = (vals - thr).abs() < 1e-9
    k = int((vals > thr).sum())
    n_decided = int(len(vals) - push_mask.sum())
    if n_decided <= 0:
        return float("nan"), 0, 0
    p = float(k) / float(n_decided)
    return p, k, n_decided


def rank_ladder(player_id: str, stat: str,
                ladder_rungs,
                lookback_games: int | None = None
                ) -> pd.DataFrame:
    """Score each rung in a ladder and sort by EV.

    ladder_rungs: each rung is one of:
      • (threshold, "over"|"under", american_odds)
      • (threshold, "over"|"under", american_odds, paired_other_side_odds)
        — preferred: lets us compute true vig-free fair value

    `edge` is reported in two flavors:
      • `edge` = p_model − p_implied_fair (vig-removed, the real edge)
      • `edge_raw` = p_model − p_implied_raw (legacy, biased toward 0)

    Each row also carries a Wilson 95% CI on the model probability and
    a conservative `ev_low` computed at the lower CI bound.
    """
    rows: list[RungEV] = []
    for rung in ladder_rungs:
        # Backward-compatible 3-tuple; new 4-tuple includes other side
        if len(rung) == 4:
            threshold, side, odds, other_side_odds = rung
        else:
            threshold, side, odds = rung
            other_side_odds = None

        p_over, k_over, n = p_over_threshold(player_id, stat, threshold,
                                               lookback_games)
        if np.isnan(p_over):
            continue
        if side.lower() == "over":
            k_side = k_over
            p_raw = p_over
            ci_lo, ci_hi = wilson_interval(k_over, n)
        else:
            k_side = n - k_over
            p_raw = 1.0 - p_over
            ci_lo, ci_hi = wilson_interval(n - k_over, n)

        # Beta-shrunk posterior — preferred over raw for display
        p_shrunk = beta_shrink(k_side, n)
        decimal = american_to_decimal(odds)
        p_imp_raw = decimal_to_implied_prob(decimal)

        # Vig-free fair implied — the book's TRUE estimate
        if side.lower() == "over":
            p_o_fair, p_u_fair = vig_free_implied(odds, other_side_odds)
            p_imp_fair = p_o_fair
        else:
            p_o_fair, p_u_fair = vig_free_implied(other_side_odds, odds)
            p_imp_fair = p_u_fair

        ev = expected_value(p_shrunk, decimal)
        ev_raw = expected_value(p_raw, decimal)
        ev_low = expected_value(ci_lo, decimal)
        rows.append(RungEV(
            threshold=float(threshold),
            side=side.lower(),
            american_odds=int(odds),
            decimal_odds=decimal,
            p_model=p_shrunk,
            p_model_raw=p_raw,
            p_model_ci_low=ci_lo,
            p_model_ci_high=ci_hi,
            p_implied=p_imp_raw,
            p_implied_fair=p_imp_fair,
            edge=p_shrunk - p_imp_fair,
            edge_raw=p_raw - p_imp_raw,
            ev=ev,
            ev_raw=ev_raw,
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

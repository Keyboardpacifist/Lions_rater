"""Fantasy scoring engine.

Converts a player's stat row into fantasy points under any rule set.
Same engine drives weekly leaderboards, per-game projections, and
custom-league configs once those land. Designed to be data-source
agnostic: pass any row/DataFrame with stat columns and we'll score it.

Pre-built configs in the MVP
----------------------------
    PPR_CONFIG          — reception = 1.0
    HALF_PPR_CONFIG     — reception = 0.5
    STANDARD_CONFIG     — reception = 0.0  (no-PPR, "old-school")
    TE_PREMIUM_CONFIG   — reception = 1.0 + 0.5 bonus for TEs

Fast-follow configs (drop-in additions, ~10 minutes each):
    SUPERFLEX           — pass_td = 6.0 (or 8.0 in some leagues)
    DRAFTKINGS_DFS      — DFS-specific (3pt 100yd bonus, etc.)
    FANDUEL_DFS

v1 limitations
--------------
- 2-point conversions and fumbles-lost columns aren't in the weekly
  file we use as the primary source; the pre-built configs include
  the multipliers but they default to 0 unless you pass those columns
  explicitly. fantasy_points / fantasy_points_ppr in the source already
  include those events, so v1 leaderboards match public sources.
- DST and Kicker scoring not in MVP. (Build alongside K and D/ST when
  we add those positions to the fantasy page.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class ScoringConfig:
    """Multiplier table — applied per stat per row.

    Every field is the FANTASY POINTS contributed by ONE unit of that
    stat. So pass_yard = 0.04 means "1 fantasy point per 25 passing yards".
    """
    name: str
    # ── Passing ──
    pass_yard: float = 0.04   # 1 pt / 25 yards
    pass_td: float = 4.0
    pass_int: float = -2.0
    pass_2pt: float = 2.0
    # ── Rushing ──
    rush_yard: float = 0.1    # 1 pt / 10 yards
    rush_td: float = 6.0
    rush_2pt: float = 2.0
    # ── Receiving ──
    reception: float = 1.0    # the PPR knob
    rec_yard: float = 0.1
    rec_td: float = 6.0
    rec_2pt: float = 2.0
    # ── TE bonus reception (TE Premium) ──
    te_premium_bonus: float = 0.0
    # ── Misc ──
    fumble_lost: float = -2.0
    return_td: float = 6.0


# ── Pre-built configs ────────────────────────────────────────────

PPR_CONFIG = ScoringConfig(
    name="PPR",
    reception=1.0,
)

HALF_PPR_CONFIG = ScoringConfig(
    name="Half-PPR",
    reception=0.5,
)

STANDARD_CONFIG = ScoringConfig(
    name="Standard",
    reception=0.0,
)

TE_PREMIUM_CONFIG = ScoringConfig(
    name="TE Premium",
    reception=1.0,
    te_premium_bonus=0.5,
)

ALL_CONFIGS = [
    PPR_CONFIG,
    HALF_PPR_CONFIG,
    STANDARD_CONFIG,
    TE_PREMIUM_CONFIG,
]
CONFIG_BY_NAME = {c.name: c for c in ALL_CONFIGS}


# ── Scoring functions ───────────────────────────────────────────

def _g(stat_row, col: str) -> float:
    """Safe getter — returns 0.0 for missing or NaN."""
    if hasattr(stat_row, "get"):
        v = stat_row.get(col, 0.0)
    else:
        v = getattr(stat_row, col, 0.0)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    return float(v)


def score_stats(stat_row, config: ScoringConfig,
                  position: Optional[str] = None) -> float:
    """Compute fantasy points for one player-row given a scoring config.

    `stat_row` may be a dict, pd.Series, or any object with .get / attrs.
    `position` is required for TE Premium bonus to apply.
    """
    pts = 0.0

    # Passing
    pts += _g(stat_row, "passing_yards") * config.pass_yard
    pts += _g(stat_row, "passing_tds") * config.pass_td
    pts += _g(stat_row, "passing_interceptions") * config.pass_int
    pts += _g(stat_row, "passing_2pt_conversions") * config.pass_2pt

    # Rushing
    pts += _g(stat_row, "rushing_yards") * config.rush_yard
    pts += _g(stat_row, "rushing_tds") * config.rush_td
    pts += _g(stat_row, "rushing_2pt_conversions") * config.rush_2pt

    # Receiving (with TE Premium bonus)
    rec_value = config.reception
    if position == "TE" and config.te_premium_bonus > 0:
        rec_value += config.te_premium_bonus
    pts += _g(stat_row, "receptions") * rec_value
    pts += _g(stat_row, "receiving_yards") * config.rec_yard
    pts += _g(stat_row, "receiving_tds") * config.rec_td
    pts += _g(stat_row, "receiving_2pt_conversions") * config.rec_2pt

    # Fumbles (sum across categories — not all sources split them)
    fumbles = (
        _g(stat_row, "rushing_fumbles_lost")
        + _g(stat_row, "receiving_fumbles_lost")
        + _g(stat_row, "sack_fumbles_lost")
    )
    pts += fumbles * config.fumble_lost

    return pts


def score_dataframe(df: pd.DataFrame, config: ScoringConfig,
                       position_col: str = "position") -> pd.Series:
    """Vectorized scoring — returns one fantasy-point value per row."""

    def _safe(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        return df[col].fillna(0.0).astype(float)

    pts = pd.Series(0.0, index=df.index)
    pts += _safe("passing_yards") * config.pass_yard
    pts += _safe("passing_tds") * config.pass_td
    pts += _safe("passing_interceptions") * config.pass_int
    pts += _safe("passing_2pt_conversions") * config.pass_2pt

    pts += _safe("rushing_yards") * config.rush_yard
    pts += _safe("rushing_tds") * config.rush_td
    pts += _safe("rushing_2pt_conversions") * config.rush_2pt

    rec_pts = _safe("receptions") * config.reception
    if config.te_premium_bonus > 0 and position_col in df.columns:
        is_te = (df[position_col] == "TE").astype(float)
        rec_pts = rec_pts + (is_te * _safe("receptions")
                              * config.te_premium_bonus)
    pts += rec_pts

    pts += _safe("receiving_yards") * config.rec_yard
    pts += _safe("receiving_tds") * config.rec_td
    pts += _safe("receiving_2pt_conversions") * config.rec_2pt

    fumbles = (_safe("rushing_fumbles_lost")
                + _safe("receiving_fumbles_lost")
                + _safe("sack_fumbles_lost"))
    pts += fumbles * config.fumble_lost

    return pts

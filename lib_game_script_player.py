"""Per-player game-script splits.

For each player, classify every historical game by the FINAL margin
(team_score - opponent_score) into 5 buckets and compute their
average stat in each bucket. The multiplier for a target game-script
is `bucket_avg / overall_avg`.

Why this matters
----------------
Usage shifts dramatically with game flow. A blowout victory means
the RB1 sees +30% carries (clock kill) while the WR1 sees -25%
targets (running out the clock). A blowout loss flips it. The
model needs this so projections don't sit fixed regardless of
expected flow.

Buckets
-------
    BLOWOUT_LOSS    — margin ≤ -15  (team lost by 15+)
    MODERATE_TRAIL  — margin in [-14, -8]
    CLOSE           — margin in [-7, +7]
    MODERATE_LEAD   — margin in [+8, +14]
    BLOWOUT_WIN     — margin ≥ +15

Public entry points
-------------------
    GameScriptBucket (str enum)
    classify_margin(margin) -> bucket
    player_game_script_splits(player_id, stat, lookback_games=None)
        → dict of {bucket: (mean, n)}
    multiplier_for_game_script(player_id, stat, target_bucket,
                                  fallback_position=None)
        → float (1.0 = no shift)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


REPO = Path(__file__).resolve().parent
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
SCHEDULES = REPO / "data" / "nfl_schedules.parquet"


class GameScriptBucket(str, Enum):
    BLOWOUT_LOSS = "BLOWOUT_LOSS"
    MODERATE_TRAIL = "MODERATE_TRAIL"
    CLOSE = "CLOSE"
    MODERATE_LEAD = "MODERATE_LEAD"
    BLOWOUT_WIN = "BLOWOUT_WIN"


# Friendly labels (for UI)
BUCKET_LABEL = {
    GameScriptBucket.BLOWOUT_LOSS:    "Blowout loss (-15+)",
    GameScriptBucket.MODERATE_TRAIL:  "Moderate trail (-8 to -14)",
    GameScriptBucket.CLOSE:           "Close game (within 7)",
    GameScriptBucket.MODERATE_LEAD:   "Moderate lead (+8 to +14)",
    GameScriptBucket.BLOWOUT_WIN:     "Blowout win (+15+)",
}

BUCKET_ORDER = [
    GameScriptBucket.BLOWOUT_LOSS,
    GameScriptBucket.MODERATE_TRAIL,
    GameScriptBucket.CLOSE,
    GameScriptBucket.MODERATE_LEAD,
    GameScriptBucket.BLOWOUT_WIN,
]


def classify_margin(margin: float) -> GameScriptBucket:
    if margin <= -15:
        return GameScriptBucket.BLOWOUT_LOSS
    if margin <= -8:
        return GameScriptBucket.MODERATE_TRAIL
    if margin <= 7:
        return GameScriptBucket.CLOSE
    if margin <= 14:
        return GameScriptBucket.MODERATE_LEAD
    return GameScriptBucket.BLOWOUT_WIN


def infer_bucket_from_spread(spread_pov: float | None,
                              total: float | None = None
                              ) -> GameScriptBucket:
    """Default expected bucket based on closing spread (from team's POV).
    Positive spread_pov = team is favored by that many points.
    +15 favorite → BLOWOUT_WIN; +7-14 → MODERATE_LEAD;
    +/-7 → CLOSE; -7-14 → MODERATE_TRAIL; -15+ → BLOWOUT_LOSS.
    Useful as a sane default when the user opens the dropdown."""
    if spread_pov is None:
        return GameScriptBucket.CLOSE
    if spread_pov >= 15:
        return GameScriptBucket.BLOWOUT_WIN
    if spread_pov >= 7:
        return GameScriptBucket.MODERATE_LEAD
    if spread_pov <= -15:
        return GameScriptBucket.BLOWOUT_LOSS
    if spread_pov <= -7:
        return GameScriptBucket.MODERATE_TRAIL
    return GameScriptBucket.CLOSE


@st.cache_data(show_spinner=False)
def _load_stats() -> pd.DataFrame:
    return pd.read_parquet(PLAYER_STATS)


@st.cache_data(show_spinner=False)
def _load_schedules() -> pd.DataFrame:
    return pd.read_parquet(SCHEDULES)


def _attach_team_margin(stats: pd.DataFrame) -> pd.DataFrame:
    """For each player-game row, compute the team's final margin
    (team_score - opp_score) using the schedules table."""
    sch = _load_schedules()
    home = sch[["season", "week", "home_team", "home_score",
                "away_score"]].copy()
    home = home.rename(columns={"home_team": "team"})
    home["team_margin"] = home["home_score"] - home["away_score"]
    home = home[["season", "week", "team", "team_margin"]]

    away = sch[["season", "week", "away_team", "home_score",
                "away_score"]].copy()
    away = away.rename(columns={"away_team": "team"})
    away["team_margin"] = away["away_score"] - away["home_score"]
    away = away[["season", "week", "team", "team_margin"]]

    margins = pd.concat([home, away], ignore_index=True)
    return stats.merge(margins, on=["season", "week", "team"], how="left")


@dataclass
class GameScriptSplit:
    bucket: GameScriptBucket
    n: int
    mean_stat: float
    multiplier: float       # mean_stat / overall_mean


def player_game_script_splits(player_id: str, stat: str,
                                lookback_games: int | None = None
                                ) -> dict[GameScriptBucket, GameScriptSplit]:
    """Return per-bucket splits for the player's primary stat."""
    stats = _load_stats()
    if stats.empty or stat not in stats.columns:
        return {}
    sub = stats[(stats["player_id"] == player_id)
                & stats[stat].notna()].copy()
    if sub.empty:
        return {}
    sub = _attach_team_margin(sub)
    sub = sub.dropna(subset=["team_margin"])
    if sub.empty:
        return {}
    sub = sub.sort_values(["season", "week"], ascending=[False, False])
    if lookback_games:
        sub = sub.head(int(lookback_games))

    overall_mean = float(sub[stat].astype(float).mean())
    if overall_mean <= 0:
        return {}

    sub["bucket"] = sub["team_margin"].apply(classify_margin)

    out: dict[GameScriptBucket, GameScriptSplit] = {}
    for bucket in BUCKET_ORDER:
        cell = sub[sub["bucket"] == bucket]
        n = len(cell)
        if n == 0:
            continue
        mean = float(cell[stat].astype(float).mean())
        out[bucket] = GameScriptSplit(
            bucket=bucket,
            n=n,
            mean_stat=mean,
            multiplier=(mean / overall_mean) if overall_mean > 0 else 1.0,
        )
    return out


def position_game_script_splits(position: str, stat: str
                                  ) -> dict[GameScriptBucket, GameScriptSplit]:
    """League-wide position-level fallback when a player's own sample
    is too thin in a bucket."""
    stats = _load_stats()
    if stats.empty or stat not in stats.columns:
        return {}
    sub = stats[(stats["position"] == position)
                & stats[stat].notna()].copy()
    if sub.empty:
        return {}
    sub = _attach_team_margin(sub)
    sub = sub.dropna(subset=["team_margin"])
    sub["bucket"] = sub["team_margin"].apply(classify_margin)
    overall = float(sub[stat].astype(float).mean())
    if overall <= 0:
        return {}
    out = {}
    for bucket in BUCKET_ORDER:
        cell = sub[sub["bucket"] == bucket]
        n = len(cell)
        if n == 0:
            continue
        mean = float(cell[stat].astype(float).mean())
        out[bucket] = GameScriptSplit(
            bucket=bucket, n=n, mean_stat=mean,
            multiplier=mean / overall,
        )
    return out


def multiplier_for_game_script(player_id: str, stat: str,
                                target_bucket: GameScriptBucket,
                                fallback_position: str | None = None,
                                min_player_n: int = 5
                                ) -> tuple[float, str, int]:
    """Return (multiplier, source, n) for a target game-script bucket.

    `source` is one of: 'player' (cohort matched), 'position' (league
    fallback), or 'none' (no data, multiplier=1.0).
    """
    splits = player_game_script_splits(player_id, stat)
    if target_bucket in splits and splits[target_bucket].n >= min_player_n:
        s = splits[target_bucket]
        return s.multiplier, "player", s.n

    if fallback_position:
        league = position_game_script_splits(fallback_position, stat)
        if target_bucket in league:
            s = league[target_bucket]
            return s.multiplier, "position", s.n

    return 1.0, "none", 0

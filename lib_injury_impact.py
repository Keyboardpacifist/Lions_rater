"""Accessor for player self-injury deltas + team starter-absence deltas.

Two tables, two lookups:

  • Player self-delta — when THIS specific player is on a given Friday
    designation (Q/Limited, etc.), how does HIS production differ from
    his own healthy baseline? Opp-adjusted.

  • Team starter-absence delta — when THIS specific starter is OUT,
    how do THIS team's points/game shift? Opp-adjusted.

Both serve `lib_decomposed_projection.py` and the matchup/player auto
reports. They sharpen the cohort engine: instead of a league-wide
"WRs with hamstrings retain 86%," the engine first tries the player's
own historical retention with that body part / status — only falling
back to the cohort number when the player's sample is thin.

Public entry points
-------------------
    lookup_player_self_delta(player_id, stat, bucket, min_n=5)
        → (retention, delta_adj, n, source) | None
    lookup_team_starter_absence(team, season, role, min_n=2)
        → (adj_pts_delta, raw_pts_delta, n_out, n_active, thin) | None
    bucket_for(report_status, practice_status)
        → bucket label matching the deltas table
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
PLAYER_SELF = DATA / "player_self_injury_deltas.parquet"
TEAM_ABSENCE = DATA / "team_starter_absence_deltas.parquet"


@st.cache_data(show_spinner=False)
def _load_player_self() -> pd.DataFrame:
    if not PLAYER_SELF.exists():
        return pd.DataFrame()
    return pd.read_parquet(PLAYER_SELF)


@st.cache_data(show_spinner=False)
def _load_team_absence() -> pd.DataFrame:
    if not TEAM_ABSENCE.exists():
        return pd.DataFrame()
    return pd.read_parquet(TEAM_ABSENCE)


# Bucket scheme (must match build_player_injury_self_deltas.py)
def bucket_for(report, practice) -> str:
    if report is None or (isinstance(report, float) and pd.isna(report)):
        return "HEALTHY"
    r = str(report).upper().strip()
    if r in ("", "NONE", "NAN"):
        return "HEALTHY"
    if practice is None or (isinstance(practice, float)
                              and pd.isna(practice)):
        p = "NONE"
    else:
        p = str(practice).upper().strip()
    if "DID NOT" in p or p == "DNP":
        p = "DNP"
    elif "LIMITED" in p:
        p = "LIMITED"
    elif "FULL" in p:
        p = "FULL"
    if r == "PROBABLE":
        return "PROBABLE_FULL" if p == "FULL" else "PROBABLE_LIMITED"
    if r == "QUESTIONABLE":
        return {"FULL": "QUESTIONABLE_FULL",
                "LIMITED": "QUESTIONABLE_LIMITED",
                "DNP": "QUESTIONABLE_DNP"}.get(p, "QUESTIONABLE_LIMITED")
    if r == "DOUBTFUL":
        return "DOUBTFUL_ANY"
    if r == "OUT":
        return "OUT_PLAYED"
    return "HEALTHY"


@dataclass
class PlayerSelfDeltaResult:
    retention_adj: float    # mean_raw / healthy_raw (clipped)
    delta_adj: float        # mean_adj - healthy_adj (yards)
    n: int                  # bucket sample size
    n_total: int            # player's overall sample
    source: str             # "player" / "none"


def lookup_player_self_delta(player_id: str, stat: str,
                               bucket: str,
                               min_n: int = 5
                               ) -> PlayerSelfDeltaResult | None:
    """Return the player's own retention multiplier for this bucket.
    None if sample is too thin or table not loaded."""
    df = _load_player_self()
    if df.empty:
        return None
    sub = df[(df["player_id"] == player_id)
             & (df["stat"] == stat)
             & (df["bucket"] == bucket)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    n = int(row["n"])
    if n < min_n:
        return None
    return PlayerSelfDeltaResult(
        retention_adj=float(row["retention_adj"]),
        delta_adj=float(row["delta_adj"]),
        n=n,
        n_total=int(row.get("n_total_player", 0)),
        source="player",
    )


@dataclass
class TeamAbsenceResult:
    adj_pts_delta: float    # opp-adjusted points/game when starter out
    raw_pts_delta: float    # raw points/game when starter out
    n_active: int
    n_out: int
    thin_sample: bool
    player_lost_name: str | None
    delta_ci_low: float
    delta_ci_high: float


def lookup_team_starter_absence(team: str, season: int, role: str,
                                  min_n_out: int = 1
                                  ) -> TeamAbsenceResult | None:
    """Return team-specific scoring shift when the named starter is OUT.
    `role` ∈ {QB1, RB1, WR1, TE1}."""
    df = _load_team_absence()
    if df.empty:
        return None
    sub = df[(df["team"] == team)
             & (df["season"] == int(season))
             & (df["role_lost"] == role)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    if int(row["n_out"]) < min_n_out:
        return None
    return TeamAbsenceResult(
        adj_pts_delta=float(row["adj_pts_delta"]),
        raw_pts_delta=float(row["raw_pts_delta"]),
        n_active=int(row["n_active"]),
        n_out=int(row["n_out"]),
        thin_sample=bool(row["thin_sample"]),
        player_lost_name=(str(row["player_lost_name"])
                            if pd.notna(row["player_lost_name"])
                            else None),
        delta_ci_low=float(row["delta_ci_low"]),
        delta_ci_high=float(row["delta_ci_high"]),
    )


def lookup_team_absence_history(team: str, role: str,
                                  recent_seasons: int = 5
                                  ) -> pd.DataFrame:
    """Multi-season history for THIS team + role (e.g., DET QB1 across
    last 5 years). Useful for context when current season has n_out=0."""
    df = _load_team_absence()
    if df.empty:
        return df
    most_recent = int(df["season"].max())
    cutoff = most_recent - recent_seasons + 1
    return df[(df["team"] == team)
              & (df["role_lost"] == role)
              & (df["season"] >= cutoff)
              ].sort_values("season", ascending=False).reset_index(drop=True)

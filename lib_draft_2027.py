"""Draft 2027 prospect-board engine.

Joins 2025-season production stats × recruiting-year × draft-status
to surface players draft-eligible for the 2027 NFL Draft, with a
composite z-score per prospect ready for the big-board UI.

Eligibility rules (v1):
  • Player has 2025-season stats in our position parquets.
  • recruit_year ≤ 2024 OR recruit_year unknown — true frosh from
    the 2025 class (recruit_year = 2025) are excluded as too young.
  • Not in college_to_nfl_draft_linkage (already drafted).

The position parquet only matches recruiting on player_id ~36% of
the time, so we deliberately don't *require* a recruit-year hit;
fans sort by composite z and the realistic top-100 mostly has the
recruiting data anyway. We tag verified vs unverified eligibility
in the UI so fans know.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"
_COLLEGE = _DATA / "college"

DRAFT_YEAR = 2027
_PROSPECT_SEASON = 2025

# Position parquet sources. Defense rolls up multiple positions —
# we split via listed_position to land prospects on the right
# College-mode leaderboard when the user clicks "Profile".
# `min_volume` filters out tiny-sample artifacts (the 1-attempt QB
# with 100% completion problem). The volume_col is the key counting
# stat for the position.
_POSITION_SOURCES = [
    # (label, parquet, name_col, cm_pos, volume_col, min_volume)
    ("QB",  "college_qb_all_seasons.parquet",  "player", "QB",  "pass_att",      100),
    ("RB",  "college_rb_all_seasons.parquet",  "player", "RB",  "rush_carries",  30),
    ("WR",  "college_wr_all_seasons.parquet",  "player", "WR",  "receptions",    15),
    ("TE",  "college_te_all_seasons.parquet",  "player", "TE",  "receptions",    10),
    ("OL",  "college_ol_roster.parquet",       "player", "OL",  None,            None),
    ("DEF", "college_def_all_seasons.parquet", "player", None,  "tackles_total", 15),
]

# Defense listed_position → College-mode key (matches the mapping
# in pages/CollegeTeam.py for the roster click-through).
_DEF_POS_MAP = {
    "CB": "CB", "DB": "S", "S": "S",
    "DE": "DE", "EDGE": "DE",
    "DT": "DT", "DL": "DT", "NT": "DT",
    "LB": "LB", "ILB": "LB", "OLB": "LB",
}


@st.cache_data(show_spinner=False)
def _load_recruiting() -> pd.DataFrame:
    path = _COLLEGE / "college_recruiting.parquet"
    if not path.exists():
        return pd.DataFrame()
    keep = ["player_id", "recruit_year", "stars", "ranking",
            "rating", "height", "weight", "city", "state"]
    df = pd.read_parquet(path)
    return df[[c for c in keep if c in df.columns]]


@st.cache_data(show_spinner=False)
def _load_drafted_ids() -> set:
    """player_ids already drafted (drop them from the prospect pool).
    Linkage stores college_id as float; we cast to string-of-int to
    match the position-parquet player_id format."""
    path = _COLLEGE / "college_to_nfl_draft_linkage.parquet"
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    if "college_id" not in df.columns:
        return set()
    ids = df["college_id"].dropna()
    return {str(int(x)) for x in ids}


@st.cache_data(show_spinner=False)
def load_2027_prospects(*,
                          fbs_only: bool = True,
                          min_z_coverage: float = 0.6) -> pd.DataFrame:
    """Master DataFrame of 2027-eligible prospects across positions.

    `fbs_only` — exclude FCS programs (the realistic 2027 NFL draft
    pool is FBS, plus a tiny handful of FCS exceptions). On by
    default; users can toggle OFF in the UI to see FCS outliers.

    `min_z_coverage` — require at least this fraction of the
    position's z-cols to be populated. Filters out 1-attempt QBs
    and other tiny-sample artifacts that dominate raw composite z.

    Columns: player, player_id, team, conference, position,
    composite_z, n_z_valid, recruit_year, stars, ranking, rating,
    height, weight, city, state, eligibility_verified.
    """
    from lib_college_team_comps import _classify_tier
    rec = _load_recruiting()
    drafted = _load_drafted_ids()

    frames = []
    for pos_label, fname, name_col, cm_pos, vol_col, min_vol in _POSITION_SOURCES:
        path = _COLLEGE / fname
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "season" in df.columns:
            df = df[df["season"] == _PROSPECT_SEASON].copy()
        if df.empty:
            continue
        z_cols = [c for c in df.columns if c.endswith("_z")]
        if not z_cols:
            continue
        df["composite_z"] = df[z_cols].mean(axis=1, skipna=True)
        df["n_z_valid"] = df[z_cols].notna().sum(axis=1)
        df = df.dropna(subset=["composite_z"])
        # Filter tiny-sample artifacts. Two gates:
        #   1. enough z-cols populated to trust the composite
        #   2. enough volume on the position's key counting stat
        df = df[df["n_z_valid"] >= max(1, int(len(z_cols) * min_z_coverage))]
        if vol_col and vol_col in df.columns and min_vol is not None:
            df = df[df[vol_col].fillna(0) >= min_vol]
        if df.empty:
            continue
        if name_col != "player" and name_col in df.columns:
            df = df.rename(columns={name_col: "player"})

        # Bring in recruit info via player_id (cast both to string)
        if "player_id" in df.columns:
            df["player_id"] = df["player_id"].astype(str)
        if not rec.empty and "player_id" in rec.columns:
            rec_str = rec.copy()
            rec_str["player_id"] = rec_str["player_id"].astype(str)
            df = df.merge(rec_str, on="player_id", how="left")
        else:
            for c in ("recruit_year", "stars", "ranking", "rating",
                       "height", "weight", "city", "state"):
                df[c] = np.nan

        # Drop confirmed-too-young (recruit_year > 2024 = 2025 class)
        df = df[~(df["recruit_year"].notna()
                    & (df["recruit_year"] > DRAFT_YEAR - 3))]
        # Drop already-drafted
        if "player_id" in df.columns:
            df = df[~df["player_id"].astype(str).isin(drafted)]

        # Position assignment
        if cm_pos:
            df["position"] = cm_pos
        else:
            lp_col = ("listed_position" if "listed_position" in df.columns
                      else "pos_group")
            df["position"] = (df[lp_col].fillna("LB").astype(str)
                              .str.upper().map(_DEF_POS_MAP).fillna("LB"))

        df["eligibility_verified"] = (df["recruit_year"].notna()
                                        & (df["recruit_year"] <= DRAFT_YEAR - 3))

        keep = ["player", "player_id", "team", "conference", "position",
                "composite_z", "n_z_valid", "recruit_year", "stars",
                "ranking", "rating", "height", "weight", "city", "state",
                "eligibility_verified"]
        keep = [c for c in keep if c in df.columns]
        frames.append(df[keep])

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # Dedupe by player_id keeping the highest-composite row (defense
    # players may show up across pos_group buckets).
    out = (out.sort_values("composite_z", ascending=False)
              .drop_duplicates("player_id", keep="first")
              .reset_index(drop=True))

    # FBS-only filter (default ON)
    if fbs_only:
        # Re-use the tier classifier from the team-comp engine. A
        # team-season is FBS if its tier classifies as P4 or G5.
        def _is_fbs(row):
            tier = _classify_tier(str(row["team"]),
                                    row.get("conference"),
                                    _PROSPECT_SEASON)
            return tier in ("P4", "G5")
        out = out[out.apply(_is_fbs, axis=1)].reset_index(drop=True)

    return out

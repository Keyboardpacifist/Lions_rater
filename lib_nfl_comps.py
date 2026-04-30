"""NFL statistical-comp engine for college prospects.

For each prospect, find historical college players whose pre-draft
production profile most closely matches, and reveal their NFL
outcome (draft round, pick, team).

Stats used per position are stable across eras (counting + rate
stats — not CFBD-derived EPA which has uneven historical coverage
back to 2014). Cosine similarity on z-vectors with NaN → 0 fill,
top-N reveal.

Coverage:
  • QB / WR / TE / RB → use data/college/college_to_nfl_linked.parquet
  • CB / S / LB / DE / DT → join data/college/college_def_all_seasons
    with college_to_nfl_draft_linkage on player_id
  • OL → not available in v1 (no historical OL linkage parquet yet)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"
_COLLEGE = _DATA / "college"

# Stable stat sets per position. Each stat must exist as a `_z` col in
# the relevant historical pool. Avoid EPA / NGS-era stats (coverage
# changes) and per-snap usage stats (data missing pre-2018).
_STATS = {
    "QB":  ["completion_pct_z", "td_rate_z", "int_rate_z",
             "yards_per_attempt_z", "pass_tds_z",
             "rush_yards_total_z"],
    "WR":  ["rec_yards_total_z", "rec_tds_total_z",
             "receptions_total_z", "yards_per_rec_z"],
    "TE":  ["rec_yards_total_z", "rec_tds_total_z",
             "receptions_total_z", "yards_per_rec_z"],
    "RB":  ["rush_yards_total_z", "rush_tds_total_z",
             "carries_total_z", "yards_per_carry_z",
             "total_yards_z"],
    # Defensive z-cols come from college_def_all_seasons.parquet
    "CB":  ["tackles_per_game_z", "pd_per_game_z",
             "int_per_game_z", "solo_tackles_per_game_z"],
    "S":   ["tackles_per_game_z", "pd_per_game_z",
             "int_per_game_z", "solo_tackles_per_game_z"],
    "LB":  ["tackles_per_game_z", "sacks_per_game_z",
             "tfl_per_game_z", "solo_tackles_per_game_z"],
    "DE":  ["sacks_per_game_z", "tfl_per_game_z",
             "qb_hurries_per_game_z", "tackles_per_game_z"],
    "DT":  ["sacks_per_game_z", "tfl_per_game_z",
             "tackles_per_game_z"],
}

# Map our internal position keys → the values found in
# college_def_all_seasons listed_position. We pool synonyms so a
# 'DB' in older data can serve as a Safety comp.
_DEF_POS_MATCH = {
    "CB": ["CB"],
    "S":  ["S", "DB"],
    "LB": ["LB", "ILB", "OLB"],
    "DE": ["DE", "EDGE"],
    "DT": ["DT", "DL", "NT"],
}


@st.cache_data(show_spinner=False)
def _skill_pool() -> pd.DataFrame:
    path = _COLLEGE / "college_to_nfl_linked.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "season" not in df.columns or "draft_year" not in df.columns:
        return pd.DataFrame()
    pool = df[df["season"] == df["draft_year"] - 1].copy()
    return pool


@st.cache_data(show_spinner=False)
def _def_pool() -> pd.DataFrame:
    df_path = _COLLEGE / "college_def_all_seasons.parquet"
    dl_path = _COLLEGE / "college_to_nfl_draft_linkage.parquet"
    if not df_path.exists() or not dl_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(df_path)
    dl = pd.read_parquet(dl_path)
    df["player_id"] = df["player_id"].astype(str)
    dl = dl.dropna(subset=["college_id"]).copy()
    dl["college_id_str"] = dl["college_id"].astype(int).astype(str)
    joined = df.merge(
        dl[["college_id_str", "draft_year", "round", "pick",
            "overall", "nfl_team"]].rename(columns={
            "round": "draft_round", "pick": "draft_pick",
            "overall": "draft_overall",
        }),
        left_on="player_id", right_on="college_id_str", how="inner",
    )
    return joined[joined["season"] == joined["draft_year"] - 1].copy()


@st.cache_data(show_spinner=False)
def _pool_matrix(position: str) -> tuple:
    """Cached per-position (matrix, candidate_dicts) for vectorized
    similarity. Builds once per position, reused for every prospect.

    Returns:
      M: np.ndarray of shape (N_candidates, len(stats)) — already
         NaN→0 and clipped to ±_Z_CLIP
      candidates: list of dicts (one per row) with the metadata we
         need to format the comp output
      stats: list of stat columns used
    """
    if position not in _STATS:
        return np.zeros((0, 0)), [], []
    stats = _STATS[position]

    if position in ("QB", "WR", "TE", "RB"):
        pool = _skill_pool()
        if pool.empty:
            return np.zeros((0, 0)), [], stats
        pool = pool[pool["pos_group"] == position]
    else:
        pool = _def_pool()
        if pool.empty:
            return np.zeros((0, 0)), [], stats
        match_set = _DEF_POS_MATCH.get(position, [position])
        pool = pool[pool["listed_position"].isin(match_set)]

    if pool.empty:
        return np.zeros((0, 0)), [], stats

    available = [c for c in stats if c in pool.columns]
    if not available:
        return np.zeros((0, 0)), [], stats

    # Drop rows where ALL z-values are NaN; replace remaining NaN
    # with 0 (the cohort mean). Clip extremes to neutralize tiny-
    # sample artifacts before similarity calc.
    M_raw = pool[available].astype(float).values
    keep = ~np.isnan(M_raw).all(axis=1)
    pool = pool.iloc[keep].reset_index(drop=True)
    M = np.where(np.isnan(M_raw[keep]), 0.0, M_raw[keep])
    M = np.clip(M, -_Z_CLIP, _Z_CLIP)

    candidates = []
    for _, c in pool.iterrows():
        candidates.append({
            "player": c.get("player", "—"),
            "school": c.get("team", "—"),
            "season": (int(c["season"])
                        if pd.notna(c.get("season")) else None),
            "draft_year": (int(c["draft_year"])
                            if pd.notna(c.get("draft_year")) else None),
            "draft_round": (int(c["draft_round"])
                             if pd.notna(c.get("draft_round")) else None),
            "draft_pick": (int(c["draft_pick"])
                            if pd.notna(c.get("draft_pick")) else None),
            "draft_overall": (int(c["draft_overall"])
                               if pd.notna(c.get("draft_overall")) else None),
            "nfl_team": c.get("nfl_team", "—"),
        })
    return M, candidates, available


_Z_CLIP = 2.5  # Winsorize z-values beyond ±2.5σ — small-sample artifacts
              # (a defender with 1 game / 3 sacks gets sacks_per_game_z = 5+
              # which would dominate Euclidean distance and zero out every
              # comp similarity score).


def _euclidean_similarity(a: np.ndarray, b: np.ndarray,
                              scale: float = 4.0) -> float:
    """Convert z-vector Euclidean distance into a 0-1 similarity score.
    With z-clipping at ±2.5 the max per-dim distance is 5; for a 4-dim
    vector that's max total ~10, so scale=4 means players within 1z
    everywhere score ~80%, very different players land near 0."""
    diff = a - b
    dist = float(np.sqrt(np.dot(diff, diff)))
    return max(0.0, 1.0 - dist / scale)


def _vec(row: pd.Series, stats: list[str]) -> np.ndarray | None:
    """Clipped z-vector for the prospect/comp. NaN → 0 (mean), values
    beyond ±_Z_CLIP get clipped. Returns None if too sparse."""
    available = [c for c in stats if c in row.index]
    if not available:
        return None
    vals = row[available].astype(float).values
    if np.isnan(vals).all():
        return None
    vals = np.where(np.isnan(vals), 0.0, vals)
    return np.clip(vals, -_Z_CLIP, _Z_CLIP)


def _find_comps_vectorized(prospect_row: pd.Series, position: str,
                              top_pool_n: int = 50) -> list[dict]:
    """Vectorized core: returns the top-N most similar historical
    profiles in one matrix op. Cached pool matrix means this is
    ~50x faster than the old iterrows version on a 1k-row pool."""
    if position not in _STATS:
        return []
    M, candidates, available = _pool_matrix(position)
    if M.size == 0 or len(candidates) == 0:
        return []

    target = _vec(prospect_row, available)
    if target is None:
        return []

    # One matrix subtraction + norm — replaces the per-row iterrows.
    diffs = M - target
    dists = np.sqrt((diffs * diffs).sum(axis=1))
    sims = np.maximum(0.0, 1.0 - dists / 4.0)

    # Top-N indices by similarity (descending).
    n = min(top_pool_n, len(sims))
    top_idx = np.argpartition(-sims, n - 1)[:n] if n > 0 else []
    # Sort the top-N portion for ranked display
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    out = []
    for i in top_idx:
        c = dict(candidates[i])
        c["similarity"] = float(sims[i])
        out.append(c)
    return out


def find_nfl_comps(prospect_row: pd.Series, position: str,
                      n: int = 5) -> list[dict]:
    """Return top-N NFL-outcome comps for a college prospect."""
    return _find_comps_vectorized(prospect_row, position, top_pool_n=n)


def hit_rate_distribution(prospect_row: pd.Series, position: str,
                              top_pool_n: int = 50) -> dict:
    """Bust-probability framing — of the top-N most similar historical
    profiles, what fraction landed in each draft tier."""
    comps = _find_comps_vectorized(prospect_row, position,
                                       top_pool_n=top_pool_n)
    if not comps:
        return {}
    rounds = [c["draft_round"] for c in comps if c["draft_round"]]
    if not rounds:
        return {}
    n = len(rounds)
    r1 = sum(1 for r in rounds if r == 1)
    r2_3 = sum(1 for r in rounds if r in (2, 3))
    r4_7 = sum(1 for r in rounds if r >= 4)
    return {
        "r1": r1 / n,
        "r2_3": r2_3 / n,
        "r4_7": r4_7 / n,
        "top_pool_n": n,
    }


def find_comps_with_hit_rate(prospect_row: pd.Series, position: str,
                                  comp_n: int = 5,
                                  pool_n: int = 50) -> tuple[list[dict], dict]:
    """Single-pass version: returns (top_comp_n comps, hit_rate dict)
    from one similarity computation. Used by attach_nfl_comps to
    avoid doing the same work twice per prospect."""
    full = _find_comps_vectorized(prospect_row, position,
                                       top_pool_n=pool_n)
    if not full:
        return [], {}
    top = full[:comp_n]
    rounds = [c["draft_round"] for c in full if c["draft_round"]]
    if not rounds:
        return top, {}
    n = len(rounds)
    hr = {
        "r1":   sum(1 for r in rounds if r == 1) / n,
        "r2_3": sum(1 for r in rounds if r in (2, 3)) / n,
        "r4_7": sum(1 for r in rounds if r >= 4) / n,
        "top_pool_n": n,
    }
    return top, hr


# ── Strengths & concerns ────────────────────────────────────────
# Plain-English label per z-stat. The position parquets already
# sign-align z-cols so positive = good direction (e.g. INT-rate z is
# flipped: positive z = LOWER INT rate = better ball security).

_STAT_LABELS = {
    # QB
    "completion_pct_z":      "Accuracy (completion %)",
    "td_rate_z":             "TD per attempt",
    "int_rate_z":            "Ball security (low INT)",
    "yards_per_attempt_z":   "Big-play passing (Y/A)",
    "pass_tds_z":            "TD volume",
    "rush_yards_total_z":    "Rushing yardage",
    # WR / TE
    "rec_yards_total_z":     "Receiving yardage",
    "rec_tds_total_z":       "TD production",
    "receptions_total_z":    "Target absorber / volume",
    "yards_per_rec_z":       "Explosive playmaker (Y/R)",
    # RB
    "rush_tds_total_z":      "TD finishing",
    "carries_total_z":       "Workhorse usage",
    "yards_per_carry_z":     "Explosive runner (YPC)",
    "total_yards_z":         "Total scrimmage yardage",
    # Defense
    "tackles_per_game_z":    "Tackle production",
    "sacks_per_game_z":      "Pass-rush production",
    "tfl_per_game_z":        "Backfield disruption (TFL)",
    "qb_hurries_per_game_z": "Pressure generation",
    "pd_per_game_z":         "Coverage / pass break-ups (PBU)",
    "int_per_game_z":        "Ball-hawk (INT rate)",
    "solo_tackles_per_game_z": "Solo-tackle volume",
    "pressure_rate_z":       "Pressures per game",
}

# Position-specific overrides where the same z-col means different
# things (e.g. rush_yards_total is "rushing volume" for an RB but
# "QB mobility" for a QB).
_POS_LABEL_OVERRIDES = {
    "QB": {
        "rush_yards_total_z": "QB mobility (rush yds)",
    },
}


def _label_for(stat: str, position: str) -> str:
    pos_map = _POS_LABEL_OVERRIDES.get(position, {})
    return (pos_map.get(stat)
            or _STAT_LABELS.get(stat, stat.replace("_z", "")
                                          .replace("_", " ").title()))


def _z_to_percentile(z: float) -> int:
    """Standard-normal CDF → percentile. No scipy dependency."""
    import math
    return round(0.5 * (1 + math.erf(z / math.sqrt(2))) * 100)


def get_stat_profile(prospect_row: pd.Series, position: str) -> dict:
    """Return the prospect's top strengths and concerns based on the
    z-vector. Strengths = top 3 stats with z ≥ 1.0; concerns = bottom
    3 with z ≤ -0.5. Empty lists if nothing qualifies."""
    if position not in _STATS:
        return {"strengths": [], "concerns": []}
    items = []
    for s in _STATS[position]:
        if s in prospect_row.index and pd.notna(prospect_row.get(s)):
            z = float(prospect_row[s])
            z_clip = max(-3.0, min(3.0, z))
            # Cap percentile at 99 — '100th pctl' isn't a real thing.
            pct = min(99, max(1, _z_to_percentile(z_clip)))
            items.append({
                "stat": s,
                "label": _label_for(s, position),
                "z": z,
                "pct": pct,
            })
    # Strengths — top 3 with z ≥ 1
    strengths = sorted([i for i in items if i["z"] >= 1.0],
                          key=lambda x: -x["z"])[:3]
    # Bottom 3 stats — ALWAYS shown. The page decides framing:
    # 'Statistical Weaknesses' if any are below -0.5σ, else
    # 'Profile Gaps' (lowest stats in an above-average profile —
    # still useful info, frames honestly that nothing is bad).
    bottom = sorted(items, key=lambda x: x["z"])[:3]
    return {"strengths": strengths, "concerns": bottom}


@st.cache_data(show_spinner=False)
def _recruiting_lookup() -> pd.DataFrame:
    """Recruiting parquet, normalized for lookup. Cached once.
    Note: player_id has collisions (multiple recruits sharing the
    same id), so we match on name + school primarily."""
    path = _COLLEGE / "college_recruiting.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    keep = [c for c in ("player_id", "name", "school", "position",
                          "stars", "ranking", "rating",
                          "height", "weight", "city", "state",
                          "recruit_year")
             if c in df.columns]
    out = df[keep].copy()
    if "player_id" in out.columns:
        out["player_id"] = out["player_id"].astype(str)
    if "name" in out.columns:
        out["_n"] = out["name"].astype(str).str.lower().str.strip()
    if "school" in out.columns:
        out["_s"] = out["school"].astype(str).str.lower().str.strip()
    return out


def lookup_prospect_row(player_name: str, school: str,
                            position: str) -> pd.Series | None:
    """Find the prospect's 2025-season row from the relevant position
    parquet, with recruiting fields (stars, rating, height, weight)
    merged in. Used by the Draft page to feed find_nfl_comps and the
    athleticism scorers."""
    pos_files = {
        "QB": "college_qb_all_seasons.parquet",
        "WR": "college_wr_all_seasons.parquet",
        "TE": "college_te_all_seasons.parquet",
        "RB": "college_rb_all_seasons.parquet",
    }
    if position in pos_files:
        path = _COLLEGE / pos_files[position]
    elif position in ("CB", "S", "LB", "DE", "DT"):
        path = _COLLEGE / "college_def_all_seasons.parquet"
    else:
        return None
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = df[df["season"] == 2025]
    match = df[(df["player"] == player_name) & (df["team"] == school)]
    if match.empty:
        match = df[df["player"] == player_name]
    if match.empty:
        return None
    row = match.iloc[0].copy()

    # Merge in recruiting fields. We match on NAME + SCHOOL because
    # player_id has collisions (e.g. Will Smith Jr. and Jeremiah
    # Smith share player_id 5079720 at Ohio State, which previously
    # made Smith inherit Will's 4-star DL profile). Falls back to
    # name-only with most recent recruit_year for transfer-portal
    # cases. Doesn't overwrite values already on the row.
    rec = _recruiting_lookup()
    if not rec.empty:
        norm_name = player_name.lower().strip()
        norm_school = school.lower().strip()
        cands = rec[rec["_n"] == norm_name]
        if not cands.empty:
            school_match = cands[cands["_s"] == norm_school]
            if not school_match.empty:
                rec_match = school_match.iloc[0]
            else:
                # Transfer or school-name mismatch — take most
                # recent recruit_year for that name.
                rec_match = (cands.sort_values("recruit_year",
                                                  ascending=False)
                                  .iloc[0])
            for c in rec.columns:
                if c in ("player_id", "_n", "_s", "name",
                          "school", "position"):
                    continue
                if c not in row.index or pd.isna(row.get(c)):
                    row[c] = rec_match[c]
    return row

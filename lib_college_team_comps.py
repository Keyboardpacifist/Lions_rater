"""College team comp engine — same cosine-similarity + narrative
pattern as lib_team_comps.py, but on the position-group strength
rollups for college teams.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_college_team_seasons() -> pd.DataFrame:
    path = _DATA / "college_team_seasons.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# Stat columns we compare on. All are *_z derived position-group
# strength scores from tools/build_college_team_seasons.py.
_OFFENSE_COLS = [
    "qb_strength_z", "wr_te_pass_strength_z", "te_strength_z",
    "rb_strength_z", "ol_strength_z",
]
_DEFENSE_COLS = ["def_all_strength_z"]
_FULL_COLS = _OFFENSE_COLS + _DEFENSE_COLS

SCOPES = {
    "offense": _OFFENSE_COLS,
    "defense": _DEFENSE_COLS,
    "full":    _FULL_COLS,
}

# (z-stat, threshold, short, full phrase)
_NARRATIVE_TEMPLATES = [
    ("qb_strength_z",         0.7, "QB play",          "had elite quarterback play"),
    ("wr_te_pass_strength_z", 0.7, "receiver corps",   "fielded a loaded receiver corps"),
    ("te_strength_z",         0.7, "tight end",        "had a difference-maker at tight end"),
    ("rb_strength_z",         0.7, "rushing attack",   "ran the ball with a dominant backfield"),
    ("ol_strength_z",         0.5, "offensive line",   "won up front with a veteran O-line"),
    ("def_all_strength_z",    0.5, "defense",          "smothered offenses on D"),
]


def _get_team_vector(df: pd.DataFrame, team: str, season: int,
                       stat_cols: list[str]) -> np.ndarray | None:
    row = df[(df["team"] == team) & (df["season"] == season)]
    if row.empty:
        return None
    available = [c for c in stat_cols if c in df.columns]
    if not available:
        return None
    vals = row.iloc[0][available].astype(float).values
    if np.isnan(vals).all():
        return None
    return np.where(np.isnan(vals), 0.0, vals)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_college_team_comps(team: str, season: int, *,
                                scope: str = "full",
                                n: int = 3,
                                exclude_same_team: bool = True,
                                min_seasons_apart: int = 0,
                                min_total_players: int = 20) -> list[dict]:
    """Return top-N most comparable college team-seasons.

    `min_total_players` — only consider team-seasons with at least this
    many players counted in our system. The dataset caps at 23 players
    per team (1 QB + 3 WR/TE + 1 TE + 2 RB + 5 OL + 11 DEF), so this
    threshold filters small-sample FCS programs while still keeping
    full Power-4 rosters in the pool.
    """
    df = load_college_team_seasons()
    if df.empty:
        return []
    stat_cols = SCOPES.get(scope, _FULL_COLS)
    target = _get_team_vector(df, team, season, stat_cols)
    if target is None:
        return []

    # Each row's total roster sample = sum of *_n columns
    n_cols = [c for c in df.columns if c.endswith("_n")]
    if n_cols:
        df = df.assign(_total_n=df[n_cols].sum(axis=1, skipna=True))
    else:
        df = df.assign(_total_n=0)

    # Cosine works well in multi-D (captures shape: which traits are
    # the team's strengths). In 1-D (defense scope, until we add more
    # defensive z-stats) it degenerates to ±1, so every above-avg
    # defense looks 100% similar to every other one. Fall back to
    # value-distance in that case so we actually find teams with a
    # similar defensive strength level.
    use_distance = (len(stat_cols) == 1)

    sims: list[tuple[float, str, int, np.ndarray]] = []
    for _, row in df.iterrows():
        t, s = row["team"], int(row["season"])
        total_n = int(row.get("_total_n", 0) or 0)
        if t == team and s == season:
            continue
        if exclude_same_team and t == team:
            continue
        if min_seasons_apart and abs(s - season) < min_seasons_apart:
            continue
        if total_n < min_total_players:
            continue
        vec = _get_team_vector(df, t, s, stat_cols)
        if vec is None:
            continue
        if use_distance:
            # Map |Δz| → similarity in [0, 1]. 4 z-units apart = 0.
            sim = max(0.0, 1.0 - abs(float(target[0]) - float(vec[0])) / 4.0)
        else:
            sim = _cosine(target, vec)
        sims.append((sim, t, s, vec))

    sims.sort(key=lambda x: x[0], reverse=True)
    top = sims[:n]
    out = []
    for sim, t, s, comp_vec in top:
        reason = _build_comp_reason(target, comp_vec, stat_cols)
        out.append({
            "team": t,
            "season": s,
            "similarity": sim,
            "reason": reason,
        })
    return out


def _build_comp_reason(target: np.ndarray, comp: np.ndarray,
                         stat_cols: list[str]) -> str:
    df = load_college_team_seasons()
    available = [c for c in stat_cols if c in df.columns]
    if len(available) != len(target):
        return ""

    candidates: list[tuple[float, str]] = []
    for stat, thresh, _short, full in _NARRATIVE_TEMPLATES:
        if stat not in available:
            continue
        i = available.index(stat)
        if target[i] >= thresh and comp[i] >= thresh:
            candidates.append((float(target[i]) + float(comp[i]), full))
    candidates.sort(reverse=True)
    shared_phrases = [c[1] for c in candidates[:3]]

    if not shared_phrases:
        diffs = np.abs(target - comp)
        cands = [(diffs[i], available[i])
                 for i in range(len(available))
                 if not np.isnan(diffs[i])
                 and target[i] > -0.5 and comp[i] > -0.5]
        cands.sort()
        for _, stat in cands[:2]:
            for tmpl in _NARRATIVE_TEMPLATES:
                if tmpl[0] == stat:
                    shared_phrases.append(tmpl[3])
                    break

    if not shared_phrases:
        return ("Statistically the closest match in our system — no "
                "single trait dominates, but the overall profile lines up.")
    if len(shared_phrases) == 1:
        return f"Both {shared_phrases[0]}."
    if len(shared_phrases) == 2:
        return f"Both {shared_phrases[0]} AND {shared_phrases[1]}."
    return f"Both {', '.join(shared_phrases[:-1])}, and {shared_phrases[-1]}."

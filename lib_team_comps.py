"""Team comp engine — cosine similarity over z-vectors + narrative.

Find the most comparable team-seasons in our database to a given
team-season, with a generated reason explaining what makes them similar.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_team_seasons() -> pd.DataFrame:
    return pd.read_parquet(_DATA / "team_seasons.parquet")


# Stat groups for similarity computation. Pick which subset you compare on.
_OFFENSE_STATS = [
    "off_epa_per_play_z", "off_pass_epa_per_play_z", "off_rush_epa_per_play_z",
    "off_success_rate_z", "off_explosive_rate_z",
    "off_red_zone_td_rate_z", "off_third_down_conv_rate_z",
    "off_giveaway_rate_z", "points_per_game_z",
    "fourth_q_off_epa_z",
]
_DEFENSE_STATS = [
    "def_epa_per_play_z", "def_pass_epa_allowed_z", "def_rush_epa_allowed_z",
    "def_success_rate_allowed_z", "def_takeaway_rate_z",
    "def_pressure_rate_z", "def_sack_rate_z",
    "points_allowed_per_game_z", "fourth_q_def_epa_z",
]
_FULL_TEAM_STATS = _OFFENSE_STATS + _DEFENSE_STATS + [
    "point_differential_per_game_z", "penalty_yards_per_game_z",
]

SCOPES = {
    "offense": _OFFENSE_STATS,
    "defense": _DEFENSE_STATS,
    "full":    _FULL_TEAM_STATS,
}

# Narrative phrases — each entry: (z-stat threshold check fn, phrase template)
# When BOTH the target team-season AND the comp team-season exceed the
# threshold for a stat, we write that phrase into the comp's reason.
_NARRATIVE_TEMPLATES = [
    # (stat name, min z to qualify "elite", short phrase, full phrase)
    ("off_rush_epa_per_play_z", 0.7, "ground game",
     "ran the ball with EPA you couldn't stop"),
    ("off_pass_epa_per_play_z", 0.7, "passing attack",
     "passed efficiently in volume"),
    ("off_explosive_rate_z", 0.7, "explosive plays",
     "broke off chunk plays at an elite rate"),
    ("off_red_zone_td_rate_z", 0.5, "red zone closers",
     "didn't settle for field goals — finished drives with TDs"),
    ("off_third_down_conv_rate_z", 0.5, "money downs",
     "kept drives alive on third down"),
    ("off_giveaway_rate_z", 0.5, "ball security",
     "protected the football"),
    ("points_per_game_z", 0.7, "scoring punch",
     "put up points in volume"),
    ("def_epa_per_play_z", 0.7, "elite defense",
     "smothered offenses on a per-play basis"),
    ("def_pass_epa_allowed_z", 0.7, "secondary",
     "made life miserable on opposing QBs"),
    ("def_rush_epa_allowed_z", 0.7, "front seven",
     "stopped the run cold"),
    ("def_takeaway_rate_z", 0.7, "ball-hawks",
     "ripped the ball away from offenses"),
    ("def_pressure_rate_z", 0.7, "pass rush",
     "lived in opposing pockets"),
    ("def_sack_rate_z", 0.7, "sack production",
     "got home — high sack rate"),
    ("points_allowed_per_game_z", 0.7, "scoring defense",
     "shut down opposing scoring"),
    ("fourth_q_off_epa_z", 0.5, "fourth-quarter offense",
     "closed games in the fourth quarter"),
    ("fourth_q_def_epa_z", 0.5, "fourth-quarter defense",
     "locked down in late-game situations"),
]


def _get_team_vector(df: pd.DataFrame, team: str, season: int,
                       stat_cols: list[str]) -> np.ndarray | None:
    row = df[(df["team"] == team) & (df["season"] == season)]
    if row.empty:
        return None
    available = [c for c in stat_cols if c in df.columns]
    vals = row.iloc[0][available].astype(float).values
    if np.isnan(vals).all():
        return None
    # Replace NaN with 0 for cosine — equivalent to "average for missing"
    return np.where(np.isnan(vals), 0.0, vals)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_team_comps(team: str, season: int, *,
                      scope: str = "offense",
                      n: int = 3,
                      exclude_same_team: bool = False,
                      exclude_recent_seasons: int = 0) -> list[dict]:
    """Return top-N most comparable team-seasons.

    `scope` — "offense" / "defense" / "full"
    `exclude_same_team` — skip other seasons of the same franchise
    `exclude_recent_seasons` — skip seasons within N years of the target
        (use this if you want "historical" comps and not just last year's
        version of the same team)
    """
    df = load_team_seasons()
    stat_cols = SCOPES.get(scope, _OFFENSE_STATS)
    target = _get_team_vector(df, team, season, stat_cols)
    if target is None:
        return []

    sims: list[tuple[float, str, int, np.ndarray]] = []
    for _, row in df.iterrows():
        t, s = row["team"], int(row["season"])
        if t == team and s == season:
            continue
        if exclude_same_team and t == team:
            continue
        if exclude_recent_seasons and abs(s - season) <= exclude_recent_seasons:
            continue
        vec = _get_team_vector(df, t, s, stat_cols)
        if vec is None:
            continue
        sim = _cosine(target, vec)
        sims.append((sim, t, s, vec))

    sims.sort(key=lambda x: x[0], reverse=True)
    top = sims[:n]

    # Build narrative for each — why similar AND where they diverge
    out = []
    for sim, t, s, comp_vec in top:
        reason = _build_comp_reason(target, comp_vec, stat_cols)
        divergence = _build_comp_divergence(
            target, comp_vec, stat_cols,
            target_label=f"{team} {season}",
            comp_label=f"{s} {t}",
        )
        out.append({
            "team": t,
            "season": s,
            "similarity": sim,
            "reason": reason,
            "divergence": divergence,
        })
    return out


# When listing differences we want to mention the trait + the side that
# owned it. Map z-stat to a noun phrase that fits both directions.
_DIVERGENCE_PHRASES = {
    "off_rush_epa_per_play_z":      "ground game",
    "off_pass_epa_per_play_z":      "passing attack",
    "off_explosive_rate_z":         "explosive-play rate",
    "off_red_zone_td_rate_z":       "red zone finishing",
    "off_third_down_conv_rate_z":   "3rd-down conversion",
    "off_giveaway_rate_z":          "ball security",
    "points_per_game_z":            "scoring volume",
    "def_epa_per_play_z":           "overall defense",
    "def_pass_epa_allowed_z":       "pass defense",
    "def_rush_epa_allowed_z":       "run defense",
    "def_takeaway_rate_z":          "takeaway production",
    "def_pressure_rate_z":          "pass rush",
    "def_sack_rate_z":              "sack production",
    "points_allowed_per_game_z":    "scoring defense",
    "fourth_q_off_epa_z":           "4th-quarter offense",
    "fourth_q_def_epa_z":           "4th-quarter defense",
}


def _z_descriptor(z: float) -> str:
    """Plain-English level for a z-score."""
    if z >= 1.5:   return "elite"
    if z >= 0.7:   return "above-average"
    if z >= -0.7:  return "average"
    if z >= -1.5:  return "below-average"
    return "poor"


def _build_comp_divergence(target: np.ndarray, comp: np.ndarray,
                              stat_cols: list[str],
                              target_label: str,
                              comp_label: str) -> str:
    """Identify the 2-3 stats where the two team-seasons diverge most
    (large |z-delta|) AND at least one side hits elite/poor levels.
    Returns a fan-readable "where they diverge" sentence."""
    df = load_team_seasons()
    available = [c for c in stat_cols if c in df.columns]
    if len(available) != len(target):
        return ""

    candidates = []  # (abs_delta, stat, target_z, comp_z)
    for i, stat in enumerate(available):
        if stat not in _DIVERGENCE_PHRASES:
            continue
        if np.isnan(target[i]) or np.isnan(comp[i]):
            continue
        delta = float(target[i]) - float(comp[i])
        # Require ≥1 full descriptor-tier of difference (z gap ≥ 1.0)
        # so we don't surface "elite vs above-average" as divergence
        if abs(delta) < 1.0:
            continue
        # Skip unless the descriptors actually land in different tiers —
        # that's what makes the difference fan-readable.
        if _z_descriptor(target[i]) == _z_descriptor(comp[i]):
            continue
        # And require at least one side to be notable (≥ "above-average"
        # or ≤ "below-average") — otherwise it's noise around the mean.
        if abs(target[i]) < 0.7 and abs(comp[i]) < 0.7:
            continue
        candidates.append((abs(delta), stat, float(target[i]),
                            float(comp[i])))

    if not candidates:
        return ""
    candidates.sort(reverse=True)
    top = candidates[:3]

    fragments = []
    for _, stat, tz, cz in top:
        phrase = _DIVERGENCE_PHRASES.get(stat, stat.replace("_z", "")
                                                  .replace("_", " "))
        if tz > cz:
            # Target had it, comp didn't
            fragments.append(
                f"**{target_label}**'s {phrase} was {_z_descriptor(tz)} "
                f"({_z_descriptor(cz)} for {comp_label})"
            )
        else:
            fragments.append(
                f"**{comp_label}**'s {phrase} was {_z_descriptor(cz)} "
                f"({_z_descriptor(tz)} for {target_label})"
            )

    if len(fragments) == 1:
        return f"Where they diverge: {fragments[0]}."
    return "Where they diverge: " + "; ".join(fragments) + "."


def _build_comp_reason(target: np.ndarray, comp: np.ndarray,
                         stat_cols: list[str]) -> str:
    """Identify which stats both team-seasons share at elite levels and
    write a sentence about it.
    """
    df = load_team_seasons()
    available = [c for c in stat_cols if c in df.columns]
    if len(available) != len(target):
        return ""

    # Find templates where both target & comp clear the elite threshold,
    # ranked by combined strength (target_z + comp_z). Pick top 3 traits
    # for narrative — more than that and every elite team's story sounds
    # the same.
    candidates: list[tuple[float, str]] = []
    for tmpl in _NARRATIVE_TEMPLATES:
        stat, thresh, _short, full = tmpl
        if stat not in available:
            continue
        i = available.index(stat)
        if target[i] >= thresh and comp[i] >= thresh:
            combined = float(target[i]) + float(comp[i])
            candidates.append((combined, full))
    candidates.sort(reverse=True)
    shared_phrases = [c[1] for c in candidates[:3]]

    if not shared_phrases:
        # Fall back to the closest-match stats
        diffs = np.abs(target - comp)
        # Top 2 most similar (smallest diff) where both are at least
        # mediocre (not both bad)
        candidates = [(diffs[i], available[i], target[i], comp[i])
                      for i in range(len(available))
                      if not np.isnan(diffs[i]) and target[i] > -0.5 and comp[i] > -0.5]
        candidates.sort()
        for _, stat, _t, _c in candidates[:2]:
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
    return (f"Both {', '.join(shared_phrases[:-1])}, and {shared_phrases[-1]}.")

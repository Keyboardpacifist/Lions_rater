"""Two-axis athleticism scoring for 2027 Draft prospects.

Tested:     stars · 247 composite rating · height/weight (recruit
            data + listed roster); will absorb Feldman + Pro Day +
            Combine layers as those drop without code changes.

Contextual: on-field athletic markers per position (yards/touch,
            long catches/runs, pass-rush production rates, range
            tackles, ball-skills counts).

Both scored 1-10 via z → score linear mapping (z=-3→1, z=0→5,
z=+3→10, clipped). Divergence between the two is its own insight:
  • Tested high / Contextual low  → workout warrior
  • Tested low / Contextual high  → undersized overachiever
  • Both high                     → trait + translation = elite
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"
_COLLEGE = _DATA / "college"

# Per-position contextual stat sets — each must be a `_z` column in
# the relevant primary parquet (no advanced-parquet merge needed
# for v1; EPA can be added in v1.1).
_CONTEXTUAL_STATS = {
    "QB": ["yards_per_attempt_z", "rush_yards_total_z"],
    "WR": ["yards_per_rec_z", "rec_long_z"],
    "TE": ["yards_per_rec_z", "rec_long_z"],
    "RB": ["yards_per_carry_z"],
    # OL: no public individual signal; explicitly punt for v1
    "DE": ["sacks_per_game_z", "tfl_per_game_z",
           "qb_hurries_per_game_z", "pressure_rate_z"],
    "DT": ["sacks_per_game_z", "tfl_per_game_z", "pressure_rate_z"],
    "LB": ["tfl_per_game_z", "solo_tackles_per_game_z", "pd_per_game_z"],
    "CB": ["pd_per_game_z", "int_per_game_z", "solo_tackles_per_game_z"],
    "S":  ["pd_per_game_z", "int_per_game_z", "tfl_per_game_z"],
}

_LABELS = {
    "yards_per_attempt_z":     "Y/A (arm + decision)",
    "rush_yards_total_z":      "Rushing yardage (mobility)",
    "yards_per_rec_z":         "Y/R (explosive playmaker)",
    "rec_long_z":              "Long reception (speed)",
    "yards_per_carry_z":       "YPC (explosive runner)",
    "sacks_per_game_z":        "Sacks/g (burst + finish)",
    "tfl_per_game_z":          "TFL/g (closing speed)",
    "qb_hurries_per_game_z":   "Hurries/g (get-off)",
    "pressure_rate_z":         "Pressure rate efficiency",
    "solo_tackles_per_game_z": "Solo tackles/g (range)",
    "pd_per_game_z":           "PD/g (coverage athleticism)",
    "int_per_game_z":          "INT/g (anticipation + closing)",
}


def _z_to_score(z) -> float | None:
    """Linear z → 1-10. z=-3 → 1, z=0 → 5, z=+3 → 10, clipped."""
    if z is None or pd.isna(z):
        return None
    return float(np.clip(5.0 + (float(z) * 5.0 / 3.0), 1.0, 10.0))


def _stars_to_score(stars) -> float | None:
    """Recruit-star scoring: 5★ ≈ elite athletic ceiling, 1★ unrated.
    Unrated returns None (not "0") so we don't punish prospects whose
    recruit data is missing — common given the 36% join coverage."""
    if stars is None or pd.isna(stars):
        return None
    s = int(stars)
    return {5: 9.5, 4: 7.5, 3: 5.5, 2: 3.5, 1: 2.0}.get(s)


def _rating_to_score(rating) -> float | None:
    """247 composite rating → 1-10. Tiered against typical class
    distribution: 0.99+ is generational, 0.85+ is high-major P4."""
    if rating is None or pd.isna(rating):
        return None
    r = float(rating)
    if r >= 0.99: return 10.0
    if r >= 0.97: return 9.5
    if r >= 0.95: return 9.0
    if r >= 0.92: return 8.0
    if r >= 0.90: return 7.0
    if r >= 0.85: return 5.5
    if r >= 0.80: return 4.0
    return 2.5


@st.cache_data(show_spinner=False)
def _hw_position_norms() -> dict:
    """Build {position: {height_mean, height_std, weight_mean, weight_std}}
    from the 2025-season position parquets. Used to z-score height
    and weight against position cohort so a 6'2" 215lb LB scores
    high while same dimensions at OT score low."""
    norms = {}
    sources = {
        "QB": "college_qb_all_seasons.parquet",
        "WR": "college_wr_all_seasons.parquet",
        "TE": "college_te_all_seasons.parquet",
        "RB": "college_rb_all_seasons.parquet",
        "OL": "college_ol_roster.parquet",
    }
    for pos, fname in sources.items():
        path = _COLLEGE / fname
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "season" in df.columns:
            df = df[df["season"] == 2025]
        if df.empty:
            continue
        # height_in / weight from recruiting parquet not always merged
        # in. For now skip individual H/W norms — we already use
        # recruiting-parquet height/weight in the prospect lookup
        # which is a single value per prospect, not a season-cohort.
        norms[pos] = {}
    # Defense pool — use listed_position split
    def_path = _COLLEGE / "college_def_all_seasons.parquet"
    if def_path.exists():
        df = pd.read_parquet(def_path)
        if "season" in df.columns:
            df = df[df["season"] == 2025]
        for cm_pos, listed_set in [("CB", ["CB"]),
                                       ("S", ["S", "DB"]),
                                       ("LB", ["LB", "ILB", "OLB"]),
                                       ("DE", ["DE", "EDGE"]),
                                       ("DT", ["DT", "DL", "NT"])]:
            sub = df[df["listed_position"].isin(listed_set)]
            norms[cm_pos] = {}
    return norms


def compute_pedigree_score(prospect_row: pd.Series,
                              position: str) -> dict:
    """Pedigree: recruiting-era *evaluation* — how scouts/services
    rated this player at age 17. It's outside-source hype + scout
    consensus, NOT a measure of physical traits.

    Components: 247 composite stars · 247 composite rating · HS
    national rank.
    """
    components = {}

    star_score = _stars_to_score(prospect_row.get("stars"))
    if star_score is not None:
        components["Recruit stars"] = (
            star_score, f"{int(prospect_row['stars'])}-star",
        )

    rating_score = _rating_to_score(prospect_row.get("rating"))
    if rating_score is not None:
        components["247 composite"] = (
            rating_score, f"{prospect_row['rating']:.4f}",
        )

    ranking = prospect_row.get("ranking")
    if pd.notna(ranking):
        r = int(ranking)
        if r <= 10:
            rank_score = 10.0
        elif r <= 30:
            rank_score = 9.0
        elif r <= 100:
            rank_score = 7.5
        elif r <= 250:
            rank_score = 5.5
        elif r <= 500:
            rank_score = 4.0
        else:
            rank_score = 3.0
        components["HS national rank"] = (rank_score, f"#{r}")

    if not components:
        return {
            "score": None, "components": {}, "has_data": False,
            "note": "No recruiting data available for this prospect.",
        }
    avg = sum(s for s, _ in components.values()) / len(components)
    return {
        "score": round(avg, 1), "components": components,
        "has_data": True,
        "note": "Recruiting-era evaluation only.",
    }


def compute_tested_score(prospect_row: pd.Series, position: str) -> dict:
    """Tested: actual *measured* physical traits.

    v1 carries height/weight scored against position archetype. The
    v1.1 layer will add HS combine times (40, vertical, broad, bench,
    shuttle, 3-cone), track & field marks (100m / 200m / long jump /
    high jump / triple jump) via name+state web scrape, and Feldman
    Freak List flags (current + prior years' lists since underclassmen
    appear). v2 (Feb 2027) auto-adds NFL Combine + Pro Day from
    nflverse.
    """
    components = {}

    h, w = prospect_row.get("height"), prospect_row.get("weight")
    if pd.notna(h) and pd.notna(w):
        try:
            h_in = float(h) if not isinstance(h, str) else _parse_height(h)
            w_lb = float(w)
            hw_score = _hw_score(h_in, w_lb, position)
            if hw_score is not None:
                components["Size (H/W)"] = (
                    hw_score, f"{_height_str(h_in)}, {int(w_lb)} lbs",
                )
        except (ValueError, TypeError):
            pass

    # TODO v1.1: layer in track times, HS combine measurables,
    # Feldman Freak List flags.

    if not components:
        return {
            "score": None, "components": {}, "has_data": False,
            "note": ("No measured data yet — track times + HS combine "
                     "scrape pending; NFL Combine / Pro Day Feb 2027."),
        }
    avg = sum(s for s, _ in components.values()) / len(components)
    return {
        "score": round(avg, 1), "components": components,
        "has_data": True,
        "note": ("Size only (v1). Track / HS combine layer next; "
                 "Combine + Pro Day Feb 2027."),
    }


def _parse_height(h: str) -> float:
    """Parse heights like "6-3" or "6'3"" or "6.25" into inches."""
    s = str(h).replace("'", "").replace("\"", "").strip()
    if "-" in s:
        ft, inches = s.split("-")
        return int(ft) * 12 + int(inches)
    if "." in s:
        return float(s) * 12
    if len(s) <= 2:
        return float(s) * 12
    return float(s)


def _height_str(h_in: float) -> str:
    ft, inches = divmod(int(round(h_in)), 12)
    return f"{ft}'{inches}\""


# Position archetype height/weight expectations — 8 = ideal for slot,
# 6 = average, 4 = sub-typical, 2 = quite undersized for the role.
# Values are (ideal_h_in_low, ideal_h_in_high, ideal_w_lb_low,
# ideal_w_lb_high). Outside the band scales down.
_HW_BANDS = {
    "QB": (74, 78, 210, 235),
    "RB": (69, 73, 200, 225),
    "WR": (71, 76, 185, 215),
    "TE": (75, 79, 240, 265),
    "OL": (76, 80, 305, 335),
    "DE": (75, 79, 250, 275),
    "DT": (74, 78, 290, 320),
    "LB": (72, 76, 220, 245),
    "CB": (70, 74, 185, 205),
    "S":  (71, 75, 195, 215),
}


def _hw_score(h_in: float, w_lb: float, position: str) -> float | None:
    band = _HW_BANDS.get(position)
    if band is None:
        return None
    h_lo, h_hi, w_lo, w_hi = band
    # Score components: how close to ideal band on each axis.
    h_score = 8.0 if h_lo <= h_in <= h_hi else max(
        2.0, 8.0 - 1.5 * min(abs(h_in - h_lo), abs(h_in - h_hi))
    )
    w_score = 8.0 if w_lo <= w_lb <= w_hi else max(
        2.0, 8.0 - 0.15 * min(abs(w_lb - w_lo), abs(w_lb - w_hi))
    )
    return round((h_score + w_score) / 2.0, 1)


def compute_contextual_score(prospect_row: pd.Series,
                                position: str) -> dict:
    """Return {score (1-10 or None), components, has_data}."""
    if position not in _CONTEXTUAL_STATS:
        return {
            "score": None, "components": {}, "has_data": False,
            "note": ("OL contextual athleticism not measurable from "
                     "public data (no individual line snaps). v1.1: "
                     "PFF when available."),
        }
    stats = _CONTEXTUAL_STATS[position]
    components = {}
    for s in stats:
        if s in prospect_row.index and pd.notna(prospect_row.get(s)):
            z = float(prospect_row[s])
            score = _z_to_score(z)
            if score is None:
                continue
            label = _LABELS.get(s, s.replace("_z", "")
                                       .replace("_", " "))
            sign = "+" if z >= 0 else ""
            components[label] = (score, f"{sign}{z:.2f}σ")
    if not components:
        return {
            "score": None, "components": {}, "has_data": False,
            "note": "No 2025 production data yet.",
        }
    avg = sum(s for s, _ in components.values()) / len(components)
    return {"score": round(avg, 1), "components": components,
            "has_data": True}


def divergence_note(tested: float | None,
                       contextual: float | None) -> str | None:
    """Surface the tested-vs-contextual story when the two scores
    differ meaningfully. Returns None when the gap is small or one
    score is missing."""
    if tested is None or contextual is None:
        return None
    delta = tested - contextual
    if abs(delta) < 1.5:
        return None
    if delta >= 1.5:
        return ("⚠ Tested traits outpace on-field translation — "
                "watch for workout-warrior risk.")
    return ("💪 On-field performance outpaces tested traits — "
            "undersized overachiever profile.")

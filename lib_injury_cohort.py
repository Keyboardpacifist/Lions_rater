"""Cohort-matching engine for the gambling product's Feature 1
(Injury Probability + Usage Retention).

Given a player on a current injury report, find historical comparable
cases and compute the empirical probability he plays Sunday plus the
average snap share if active.

Data shape from nflreadpy.load_injuries():
  • One row per (player, season, week) — the FINAL Friday injury
    report for that week.
  • Columns we use: position, report_primary_injury, report_status
    (Out/Doubtful/Questionable/Probable/None), practice_status
    (DNP/Limited/Full).
  • There are NO separate Wed/Thu/Fri rows — only the Friday
    snapshot. So our "practice sequence" feature is a single Friday
    status, not a 3-day pattern.

Empirical cohort rates come from `data/injury_cohort_rates.parquet`,
which is built by `tools/build_injury_cohort_rates.py` — that script
joins the injury archive to actual game-day snap counts (2013+) so
"play rate" is the *real* fraction of comparable historical players
who took a snap on Sunday, not a hardcoded baseline.

Public entry points
-------------------
    body_part_normalize(raw)        — fuzzy → standardized bucket
    practice_status_code(raw)       — Friday status → one-char code
    report_status_code(raw)         — game-day designation → code
    predict(...)                    — single entrypoint, returns CohortResult
    find_comparable_cases(...)      — raw cohort lookup
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
INJURIES = DATA / "nfl_injuries_historical.parquet"
COHORT_RATES = DATA / "injury_cohort_rates.parquet"


# ── Body-part normalizer ────────────────────────────────────────
# nflverse's injury report strings are messy ("Knee", "Right Knee",
# "knee/ankle", "shldr", "concussion", etc.). Map fuzzy variations
# to ~25 standardized buckets. Multi-injury rows match on the FIRST
# bucket found — the primary listing is what gamblers care about.
_BUCKETS: list[tuple[str, list[str]]] = [
    ("ankle",       ["ankle", "achilles", "achilles tendon"]),
    ("knee",        ["knee", "acl", "mcl", "lcl", "pcl", "meniscus",
                      "patella"]),
    ("hamstring",   ["hamstring"]),
    ("quad",        ["quad", "quadriceps"]),
    ("calf",        ["calf"]),
    ("groin",       ["groin", "abductor"]),
    ("hip",         ["hip flexor", "hip"]),
    ("foot",        ["turf toe", "metatarsal", "lisfranc",
                      "foot", "toe"]),
    ("back",        ["lower back", "back", "spine"]),
    ("neck",        ["neck"]),
    ("shoulder",    ["rotator cuff", "ac joint", "labrum",
                      "shoulder", "shldr", "collarbone",
                      "clavicle"]),
    ("chest",       ["pectoral", "pec", "chest", "sternum",
                      "rib", "ribs"]),
    ("abdomen",     ["abdominal", "abs", "core", "oblique",
                      "abdomen"]),
    ("elbow",       ["elbow", "biceps", "tricep", "triceps"]),
    ("forearm",     ["forearm"]),
    ("wrist",       ["wrist"]),
    ("hand",        ["thumb", "finger", "hand"]),
    ("concussion",  ["concussion", "head injury", "head"]),
    ("illness",     ["non-football illness", "nfi", "illness",
                      "personal", "rest", "veteran rest day",
                      "not injury related"]),
    ("heat",        ["heat", "cramps"]),
    ("eye",         ["eye"]),
    ("face",        ["jaw", "tooth", "teeth", "face"]),
    ("hip_pointer", ["hip pointer"]),
    ("stinger",     ["stinger", "burner"]),
]


def body_part_normalize(raw: str | None) -> str:
    """Map a raw injury string to a standardized bucket. Returns
    'unknown' when nothing matches. Order of buckets matters —
    multi-word phrases like 'lower back' must be tested before
    short tokens like 'back'."""
    if not raw or not isinstance(raw, str):
        return "unknown"
    s = raw.lower().strip()
    s = re.sub(r"[^a-z\s/\-]", " ", s)
    for bucket, keywords in _BUCKETS:
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", s):
                return bucket
    return "unknown"


# ── Status normalizers ──────────────────────────────────────────

def practice_status_code(raw) -> str:
    """One-token Friday practice status: DNP / LIMITED / FULL /
    OUT / NONE."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "NONE"
    s = str(raw).upper().strip()
    if "DID NOT" in s or s == "DNP":
        return "DNP"
    if "LIMITED" in s:
        return "LIMITED"
    if "FULL" in s:
        return "FULL"
    if "OUT" in s:
        return "OUT"
    return "NONE"


def report_status_code(raw) -> str:
    """Sunday game-day designation: OUT / DOUBTFUL / QUESTIONABLE /
    PROBABLE / NONE."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "NONE"
    s = str(raw).upper().strip()
    for k in ("OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE"):
        if k in s:
            return k
    return "NONE"


# ── Empirical baselines (fallback when cohort is too thin) ──────
# Computed from the cohort-rates table aggregated by report_code only.
# These are league-wide marginals; the cohort table has the full
# (position, body_part, report, practice) crossed cells.
_PLAY_RATE_BY_REPORT = {
    "OUT":          0.001,
    "DOUBTFUL":     0.008,
    "QUESTIONABLE": 0.642,
    "PROBABLE":     0.928,
    "NONE":         0.911,
}

# Marginals by (report, practice) — second-best fallback when (pos, body)
# is too thin but the Friday designation cross is rich enough.
_PLAY_RATE_BY_REPORT_PRACTICE = {
    ("DOUBTFUL",     "DNP"):     0.009,
    ("DOUBTFUL",     "LIMITED"): 0.004,
    ("DOUBTFUL",     "FULL"):    0.000,
    ("NONE",         "DNP"):     0.732,
    ("NONE",         "LIMITED"): 0.913,
    ("NONE",         "FULL"):    0.936,
    ("OUT",          "DNP"):     0.000,
    ("OUT",          "LIMITED"): 0.001,
    ("OUT",          "FULL"):    0.000,
    ("PROBABLE",     "DNP"):     0.908,
    ("PROBABLE",     "LIMITED"): 0.961,
    ("PROBABLE",     "FULL"):    0.923,
    ("QUESTIONABLE", "DNP"):     0.407,
    ("QUESTIONABLE", "LIMITED"): 0.664,
    ("QUESTIONABLE", "FULL"):    0.775,
}


# ── Cached parquet loaders ──────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_archive() -> pd.DataFrame:
    """Loads + enriches the historical injury archive once."""
    if not INJURIES.exists():
        return pd.DataFrame()
    df = pd.read_parquet(INJURIES)
    return enrich_archive(df)


@st.cache_data(show_spinner=False)
def load_cohort_rates() -> pd.DataFrame:
    """Empirical cohort play-rate table built by
    `tools/build_injury_cohort_rates.py`."""
    if not COHORT_RATES.exists():
        return pd.DataFrame()
    return pd.read_parquet(COHORT_RATES)


def enrich_archive(df: pd.DataFrame) -> pd.DataFrame:
    """Add three derived columns to the raw archive."""
    df = df.copy()
    df["_body_part_bucket"] = df["report_primary_injury"].apply(
        body_part_normalize)
    df["_practice_code"] = df["practice_status"].apply(
        practice_status_code)
    df["_report_code"] = df["report_status"].apply(
        report_status_code)
    return df


# ── Cohort-match engine ─────────────────────────────────────────

@dataclass
class CohortResult:
    n: int                       # cohort sample size
    p_played: float              # Pr(plays Sunday) — empirical
    snap_share_if_played: float  # avg snap share when active (0–1)
    cohort_level: str            # "tight" / "loose" / "fallback" / "marginal"
    body_part: str
    position: str
    report_status: str
    practice_status: str


def _lookup_cohort(rates: pd.DataFrame, position: str, body_part: str,
                   report_code: str, practice_code: str) -> pd.Series | None:
    """Look up one cohort row by exact key. Returns the row or None."""
    if rates.empty:
        return None
    mask = ((rates["position"] == position)
            & (rates["body_part"] == body_part)
            & (rates["report_code"] == report_code)
            & (rates["practice_code"] == practice_code))
    sub = rates[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


def _lookup_loose(rates: pd.DataFrame, position: str, body_part: str,
                  report_code: str) -> tuple[int, float, float] | None:
    """Loose cohort: aggregate across all practice_codes for this
    (pos, body, report). Returns (n, play_rate, snap_share_if_played)."""
    if rates.empty:
        return None
    mask = ((rates["position"] == position)
            & (rates["body_part"] == body_part)
            & (rates["report_code"] == report_code))
    sub = rates[mask]
    if sub.empty:
        return None
    n = int(sub["n_cases"].sum())
    played = int(sub["n_played"].sum())
    rate = played / n if n else 0.0
    # Weighted avg of snap_share_if_played by n_played
    if played:
        ssip = ((sub["snap_share_if_played"].fillna(0)
                 * sub["n_played"]).sum() / played)
    else:
        ssip = 0.0
    return n, rate, float(ssip)


def _lookup_fallback(rates: pd.DataFrame, position: str,
                     body_part: str) -> tuple[int, float, float] | None:
    """Fallback: aggregate across all report+practice for (pos, body)."""
    if rates.empty:
        return None
    mask = ((rates["position"] == position)
            & (rates["body_part"] == body_part))
    sub = rates[mask]
    if sub.empty:
        return None
    n = int(sub["n_cases"].sum())
    played = int(sub["n_played"].sum())
    rate = played / n if n else 0.0
    if played:
        ssip = ((sub["snap_share_if_played"].fillna(0)
                 * sub["n_played"]).sum() / played)
    else:
        ssip = 0.0
    return n, rate, float(ssip)


def predict(position: str, body_part: str,
            report_status: str | None = None,
            practice_status: str | None = None,
            min_tight_n: int = 30) -> CohortResult:
    """Full prediction in one call.

    Tightening order:
      1. (pos, body, report, practice) — exact cohort, n ≥ min_tight_n
      2. (pos, body, report) — aggregated across practice codes
      3. (pos, body) — aggregated across report+practice
      4. (report, practice) marginal — last resort when body unknown
      5. (report) marginal — true fallback
    """
    rates = load_cohort_rates()
    pos = position.upper()
    body = body_part.lower()
    rep = report_status_code(report_status)
    pr = practice_status_code(practice_status)

    # Tier 1: exact tight cohort
    row = _lookup_cohort(rates, pos, body, rep, pr)
    if row is not None and int(row["n_cases"]) >= min_tight_n:
        ssip_raw = row.get("snap_share_if_played")
        ssip = 0.0 if pd.isna(ssip_raw) else float(ssip_raw)
        return CohortResult(
            n=int(row["n_cases"]),
            p_played=float(row["play_rate"]),
            snap_share_if_played=ssip,
            cohort_level="tight",
            body_part=body, position=pos,
            report_status=rep, practice_status=pr,
        )

    # Tier 2: loose (aggregate practice codes)
    loose = _lookup_loose(rates, pos, body, rep)
    if loose and loose[0] >= min_tight_n:
        n, rate, ssip = loose
        return CohortResult(
            n=n, p_played=rate, snap_share_if_played=ssip,
            cohort_level="loose",
            body_part=body, position=pos,
            report_status=rep, practice_status=pr,
        )

    # Tier 3: fallback (aggregate report+practice for pos+body)
    fb = _lookup_fallback(rates, pos, body)
    if fb and fb[0] >= 10:
        n, rate, ssip = fb
        return CohortResult(
            n=n, p_played=rate, snap_share_if_played=ssip,
            cohort_level="fallback",
            body_part=body, position=pos,
            report_status=rep, practice_status=pr,
        )

    # Tier 4: marginal by (report, practice) — body part unknown/rare
    marg_rate = _PLAY_RATE_BY_REPORT_PRACTICE.get((rep, pr))
    if marg_rate is not None:
        return CohortResult(
            n=0, p_played=marg_rate, snap_share_if_played=0.0,
            cohort_level="marginal",
            body_part=body, position=pos,
            report_status=rep, practice_status=pr,
        )

    # Tier 5: report-only baseline
    return CohortResult(
        n=0,
        p_played=_PLAY_RATE_BY_REPORT.get(rep, 0.5),
        snap_share_if_played=0.0,
        cohort_level="marginal",
        body_part=body, position=pos,
        report_status=rep, practice_status=pr,
    )


# ── Convenience: raw cohort dataframe lookup ────────────────────

def find_comparable_cases(position: str, body_part: str,
                          report_status: str | None = None,
                          practice_status: str | None = None
                          ) -> pd.DataFrame:
    """Return the historical injury rows that would form this cohort.
    Useful for inspection / debugging the empirical predict() result."""
    df = load_archive()
    if df.empty:
        return df
    pos = position.upper()
    body = body_part.lower()
    base = df[(df["position"].astype(str).str.upper() == pos)
              & (df["_body_part_bucket"] == body)]
    if report_status:
        base = base[base["_report_code"] == report_status_code(report_status)]
    if practice_status:
        base = base[base["_practice_code"]
                    == practice_status_code(practice_status)]
    return base

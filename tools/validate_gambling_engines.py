"""Validation harness for the Gambling Lab engines.

Runs known scenarios against each of the five engines and prints
a green/red report. The point isn't to validate exact numbers (we
don't have ground-truth labels) — it's to catch regressions:
schema breaks, NaN explosions, sanity violations (rates > 1, deltas
that don't sum to ~0, etc.).

Run with:
    python tools/validate_gambling_engines.py

Exit code 0 if all checks pass, 1 if any fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ── Helpers ──────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
results: list[tuple[bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append((ok, label))
    mark = PASS if ok else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  {mark} {label}{suffix}")


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


# ── 1. Injury cohort engine ──────────────────────────────────────

section("1. Injury Cohort Engine")
from lib_injury_cohort import predict, body_part_normalize, load_cohort_rates

cr = load_cohort_rates()
check("cohort rates table loads",
      not cr.empty, f"{len(cr):,} cohorts")

# Body-part normalizer covers expected fuzzy inputs
for raw, expected in [
    ("Knee", "knee"), ("Right Knee", "knee"), ("ACL", "knee"),
    ("Hamstring", "hamstring"), ("turf toe", "foot"),
    ("concussion", "concussion"), ("shldr", "shoulder"),
    ("NFI", "illness"), ("rest", "illness"), ("", "unknown"),
]:
    got = body_part_normalize(raw)
    check(f"normalize {raw!r} → {expected!r}", got == expected,
          f"got {got!r}")

# Predict() — known sanity scenarios
scenarios = [
    # (description, kwargs, expected_play_rate_range)
    ("OUT/DNP → near 0%",
     dict(position="WR", body_part="knee",
          report_status="OUT", practice_status="DNP"),
     (0.0, 0.05)),
    ("DOUBTFUL → near 0%",
     dict(position="WR", body_part="hamstring",
          report_status="DOUBTFUL", practice_status="DNP"),
     (0.0, 0.10)),
    ("PROBABLE/FULL → ≥85%",
     dict(position="WR", body_part="ankle",
          report_status="PROBABLE", practice_status="FULL"),
     (0.85, 1.01)),
    ("NONE/FULL → ≥80%",
     dict(position="WR", body_part="unknown",
          report_status="NONE", practice_status="FULL"),
     (0.80, 1.01)),
    ("QUESTIONABLE/DNP < 50%",
     dict(position="WR", body_part="hamstring",
          report_status="QUESTIONABLE", practice_status="DNP"),
     (0.10, 0.55)),
    ("QUESTIONABLE/FULL > QUESTIONABLE/DNP",
     None, None),  # composite check below
]
for desc, kwargs, rng in scenarios[:-1]:
    r = predict(**kwargs)
    ok = rng[0] <= r.p_played <= rng[1]
    check(desc, ok,
          f"got p={r.p_played:.3f} (n={r.n}, level={r.cohort_level})")

# Composite ordering check
q_dnp = predict(position="WR", body_part="hamstring",
                report_status="QUESTIONABLE", practice_status="DNP").p_played
q_full = predict(position="WR", body_part="hamstring",
                 report_status="QUESTIONABLE", practice_status="FULL").p_played
check("Q/FULL > Q/DNP (practice gradient)",
      q_full > q_dnp,
      f"FULL={q_full:.3f}  DNP={q_dnp:.3f}")

# Snap-share if played should be in [0, 1]
sample_rows = cr.dropna(subset=["snap_share_if_played"])
check("snap_share_if_played is in [0, 1]",
      ((sample_rows["snap_share_if_played"] >= 0)
       & (sample_rows["snap_share_if_played"] <= 1)).all(),
      f"min={sample_rows['snap_share_if_played'].min():.3f} "
      f"max={sample_rows['snap_share_if_played'].max():.3f}")

# Play rates in [0, 1]
check("play_rate is in [0, 1]",
      ((cr["play_rate"] >= 0) & (cr["play_rate"] <= 1)).all())


# ── 2. Scheme deltas ─────────────────────────────────────────────

section("2. Scheme Deltas")
from lib_scheme_deltas import load_scheme_deltas

sd = load_scheme_deltas()
check("scheme deltas table loads", not sd.empty, f"{len(sd):,} rows")

# Should have 32 teams × N seasons × 2 sides
n_teams = sd["team"].nunique()
check("team count = 32", n_teams == 32, f"got {n_teams}")

n_seasons = sd["season"].nunique()
check("season count >= 8", n_seasons >= 8, f"got {n_seasons}")

# Sides
check("two sides (offense, defense)",
      set(sd["side"].unique()) == {"offense", "defense"})

# Deltas should sum to ~0 within each (season, side, metric) group
for col in ["pass_rate_overall_delta", "blitz_rate_delta"]:
    if col in sd.columns:
        sums = sd.groupby(["season", "side"])[col].sum()
        max_abs = sums.abs().max()
        check(f"{col} sums to ~0 within (season, side)",
              max_abs < 0.05,
              f"max |sum| = {max_abs:.4f}")

# Rate columns in [0, 1]
for col in ["pass_rate_overall", "shotgun_rate", "blitz_rate",
            "man_coverage_rate"]:
    if col in sd.columns:
        vals = sd[col].dropna()
        ok = ((vals >= 0) & (vals <= 1)).all()
        check(f"{col} in [0, 1]", ok,
              f"min={vals.min():.3f} max={vals.max():.3f}")

# Pass rate should be 50-65% for most teams in modern era
modern = sd[(sd["side"] == "offense") & (sd["season"] >= 2020)]
median_pass = modern["pass_rate_overall"].median()
check("modern (2020+) median pass rate in [0.55, 0.62]",
      0.55 <= median_pass <= 0.62, f"median={median_pass:.3f}")


# ── 3. DvP ───────────────────────────────────────────────────────

section("3. Defense vs. Position (DvP)")
DVP = REPO / "data" / "dvp.parquet"
dvp = pd.read_parquet(DVP)
check("DvP table loads", not dvp.empty, f"{len(dvp):,} rows")

# Position groups
check("has WR, TE, RB groups",
      set(dvp["pos_group"].unique()) >= {"WR", "TE", "RB"})

# Per-game stats should be positive
for col in ["rec_yards_pg", "rec_tds_pg", "rush_yards_pg"]:
    if col in dvp.columns:
        vals = dvp[col].dropna()
        ok = (vals >= 0).all()
        check(f"{col} non-negative", ok,
              f"min={vals.min():.3f} max={vals.max():.3f}")

# WR receiving yards / game should be in [80, 220] for most defenses
wr = dvp[dvp["pos_group"] == "WR"].dropna(subset=["rec_yards_pg"])
check("WR rec_yards_pg in [80, 220] for ≥90%",
      ((wr["rec_yards_pg"] >= 80) & (wr["rec_yards_pg"] <= 220)).mean() > 0.90)

# Deltas should sum to ~0 within (season, pos_group)
for col in ["rec_yards_pg_delta"]:
    sums = dvp.groupby(["season", "pos_group"])[col].sum()
    max_abs = sums.abs().max()
    check(f"{col} sums to ~0 within (season, pos_group)",
          max_abs < 5.0,  # raw yards, not rate; relax tolerance
          f"max |sum| = {max_abs:.3f}")


# ── 4. Coaching tendencies ───────────────────────────────────────

section("4. Coaching Tendencies")
COACH = REPO / "data" / "coaching_tendencies.parquet"
coach = pd.read_parquet(COACH)
check("coaching table loads", not coach.empty, f"{len(coach):,} rows")

# 32 teams
check("team count = 32", coach["team"].nunique() == 32)

# Rate cols in [0, 1]
for col in ["fourth_short_go_rate", "two_pt_attempt_rate", "rz_run_rate"]:
    if col in coach.columns:
        vals = coach[col].dropna()
        ok = ((vals >= 0) & (vals <= 1)).all()
        check(f"{col} in [0, 1]", ok,
              f"min={vals.min():.3f} max={vals.max():.3f}")

# DET 2024 should be elite at 4th-and-short
det_2024 = coach[(coach["team"] == "DET") & (coach["season"] == 2024)]
if not det_2024.empty:
    rate = det_2024.iloc[0]["fourth_short_go_rate"]
    check("DET 2024 4th-and-short go rate ≥ 0.70 (Campbell signature)",
          rate >= 0.70, f"got {rate:.3f}")


# ── 5. SGP correlations ──────────────────────────────────────────

section("5. SGP Correlations")
SGP = REPO / "data" / "sgp_correlations.parquet"
sgp = pd.read_parquet(SGP)
check("SGP table loads", not sgp.empty, f"{len(sgp):,} rows")

# Correlations should be in [-1, 1]
corr = sgp["corr_qb_yds_partner_yds"].dropna()
check("corr in [-1, 1]",
      ((corr >= -1.0) & (corr <= 1.0)).all(),
      f"min={corr.min():.3f} max={corr.max():.3f}")

# QB↔WR1 correlations should be mostly positive (most stacks correlate)
wr1 = sgp[(sgp["partner_role"] == "WR1")
          & sgp["corr_qb_yds_partner_yds"].notna()]
pos_pct = (wr1["corr_qb_yds_partner_yds"] > 0).mean()
check("≥80% of QB↔WR1 correlations are positive",
      pos_pct > 0.80, f"got {pos_pct:.1%}")

# Hurts↔Brown 2023 high-correlation sanity
phi_2023 = sgp[(sgp["team"] == "PHI") & (sgp["season"] == 2023)
               & (sgp["partner_role"] == "WR1")]
if not phi_2023.empty:
    c = phi_2023.iloc[0]["corr_qb_yds_partner_yds"]
    check("Hurts↔Brown 2023 corr ≥ 0.70 (textbook stack)",
          c >= 0.70, f"got {c:.3f}")


# ── Summary ──────────────────────────────────────────────────────

section("Summary")
n_total = len(results)
n_pass = sum(1 for ok, _ in results if ok)
n_fail = n_total - n_pass

print(f"  {n_pass} / {n_total} checks passed")
if n_fail:
    print(f"  {n_fail} FAILED:")
    for ok, label in results:
        if not ok:
            print(f"    {FAIL} {label}")
    sys.exit(1)
else:
    print(f"  All green. Engines look healthy.")
    sys.exit(0)

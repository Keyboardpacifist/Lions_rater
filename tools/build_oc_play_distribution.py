"""Build the OC-attributed play-level distribution dataset.

Pre-computes a slim subset of game_pbp.parquet, filtered to plays
attributed to a play-caller in oc_team_seasons.csv, with derived
columns the play-distribution explorer panel needs:
    season, oc_name, posteam, play_type, route, run_gap, run_location,
    pass_location, air_yards, passing_yards, rushing_yards, yards_gained,
    epa, success, down, ydstogo, dnd_bucket, distance_bucket,
    pass_depth_bucket, qtr, quarter_seconds_remaining,
    score_differential, gamestate, was_pressure, pressure_cat,
    number_of_pass_rushers, defense_coverage_type, defense_man_zone_type,
    coverage_simple, offense_personnel, personnel_simple,
    offense_formation, shotgun, no_huddle, formation_simple

Output: data/oc_play_distribution.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
OC_TEAM_SEASONS = REPO / "data" / "scheme" / "curation" / "oc_team_seasons.csv"
OUT = REPO / "data" / "oc_play_distribution.parquet"


def _dnd_bucket(down, ydstogo):
    if pd.isna(down) or pd.isna(ydstogo):
        return None
    d = int(down); y = float(ydstogo)
    if d == 1: return "1st & 10" if y >= 7 else "1st & short"
    if d == 2:
        if y <= 3: return "2nd & ≤3"
        if y <= 7: return "2nd & 4-7"
        return "2nd & 8+"
    if d == 3:
        if y <= 3: return "3rd & ≤3"
        if y <= 7: return "3rd & 4-7"
        return "3rd & 8+"
    if d == 4: return "4th down"
    return None


def _distance_bucket(ydstogo):
    if pd.isna(ydstogo): return None
    y = float(ydstogo)
    if y <= 3: return "Short (≤3)"
    if y <= 7: return "Medium (4-7)"
    return "Long (8+)"


def _pass_depth_bucket(air_yards):
    if pd.isna(air_yards): return None
    ay = float(air_yards)
    if ay < 0: return "Behind LOS"
    if ay < 10: return "Short (0-9)"
    if ay < 20: return "Intermediate (10-19)"
    return "Deep (20+)"


def _gamestate(score_diff):
    if pd.isna(score_diff): return None
    sd = float(score_diff)
    if sd >= 8: return "Leading by 8+"
    if sd <= -8: return "Trailing by 8+"
    return "Tied / one-score"


def _coverage_simple(mz, cov):
    """Combine man/zone label with the cover-shell."""
    mz = str(mz or "").strip()
    cov = str(cov or "").strip().replace("_", " ")
    if mz == "MAN_COVERAGE":
        return f"Man · {cov}" if cov and cov != "nan" else "Man"
    if mz == "ZONE_COVERAGE":
        return f"Zone · {cov}" if cov and cov != "nan" else "Zone"
    return None


def _personnel_simple(personnel):
    if pd.isna(personnel): return None
    p = str(personnel)
    if "1 RB, 1 TE, 3 WR" in p: return "11"
    if "1 RB, 2 TE, 2 WR" in p: return "12"
    if "1 RB, 3 TE, 1 WR" in p: return "13"
    if "2 RB, 1 TE, 2 WR" in p: return "21"
    if "2 RB, 2 TE, 1 WR" in p: return "22"
    if "0 RB, 1 TE, 4 WR" in p: return "10/empty"
    if "0 RB" in p: return "Empty"
    return "Other"


def _formation_simple(shotgun, formation, no_huddle):
    """Coarse formation: Shotgun / Under center / Pistol / Empty / Other."""
    if pd.isna(formation):
        if pd.notna(shotgun) and int(shotgun) == 1:
            return "Shotgun"
        return None
    f = str(formation).upper()
    if "SHOTGUN" in f: return "Shotgun"
    if "PISTOL" in f: return "Pistol"
    if "EMPTY" in f: return "Empty"
    if "I_FORM" in f or "JUMBO" in f or "SINGLEBACK" in f or "UNDER" in f:
        return "Under center"
    return f.title() if f else None


def main() -> None:
    print(f"→ loading PBP {PBP.relative_to(REPO)}")
    pbp = pd.read_parquet(PBP)
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    pbp = pbp.dropna(subset=["posteam", "season", "epa"])
    pbp["season"] = pbp["season"].astype(int)
    print(f"  scrimmage plays: {len(pbp):,}")

    print(f"→ loading {OC_TEAM_SEASONS.relative_to(REPO)}")
    oc_ts = pd.read_csv(OC_TEAM_SEASONS)
    oc_ts = oc_ts[oc_ts["calls_plays"].astype(str).str.upper() == "TRUE"].copy()
    oc_ts["season"] = oc_ts["season"].astype(int)

    pbp = pbp.merge(
        oc_ts[["oc_name", "team", "season"]],
        left_on=["posteam", "season"], right_on=["team", "season"], how="inner",
    )
    print(f"  OC-attributed: {len(pbp):,} ({pbp['oc_name'].nunique()} OCs)")

    # Derive columns
    print("→ deriving filter columns")
    pbp["dnd_bucket"] = pbp.apply(
        lambda r: _dnd_bucket(r["down"], r["ydstogo"]), axis=1)
    pbp["distance_bucket"] = pbp["ydstogo"].apply(_distance_bucket)
    pbp["pass_depth_bucket"] = pbp["air_yards"].apply(_pass_depth_bucket)
    pbp["gamestate"] = pbp["score_differential"].apply(_gamestate)
    pbp["pressure_cat"] = pbp["was_pressure"].apply(
        lambda x: "Pressured" if pd.notna(x) and float(x) >= 1 else
                  ("Clean" if pd.notna(x) else None))
    pbp["coverage_simple"] = pbp.apply(
        lambda r: _coverage_simple(r.get("defense_man_zone_type"),
                                     r.get("defense_coverage_type")), axis=1)
    pbp["personnel_simple"] = pbp["offense_personnel"].apply(_personnel_simple)
    pbp["formation_simple"] = pbp.apply(
        lambda r: _formation_simple(r.get("shotgun"), r.get("offense_formation"),
                                      r.get("no_huddle")), axis=1)
    pbp["success"] = pbp.get("success", (pbp["epa"] > 0).astype(int))

    # Output schema — keep columns we actually use
    keep = [
        "oc_name", "posteam", "season", "play_type",
        "route", "run_gap", "run_location", "pass_location",
        "air_yards", "passing_yards", "rushing_yards",
        "epa", "success", "down", "ydstogo",
        "dnd_bucket", "distance_bucket", "pass_depth_bucket",
        "qtr", "quarter_seconds_remaining", "score_differential",
        "gamestate", "was_pressure", "pressure_cat",
        "number_of_pass_rushers",
        "defense_coverage_type", "defense_man_zone_type", "coverage_simple",
        "offense_personnel", "personnel_simple",
        "offense_formation", "shotgun", "no_huddle", "formation_simple",
    ]
    keep = [c for c in keep if c in pbp.columns]
    out = pbp[keep].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"✓ wrote {OUT.relative_to(REPO)}  rows={len(out):,}  cols={len(out.columns)}")

    # Spot check
    print()
    print("=== Sample distributions for Sean Payton, Career ===")
    sp = out[out["oc_name"] == "Sean Payton"]
    print(f"Total plays: {len(sp):,}")
    print()
    print("Pass routes (top 10):")
    pass_only = sp[sp["play_type"] == "pass"]
    print(pass_only["route"].value_counts(dropna=False).head(10).to_string())
    print()
    print("Run gaps:")
    run_only = sp[sp["play_type"] == "run"]
    print(run_only["run_gap"].value_counts(dropna=False).to_string())
    print()
    print("Coverage faced (top 8):")
    print(sp["coverage_simple"].value_counts().head(8).to_string())


if __name__ == "__main__":
    main()

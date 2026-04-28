#!/usr/bin/env python3
"""
Per-dropback feed for the QB panel — one row per pass attempt, sack,
or scramble. Joins nfl_participation for per-play coverage type,
man/zone, real pressure flag, time-to-throw, pass rushers, defenders
in box, offensive personnel/formation, and the targeted route.

Foundation for every QB split: pressure, situational, elite-vs-weak
competition, and the toggleable contextual throw map.

Output: data/qb_dropbacks.parquet

Run:
    python tools/build_qb_dropbacks.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "data" / "qb_dropbacks.parquet"

# Match the QB rater's season range
SEASONS = list(range(2016, 2026))

KEEP_COLS = [
    # Identification
    "game_id", "play_id", "season", "week", "season_type",
    "passer_player_id", "passer_player_name",
    "posteam", "defteam",
    # Pre-snap context
    "qtr", "quarter_seconds_remaining", "game_seconds_remaining",
    "down", "ydstogo", "yardline_100", "score_differential",
    "shotgun", "no_huddle", "play_clock", "goal_to_go",
    # Outcome
    "play_type", "pass_attempt", "sack", "complete_pass",
    "interception", "fumble_lost", "qb_hit", "qb_scramble",
    "pass_location", "pass_length", "air_yards",
    "yards_after_catch", "passing_yards",
    "epa", "wpa", "success",
    "receiver_player_id",
]


PARTICIPATION_COLS = [
    "nflverse_game_id", "play_id",
    "defense_coverage_type", "defense_man_zone_type",
    "number_of_pass_rushers", "defenders_in_box",
    "offense_personnel", "offense_formation",
    "time_to_throw", "was_pressure", "route",
]


def main() -> None:
    import nflreadpy as nfl

    print(f"Pulling pbp for {SEASONS[0]}-{SEASONS[-1]}…")
    pbp = nfl.load_pbp(SEASONS).to_pandas()
    print(f"  {len(pbp):,} total plays loaded")

    print(f"Pulling participation for {SEASONS[0]}-{SEASONS[-1]}…")
    part = nfl.load_participation(SEASONS).to_pandas()
    print(f"  {len(part):,} participation rows loaded")

    # Slim participation to what we need before merging
    part_slim = part[[c for c in PARTICIPATION_COLS if c in part.columns]].copy()
    part_slim = part_slim.rename(columns={"nflverse_game_id": "game_id"})

    # Dropback = pass attempt OR sack OR scramble (with a real passer)
    is_dropback = (
        (pbp["pass_attempt"] == 1)
        | (pbp["sack"] == 1)
        | (pbp.get("qb_scramble", 0) == 1)
    )
    has_passer = pbp["passer_player_id"].notna()
    db = pbp[is_dropback & has_passer].copy()
    print(f"  {len(db):,} dropbacks")

    keep = [c for c in KEEP_COLS if c in db.columns]
    db = db[keep].reset_index(drop=True)

    # Join participation context onto each dropback
    print("Joining participation onto dropbacks…")
    db = db.merge(part_slim, on=["game_id", "play_id"], how="left")
    matched = db["defense_coverage_type"].notna().sum()
    print(f"  {matched:,} of {len(db):,} dropbacks have coverage data "
          f"({matched/len(db)*100:.0f}%)")

    # Pre-computed split flags (faster than recomputing in the UI on every render)
    db["is_third_down"] = (db["down"] == 3).fillna(False)
    db["is_red_zone"] = (db["yardline_100"] <= 20).fillna(False)
    db["is_fourth_quarter"] = (db["qtr"] == 4).fillna(False)
    db["is_two_minute"] = (
        ((db["qtr"] == 2) | (db["qtr"] == 4))
        & (db["quarter_seconds_remaining"] <= 120)
    ).fillna(False)
    # Real per-play pressure flag (replaces the qb_hit proxy). Falls back
    # to qb_hit for plays without participation data (mostly pre-2017 or
    # early-week edge cases).
    db["is_pressured"] = db["was_pressure"].fillna(
        (db.get("qb_hit", 0) == 1)
    ).astype(bool)
    db["is_trailing"] = (db["score_differential"] < 0).fillna(False)
    db["is_one_score"] = (db["score_differential"].abs() <= 8).fillna(False)

    # Personnel grouping — collapse the verbose string into the standard
    # "11", "12", "21", "Heavy" buckets that fans recognize.
    def _group_personnel(s: str | None) -> str | None:
        if not isinstance(s, str) or not s:
            return None
        # Examples: "1 RB, 1 TE, 3 WR" or "1 C, 1 QB, 1 RB, 4 T, 1 TE, 3 WR"
        rb = te = wr = 0
        for token in s.split(","):
            token = token.strip()
            if " RB" in token:
                rb = int(token.split(" ")[0]) if token.split(" ")[0].isdigit() else 0
            elif " TE" in token:
                te = int(token.split(" ")[0]) if token.split(" ")[0].isdigit() else 0
            elif " WR" in token:
                wr = int(token.split(" ")[0]) if token.split(" ")[0].isdigit() else 0
        if rb == 0:
            return "Empty"
        if rb == 1 and te == 1 and wr == 3:
            return "11"
        if rb == 1 and te == 2 and wr == 2:
            return "12"
        if rb == 2 and te == 1 and wr == 2:
            return "21"
        if te >= 3 or rb >= 2:
            return "Heavy"
        return f"{rb}{te}"  # fallback label
    db["personnel_group"] = db["offense_personnel"].apply(_group_personnel)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    db.to_parquet(OUTPUT, index=False)
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\n✓ wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(db):,} rows × {db.shape[1]} cols ({size_mb:.1f} MB)")
    print(f"  Seasons: {sorted(int(s) for s in db['season'].dropna().unique())}")
    print(f"  Unique passers: {db['passer_player_id'].nunique():,}")


if __name__ == "__main__":
    main()

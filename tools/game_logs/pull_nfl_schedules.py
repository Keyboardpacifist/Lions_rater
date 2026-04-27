#!/usr/bin/env python3
"""
NFL schedule pull — venue, surface, weather, betting lines, etc.

Light wrapper around nflreadpy.load_schedules. Output joins cleanly
with player game logs on (season, week, team) — used by the splits
explorer to filter games by roof / surface / weather / divisional.

Output:
  data/games/nfl_schedules.parquet  — one row per game

Usage:
    python tools/game_logs/pull_nfl_schedules.py
    python tools/game_logs/pull_nfl_schedules.py --seasons 2024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nflreadpy as nfl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = REPO_ROOT / "data" / "games" / "nfl_schedules.parquet"
DEFAULT_SEASONS = list(range(2016, 2026))


def parse_seasons(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x.strip()) for x in s.split(",")]
    return [int(s)]


def main(seasons: list[int]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Pulling NFL schedules for {seasons[0]}–{seasons[-1]}…")
    df = nfl.load_schedules(seasons=seasons)
    df.write_parquet(OUT)
    print(f"  ✓ {OUT.name}  →  {df.shape[0]:,} rows × {df.shape[1]} cols")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", default=None,
                   help="Year(s): '2024', '2016-2025'. Defaults to 2016-2025.")
    args = p.parse_args()
    seasons = parse_seasons(args.seasons) if args.seasons else DEFAULT_SEASONS
    try:
        main(seasons)
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

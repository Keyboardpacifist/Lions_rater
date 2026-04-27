#!/usr/bin/env python3
"""
NFL participation pull — personnel groupings + formation per play.

Adds the on-field complement of the PBP feed: which players were on
the field, how the offense lined up (11/12/21 personnel, shotgun vs
under-center), and what the defense countered with (4-3 base, nickel,
dime, big nickel). Combined with PBP, this is enough to derive
team-level defensive scheme tendencies free of any PFF data.

NOTE: NFL participation data is only available 2016+ from nflverse.

Output:
  data/games/nfl_participation.parquet — one row per play

Usage:
    python tools/game_logs/pull_nfl_participation.py
    python tools/game_logs/pull_nfl_participation.py --seasons 2024
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nflreadpy as nfl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT = REPO_ROOT / "data" / "games" / "nfl_participation.parquet"
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
    t0 = time.time()
    print(f"Pulling NFL participation for {seasons[0]}–{seasons[-1]}…")
    df = nfl.load_participation(seasons=seasons)
    df.write_parquet(OUT)
    mb = OUT.stat().st_size / (1024 * 1024)
    print(f"  ✓ {OUT.name}  →  {df.shape[0]:,} plays × {df.shape[1]} cols "
          f"({mb:.1f} MB)  ·  {(time.time()-t0):.1f}s")


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

#!/usr/bin/env python3
"""
CFBD enrichment CLI.

Usage:
    python tools/cfbd_data_pull.py --seasons 2024
    python tools/cfbd_data_pull.py --seasons 2014-2025
    python tools/cfbd_data_pull.py --seasons 2024 --verbose

Output: data/college/college_<pos>_cfbd_advanced.parquet for QB/WR/TE/RB.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from cfbd_pipeline import runner


def parse_seasons(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x.strip()) for x in s.split(",")]
    return [int(s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True)
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--targets-only", action="store_true",
                    help="Skip the stats/PPA/usage pull and only do the playText "
                         "targets parsing pass (assumes parquets already exist).")
    ap.add_argument("--no-targets", action="store_true",
                    help="Skip the playText targets parsing pass (faster).")
    args = ap.parse_args()

    seasons = parse_seasons(args.seasons)
    verbose = not args.quiet

    if not args.targets_only:
        print(f"CFBD enrichment for seasons: {seasons}")
        df = runner.pull_seasons(seasons, verbose=verbose)
        if df.empty:
            print("No data — exiting")
            return
        print(f"\nCombined: {len(df)} player-seasons across {len(seasons)} season(s)")
        runner.write_position_files(df, verbose=verbose)

    if not args.no_targets:
        print(f"\nTargets backfill (playText parsing) for seasons: {seasons}")
        from cfbd_pipeline import targets
        targets.backfill(seasons, verbose=verbose)

    print("\nDone.")


if __name__ == "__main__":
    main()

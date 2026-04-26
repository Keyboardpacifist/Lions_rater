#!/usr/bin/env python3
"""
NFL game-log pull — every NFL player's per-week box score, 2016-2025.

Pulls four buckets from nflverse via nflreadpy and saves each as a
separate parquet under data/games/. Keeping them separate (rather
than one mega-table) means each file stays inspectable in isolation
and adding/removing a source later doesn't disturb the others.

Files written:
  data/games/nfl_weekly_stats.parquet      — player_stats (offense + defense)
  data/games/nfl_weekly_snaps.parquet      — snap counts
  data/games/nfl_weekly_ngs_passing.parquet
  data/games/nfl_weekly_ngs_rushing.parquet
  data/games/nfl_weekly_ngs_receiving.parquet
  data/games/nfl_weekly_pfr_pass.parquet
  data/games/nfl_weekly_pfr_rush.parquet
  data/games/nfl_weekly_pfr_rec.parquet
  data/games/nfl_weekly_pfr_def.parquet

Usage:
    python tools/game_logs/pull_nfl_weekly.py
    python tools/game_logs/pull_nfl_weekly.py --seasons 2024
    python tools/game_logs/pull_nfl_weekly.py --seasons 2016-2025
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nflreadpy as nfl
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "data" / "games"
DEFAULT_SEASONS = list(range(2016, 2026))


def parse_seasons(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x.strip()) for x in s.split(",")]
    return [int(s)]


def _save(df: pl.DataFrame, name: str) -> None:
    path = OUT_DIR / name
    df.write_parquet(path)
    print(f"  ✓ {name}  →  {df.shape[0]:>7,} rows × {df.shape[1]:>3} cols")


def main(seasons: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pulling NFL game logs for seasons: {seasons[0]}–{seasons[-1]}")
    print(f"Output dir: {OUT_DIR}\n")

    t0 = time.time()

    print("[1/4] player_stats (weekly box scores)")
    df = nfl.load_player_stats(seasons=seasons, summary_level="week")
    _save(df, "nfl_weekly_stats.parquet")

    print("\n[2/4] snap_counts")
    df = nfl.load_snap_counts(seasons=seasons)
    _save(df, "nfl_weekly_snaps.parquet")

    print("\n[3/4] next-gen stats (passing / rushing / receiving)")
    for stype in ("passing", "rushing", "receiving"):
        df = nfl.load_nextgen_stats(seasons=seasons, stat_type=stype)
        _save(df, f"nfl_weekly_ngs_{stype}.parquet")

    # PFR weekly advanced stats only exist 2018+; clip the requested
    # seasons to that window so older years don't blow up the whole
    # PFR pull.
    PFR_MIN = 2018
    pfr_seasons = [s for s in seasons if s >= PFR_MIN]
    if not pfr_seasons:
        print("\n[4/4] PFR advanced — skipped (no requested seasons ≥ 2018)")
    else:
        print(f"\n[4/4] PFR advanced (pass / rush / rec / def) — "
              f"{pfr_seasons[0]}–{pfr_seasons[-1]}")
        for stype in ("pass", "rush", "rec", "def"):
            try:
                df = nfl.load_pfr_advstats(
                    seasons=pfr_seasons, stat_type=stype,
                    summary_level="week",
                )
                _save(df, f"nfl_weekly_pfr_{stype}.parquet")
            except Exception as e:
                print(f"  ⚠️  pfr_{stype} skipped: {e}")

    print(f"\n✅ done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", default=None,
                   help="Year(s): '2024', '2016-2025', '2020,2024'. "
                        "Defaults to 2016-2025.")
    args = p.parse_args()
    seasons = parse_seasons(args.seasons) if args.seasons else DEFAULT_SEASONS
    try:
        main(seasons)
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

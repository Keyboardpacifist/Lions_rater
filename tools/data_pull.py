#!/usr/bin/env python3
"""
NFL Rater — Unified Data Pipeline CLI.

Usage:
    python tools/data_pull.py --position wr --seasons 2024
    python tools/data_pull.py --position wr --seasons 2016-2025
    python tools/data_pull.py --position wr --seasons 2024 --dry-run
    python tools/data_pull.py --position wr --seasons 2024 --verbose

Run from the repo root. Outputs go to data/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root to path so pipeline package can be imported
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from pipeline.positions import POSITIONS
from pipeline.runner import run_pipeline


def parse_seasons(s: str) -> list[int]:
    """Parse season argument: '2024' or '2016-2025' or '2020,2021,2024'."""
    if "-" in s:
        start, end = s.split("-", 1)
        return list(range(int(start), int(end) + 1))
    if "," in s:
        return [int(x.strip()) for x in s.split(",")]
    return [int(s)]


def main():
    parser = argparse.ArgumentParser(
        description="NFL Rater data pipeline — pull, aggregate, z-score, write.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available positions: {', '.join(sorted(POSITIONS.keys()))}",
    )
    parser.add_argument(
        "--position",
        required=True,
        choices=sorted(POSITIONS.keys()),
        help="Position to pull data for.",
    )
    parser.add_argument(
        "--seasons",
        required=True,
        help="Season(s): '2024', '2016-2025', or '2020,2021,2024'.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data"),
        help="Output directory (default: data/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress (default: on).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args()

    seasons = parse_seasons(args.seasons)
    config = POSITIONS[args.position]
    output_dir = Path(args.output_dir)
    verbose = not args.quiet

    run_pipeline(
        config=config,
        seasons=seasons,
        output_dir=output_dir,
        dry_run=args.dry_run,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()

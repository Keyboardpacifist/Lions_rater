#!/usr/bin/env python3
"""Pull NFL draft picks + active contracts from nflverse.

Outputs:
  data/nfl_draft_picks.parquet  — every drafted player, all years
  data/nfl_contracts.parquet    — active contracts (OverTheCap)

Used by lib_player_blurb to write fan-facing player blurbs:
  • Draft slot value sentence ("Pick 84, playing like a top-15 RB")
  • FA contract value sentence  ("Signed for $4.2M APY, ranks #6 by score")

Usage:
  venv/bin/python tools/pull_draft_contracts.py
"""
from __future__ import annotations

from pathlib import Path

import nflreadpy

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data"

DRAFT_OUT = OUT / "nfl_draft_picks.parquet"
CONTRACT_OUT = OUT / "nfl_contracts.parquet"


def main() -> None:
    print("→ pulling draft picks...")
    draft = nflreadpy.load_draft_picks()
    if hasattr(draft, "to_pandas"):
        draft = draft.to_pandas()
    print(f"  {len(draft):,} rows · "
          f"{draft['season'].min()}-{draft['season'].max()}")
    OUT.mkdir(parents=True, exist_ok=True)
    draft.to_parquet(DRAFT_OUT, index=False)
    print(f"  ✓ wrote {DRAFT_OUT.relative_to(REPO)}")

    print("→ pulling contracts (OverTheCap)...")
    contracts = nflreadpy.load_contracts()
    if hasattr(contracts, "to_pandas"):
        contracts = contracts.to_pandas()
    print(f"  {len(contracts):,} rows")
    contracts.to_parquet(CONTRACT_OUT, index=False)
    print(f"  ✓ wrote {CONTRACT_OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

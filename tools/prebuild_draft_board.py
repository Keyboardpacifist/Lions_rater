"""Pre-compute the 2027 draft board so the live page loads instantly.

The Draft page calls `attach_nfl_comps()` on every render — a ~18s
similarity-and-scoring loop across 100+ prospects. The function IS
cached via @st.cache_data, but cold-start (every fresh deploy or
after cache eviction) hits the full cost AND the user is told the
page "needs to be instantaneous or we won't have customers."

Fix: bake the same output into a parquet that ships in `data/`. The
live page reads the parquet first (cheap) and only falls back to
the live function if the file is missing or stale.

Output: data/draft/draft_2027_board_prebuilt.parquet
        data/draft/draft_2027_board_prebuilt.signature.json

The signature file stores the consensus board's (rank, player,
school) tuple list so the page can detect that the seed was edited
and the prebuilt is now stale (then live-recompute as fallback).

Run after any change to:
  - data/college/draft_2027_consensus.parquet (the seed board)
  - lib_nfl_comps.py / lib_draft_athleticism.py (scoring code)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Stub Streamlit so importing the lib doesn't blow up outside the app.
import streamlit as st


def _passthrough(*a, **kw):
    if a and callable(a[0]):
        return a[0]
    def deco(fn):
        return fn
    return deco


st.cache_data = _passthrough  # type: ignore[assignment]
st.cache_resource = _passthrough  # type: ignore[assignment]

import pandas as pd  # noqa: E402

from lib_draft_2027 import (  # noqa: E402
    attach_nfl_comps,
    load_consensus_board,
)

OUT_DIR = REPO / "data" / "draft"
OUT_PATH = OUT_DIR / "draft_2027_board_prebuilt.parquet"
SIG_PATH = OUT_DIR / "draft_2027_board_prebuilt.signature.json"


def main() -> None:
    print("→ loading consensus board...")
    consensus = load_consensus_board()
    if consensus.empty:
        print("✗ Consensus board not loaded — run "
              "tools/seed_draft_2027_consensus.py first.")
        sys.exit(1)
    print(f"  consensus rows: {len(consensus)}")

    sig = [
        [int(r["expert_rank"]), str(r["player"]), str(r["school"])]
        for _, r in consensus.iterrows()
    ]

    print("→ computing NFL comps + athletic scores "
          "(this is the slow part)...")
    t0 = time.time()
    df = attach_nfl_comps(tuple(map(tuple, sig)))
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s ({len(df)} rows)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # `nfl_comps` is a list-of-dicts column. PyArrow handles it fine,
    # but to be safe we serialize to JSON strings — round-trip on the
    # read side is one json.loads call. Same for components dicts.
    serial_cols = [
        "nfl_comps", "pedigree_components", "tested_components",
        "contextual_components", "strengths", "concerns",
    ]
    out = df.copy()
    for col in serial_cols:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda v: json.dumps(v, default=str)
                if v is not None else None
            )

    out.to_parquet(OUT_PATH, index=False)
    print(f"  ✓ wrote {OUT_PATH.relative_to(REPO)} "
          f"({OUT_PATH.stat().st_size / 1024:.1f} KB)")

    SIG_PATH.write_text(json.dumps({
        "signature": sig,
        "n_rows": len(df),
        "built_at": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
    }, indent=2))
    print(f"  ✓ wrote {SIG_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()

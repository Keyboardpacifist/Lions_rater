#!/usr/bin/env python3
"""
One-off fixer — drops misclassified rows from CB and Safety parquets.

The externally-built CB and Safety league parquets used a custom
"pos_group" label that grouped ALL defensive backs together regardless
of CB/S role. That dumped Khalil Dorsey (a CB) into the safety pool,
plus 250+ other CBs into S and 80+ safeties into CB.

Fix: keep rows whose actual role is correct. The complication is that
nflverse sometimes tags DBs generically as `position="DB"` rather than
CB/FS/SS specifically — Surtain II is tagged "DB" but is a CB. So we
disambiguate ambiguous "DB" rows via `depth_chart_position` from
rosters_weekly, which IS specific (CB / NCB / SCB / LCB / RCB / FS / SS).

Run:
    python tools/fix_db_position_misclassification.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# What depth_chart_position values map to CB vs Safety.
CB_DEPTH_LABELS = {"CB", "NCB", "SCB", "LCB", "RCB", "DB"}
SAFETY_DEPTH_LABELS = {"FS", "SS", "S", "SAF"}


def _load_depth_chart_lookup(seasons: list[int]) -> pd.DataFrame:
    """Build a (player_id, season) → primary depth_chart_position lookup."""
    import nflreadpy as nfl
    print(f"  Pulling rosters for {len(seasons)} seasons…")
    ros = nfl.load_rosters_weekly(seasons).to_pandas()
    if "depth_chart_position" not in ros.columns:
        return pd.DataFrame()
    # Most-frequent depth_chart_position per (player_id, season)
    pid_col = "gsis_id" if "gsis_id" in ros.columns else "player_id"
    grouped = (ros.dropna(subset=[pid_col, "depth_chart_position"])
                  .groupby([pid_col, "season"])["depth_chart_position"]
                  .agg(lambda s: s.value_counts().idxmax())
                  .reset_index()
                  .rename(columns={pid_col: "player_id",
                                    "depth_chart_position": "_dc_pos",
                                    "season": "season_year"}))
    return grouped


def fix_pool(parquet_name: str,
             primary_positions: set[str],
             ambiguous_position: str = "DB",
             allowed_depth_for_pool: set[str] = None) -> None:
    """Filter the parquet to rows that belong in this pool.

    Logic:
      - Keep rows where `position` ∈ primary_positions (clear membership).
      - For rows where `position == ambiguous_position` (e.g. "DB"),
        keep them ONLY if their depth_chart_position is in
        allowed_depth_for_pool.
      - Drop everything else.
    """
    path = DATA_DIR / parquet_name
    if not path.exists():
        print(f"⚠️  Missing: {path}")
        return
    df = pd.read_parquet(path)
    if "position" not in df.columns:
        print(f"⚠️  {parquet_name}: no `position` column — skipping")
        return

    before = len(df)

    # 1. Clear-cut keeps
    primary_mask = df["position"].isin(primary_positions)

    # 2. Ambiguous DB-tagged rows — disambiguate via depth chart
    ambiguous_mask = (df["position"] == ambiguous_position)
    if ambiguous_mask.any() and allowed_depth_for_pool:
        seasons = sorted(int(s) for s in df["season_year"].dropna().unique())
        dc = _load_depth_chart_lookup(seasons)
        df = df.merge(dc, on=["player_id", "season_year"], how="left")
        keep_dc_mask = ambiguous_mask & df["_dc_pos"].isin(allowed_depth_for_pool)
        kept = df[primary_mask | keep_dc_mask].drop(columns=["_dc_pos"])
    else:
        kept = df[primary_mask].copy()

    # Drop exact-duplicate rows that accumulate when this script (or
    # the augmenter) is re-run. Without this, repeated invocations
    # silently double the row count for affected players.
    pre_dedupe = len(kept)
    kept = kept.drop_duplicates().reset_index(drop=True)
    deduped = pre_dedupe - len(kept)

    after = len(kept)
    dropped = before - after

    print(f"\n→ {parquet_name}")
    print(f"   {before:,} rows → {after:,} rows ({dropped:,} dropped, "
          f"{deduped:,} of those were exact duplicates)")
    print(f"   Primary positions kept: {sorted(primary_positions)}")
    if allowed_depth_for_pool:
        print(f"   Ambiguous '{ambiguous_position}' rows kept when "
              f"depth_chart_position ∈ {sorted(allowed_depth_for_pool)}")

    kept.to_parquet(path, index=False)
    print(f"   ✓ wrote {after:,} rows")


def main() -> None:
    print("Fixing position misclassification in CB and Safety parquets…")

    # CB pool: position=CB clearly belongs. Position=DB ambiguous —
    # keep only if depth_chart_position is CB-flavored (NCB / SCB / etc).
    fix_pool(
        "league_cb_all_seasons.parquet",
        primary_positions={"CB"},
        allowed_depth_for_pool=CB_DEPTH_LABELS,
    )

    # Safety pool: explicit S / FS / SS / SAF clearly belong.
    # Position=DB rows kept only when depth_chart_position is safety-flavored.
    fix_pool(
        "league_s_all_seasons.parquet",
        primary_positions={"S", "FS", "SS", "SAF"},
        allowed_depth_for_pool=SAFETY_DEPTH_LABELS,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
One-off augmenter — adds PFR coverage stats to existing CB and Safety
league parquets without rebuilding the full pipeline.

The CB and S league parquets are externally-generated artifacts (Colab),
so this is the pragmatic path to surface PFR's coverage-quality data:

    targets_per_game           # how often opponents picked on them
    completion_pct_allowed     # catch rate when targeted
    yards_per_target_allowed   # Y/Tgt allowed
    passer_rating_allowed      # the classic CB stat
    avg_depth_of_target        # tested deep or short
    missed_tackle_pct          # tackle reliability

Plus z-scored versions (with appropriate inversion for "lower is better").

Run:
    python tools/augment_db_with_pfr_coverage.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# stats added to both CB and S
NEW_COVERAGE_STATS = [
    "targets_per_game",
    "completion_pct_allowed",
    "yards_per_target_allowed",
    "passer_rating_allowed",
    "avg_depth_of_target",
    "missed_tackle_pct",
]
# direction-aware: True = lower is better → invert z-score
INVERTED_COVERAGE_STATS = {
    "completion_pct_allowed": True,
    "yards_per_target_allowed": True,
    "passer_rating_allowed": True,
    "missed_tackle_pct": True,
    # not inverted: targets_per_game, avg_depth_of_target
}


def _load_pfr_def_seasons(seasons: list[int]) -> pd.DataFrame:
    """Pull PFR season-level def stats for the given seasons. Returns
    a DataFrame indexed by (pfr_id, season)."""
    import nflreadpy as nfl

    frames = []
    for s in seasons:
        try:
            df = nfl.load_pfr_advstats(
                [s], stat_type="def", summary_level="season"
            ).to_pandas()
            frames.append(df)
            print(f"  PFR def {s}: {len(df):,} rows")
        except Exception as e:
            print(f"  PFR def {s}: FAILED ({e})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def augment_position(
    parquet_path: Path,
    invert_extra: list[str] | None = None,
) -> int:
    """Augment a defensive league parquet with PFR coverage stats."""
    df = pd.read_parquet(parquet_path)
    if "pfr_player_id" not in df.columns:
        print(f"⚠️  {parquet_path.name}: no pfr_player_id column — skipping")
        return 0

    seasons = sorted(int(s) for s in df["season_year"].dropna().unique())
    print(f"\n→ {parquet_path.name}: {len(df):,} rows, seasons {seasons[0]}-{seasons[-1]}")

    # 1. Pull PFR def stats for each season
    print("  Pulling PFR def season-level stats…")
    pfr = _load_pfr_def_seasons(seasons)
    if pfr.empty:
        print("  ⚠️  No PFR data — leaving parquet unchanged")
        return 0

    # 2. Compute per-game versions where appropriate, raw passthroughs elsewhere.
    pfr["games_played_pfr"] = pfr.get("g", 1).fillna(1).replace(0, 1)
    pfr["targets_per_game"] = pfr["tgt"] / pfr["games_played_pfr"]
    # cmp_percent and rat are already pct/rating values, not counts
    pfr["completion_pct_allowed"] = pfr["cmp_percent"]
    pfr["yards_per_target_allowed"] = pfr["yds_tgt"]
    pfr["passer_rating_allowed"] = pfr["rat"]
    pfr["avg_depth_of_target"] = pfr["dadot"]
    pfr["missed_tackle_pct"] = pfr["m_tkl_percent"]

    pfr_slim = pfr[
        ["pfr_id", "season"] + NEW_COVERAGE_STATS
    ].rename(columns={"pfr_id": "pfr_player_id", "season": "season_year"})

    # 3. Drop existing copies of the new cols (re-run safety) then merge
    drop_cols = [c for c in NEW_COVERAGE_STATS if c in df.columns]
    drop_cols += [f"{c}_z" for c in NEW_COVERAGE_STATS if f"{c}_z" in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df = df.merge(pfr_slim, on=["pfr_player_id", "season_year"], how="left")

    # 4. Z-score within the parquet's position population, per season,
    # so a 2024 nickel CB is z-scored against 2024 CBs (not 2018 CBs).
    invert = dict(INVERTED_COVERAGE_STATS)
    for stat in NEW_COVERAGE_STATS:
        if stat not in df.columns:
            continue
        # All-seasons pool z-score (matches the existing parquet's z-scoring
        # convention — it uses the multi-season pool, not per-season).
        s = df[stat].astype(float)
        mu = s.mean()
        sigma = s.std(ddof=0)
        if sigma > 0:
            z = (s - mu) / sigma
            if invert.get(stat, False):
                z = -z
        else:
            z = pd.Series(np.nan, index=s.index)
        df[f"{stat}_z"] = z

    # 5. Write back
    df.to_parquet(parquet_path, index=False)
    print(f"  ✓ wrote {len(df):,} rows × {df.shape[1]} cols")

    # Sanity: how many got matched?
    matched = df[NEW_COVERAGE_STATS[0]].notna().sum()
    print(f"  Matched PFR coverage data on {matched:,} of {len(df):,} rows "
          f"({matched/len(df)*100:.0f}%)")
    return len(df)


def main() -> None:
    print("Augmenting defensive league parquets with PFR coverage stats…")
    for filename in ("league_cb_all_seasons.parquet", "league_s_all_seasons.parquet"):
        path = DATA_DIR / filename
        if not path.exists():
            print(f"⚠️  Missing: {path}")
            continue
        augment_position(path)
    print("\nDone.")


if __name__ == "__main__":
    main()

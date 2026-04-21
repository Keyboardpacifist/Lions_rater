"""
Cached nflverse data pulls.

Each function downloads once per season and caches to .data_cache/
with a 7-day TTL. Only one season is loaded at a time to keep memory
under ~500MB.

All functions return pandas DataFrames. If a source fails (e.g., NGS
not available for older seasons), they return an empty DataFrame and
print a warning — never raise.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(".data_cache")
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


def _cache_path(name: str, season: int) -> Path:
    """Return the cache file path for a given source + season."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_{season}.parquet"


def _is_fresh(path: Path) -> bool:
    """Check if a cached file exists and is within TTL."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < CACHE_TTL_SECONDS


def _load_cached(path: Path) -> pd.DataFrame | None:
    """Load from cache if fresh, else return None."""
    if _is_fresh(path):
        return pd.read_parquet(path)
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to cache."""
    df.to_parquet(path, index=False)


def _pull_with_cache(
    name: str,
    season: int,
    pull_fn,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generic pull-with-cache wrapper.

    Args:
        name: Cache key name (e.g., "pbp", "snap_counts").
        season: NFL season year.
        pull_fn: Callable that returns a pandas DataFrame.
        verbose: Print progress messages.

    Returns:
        DataFrame from cache or fresh pull. Empty DataFrame on failure.
    """
    path = _cache_path(name, season)

    cached = _load_cached(path)
    if cached is not None:
        if verbose:
            print(f"  {name} {season}: loaded from cache ({len(cached):,} rows)")
        return cached

    try:
        if verbose:
            print(f"  {name} {season}: pulling from nflverse...")
        df = pull_fn()
        _save_cache(df, path)
        if verbose:
            print(f"  {name} {season}: {len(df):,} rows cached")
        return df
    except Exception as e:
        if verbose:
            print(f"  {name} {season}: FAILED ({e}) — returning empty DataFrame")
        return pd.DataFrame()


# ── Public API ───────────────────────────────────────────────────────────────


def load_pbp(season: int, verbose: bool = True) -> pd.DataFrame:
    """Load play-by-play data for a single season."""
    import nflreadpy as nfl

    return _pull_with_cache(
        "pbp",
        season,
        lambda: nfl.load_pbp([season]).to_pandas(),
        verbose=verbose,
    )


def load_snap_counts(season: int, verbose: bool = True) -> pd.DataFrame:
    """Load per-game snap counts for a single season."""
    import nflreadpy as nfl

    return _pull_with_cache(
        "snap_counts",
        season,
        lambda: nfl.load_snap_counts([season]).to_pandas(),
        verbose=verbose,
    )


def load_rosters(season: int, verbose: bool = True) -> pd.DataFrame:
    """Load roster data for a single season."""
    import nflreadpy as nfl

    return _pull_with_cache(
        "rosters",
        season,
        lambda: nfl.load_rosters([season]).to_pandas(),
        verbose=verbose,
    )


def load_ngs(
    season: int, stat_type: str, verbose: bool = True
) -> pd.DataFrame:
    """Load NFL Next Gen Stats for a single season.

    Args:
        season: NFL season year.
        stat_type: "receiving", "rushing", or "passing".

    Returns:
        Season-aggregate NGS rows (week == 0). Empty DataFrame if unavailable.
    """
    import nflreadpy as nfl

    def _pull():
        raw = nfl.load_nextgen_stats([season], stat_type=stat_type).to_pandas()
        # Filter to season aggregates
        if "week" in raw.columns:
            return raw[raw["week"] == 0].copy()
        return raw

    return _pull_with_cache(
        f"ngs_{stat_type}",
        season,
        _pull,
        verbose=verbose,
    )


def load_pfr(
    season: int, stat_type: str, verbose: bool = True
) -> pd.DataFrame:
    """Load PFR advanced stats for a single season.

    Args:
        season: NFL season year.
        stat_type: "rush", "rec", "def", "pass".

    Returns:
        Season-level PFR advanced stats. Empty DataFrame if unavailable.
    """
    import nflreadpy as nfl

    return _pull_with_cache(
        f"pfr_{stat_type}",
        season,
        lambda: nfl.load_pfr_advstats(
            [season], stat_type=stat_type, summary_level="season"
        ).to_pandas(),
        verbose=verbose,
    )

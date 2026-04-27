"""
Remote-first parquet loader.

Locally, every parquet sits in `data/games/`. On Streamlit Cloud, those
files don't exist (they're gitignored — too big and reproducible). On
production we download them from a public Supabase Storage bucket once
per cold start, cache to a writable temp dir, and serve from there.

Usage:
    from lib_data_remote import get_parquet_path
    p = get_parquet_path("nfl_weekly_adjusted.parquet")
    if p is not None:
        df = pd.read_parquet(p)

Returns None on failure (network error, missing file). All callers
already handle None gracefully — they silently skip rendering.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

# Local repo root data dir for the dev / make-run path
_LOCAL_DIR = Path(__file__).resolve().parent / "data" / "games"
# Writable temp dir for production downloads (Streamlit Cloud allows
# /tmp writes; the FS is ephemeral but @st.cache_data keeps the bytes
# in memory anyway).
_REMOTE_CACHE = Path("/tmp") / "lions_rater_games"
_BUCKET = "lions-rater-data"


# Module-level diagnostic state — populated by failed downloads so a
# debug indicator can surface what's actually going wrong on
# Streamlit Cloud (where logs aren't visible to end users).
_LAST_FAILURE: str = ""


def get_last_failure() -> str:
    """Return the most recent failure reason from get_parquet_path
    (empty string if no failures recorded). Used by the debug
    indicator on player pages."""
    return _LAST_FAILURE


def get_parquet_path(filename: str) -> str | None:
    """Return a local filesystem path to the parquet, downloading from
    Supabase Storage if the file isn't on disk yet.

    Caching strategy: SUCCESSFUL paths are cached on disk via the
    `_REMOTE_CACHE` directory — that's the natural session cache.
    FAILURES are deliberately NOT cached so a transient miss can
    self-heal on the next call.

    Returns None if the file can't be obtained — callers handle that
    gracefully. Records the failure reason in `_LAST_FAILURE` so a
    debug indicator can surface what's wrong without log access.
    """
    global _LAST_FAILURE

    # Path 1: local development — file is checked out alongside the repo
    local = _LOCAL_DIR / filename
    if local.exists():
        return str(local)

    # Path 2: production cache hit
    cached = _REMOTE_CACHE / filename
    if cached.exists() and cached.stat().st_size > 0:
        return str(cached)

    # Path 3: download from Supabase public bucket
    try:
        base = st.secrets.get("SUPABASE_URL", "")
    except Exception as e:
        _LAST_FAILURE = f"st.secrets read failed: {e}"
        return None
    if not base:
        _LAST_FAILURE = "SUPABASE_URL not in Streamlit Cloud secrets"
        return None

    url = f"{base}/storage/v1/object/public/{_BUCKET}/{filename}"
    try:
        import requests
    except ImportError as e:
        _LAST_FAILURE = f"requests not installed: {e}"
        return None
    try:
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            _LAST_FAILURE = (f"download {filename}: HTTP {r.status_code} "
                             f"— {r.text[:150]}")
            return None
        if not r.content:
            _LAST_FAILURE = f"download {filename}: empty response"
            return None
        _REMOTE_CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(r.content)
        return str(cached)
    except Exception as e:
        _LAST_FAILURE = f"download {filename}: {type(e).__name__}: {e}"
        return None

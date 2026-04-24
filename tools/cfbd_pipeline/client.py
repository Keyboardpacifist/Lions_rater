"""
Cached CFBD HTTP client.

Reads the API key from .streamlit/secrets.toml or the CFBD_API_KEY env var.
Each (endpoint, params) request is cached as a JSON file under .cache/ so
re-runs don't burn the daily request budget. Cache TTL is generous (30
days) — CFBD season aggregates don't change after the season ends.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days

BASE = "https://api.collegefootballdata.com"


def _read_api_key() -> str:
    """Try secrets.toml first, then env var. Raise if neither has it."""
    env = os.environ.get("CFBD_API_KEY")
    if env:
        return env
    repo_root = Path(__file__).resolve().parent.parent.parent
    secrets_path = repo_root / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            if line.strip().startswith("CFBD_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(
        "CFBD_API_KEY not found in env or .streamlit/secrets.toml"
    )


def _cache_path(endpoint: str, params: dict) -> Path:
    key_blob = json.dumps({"e": endpoint, "p": params}, sort_keys=True)
    h = hashlib.sha256(key_blob.encode()).hexdigest()[:16]
    name = endpoint.strip("/").replace("/", "_")
    return CACHE_DIR / f"{name}__{h}.json"


def _is_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_TTL_SECONDS


def get(endpoint: str, params: dict | None = None, verbose: bool = False) -> list:
    """GET an endpoint with caching. Returns parsed JSON list."""
    params = params or {}
    cache_file = _cache_path(endpoint, params)
    if _is_fresh(cache_file):
        if verbose:
            print(f"  CFBD {endpoint} {params}: cached ({cache_file.name})")
        return json.loads(cache_file.read_text())

    if verbose:
        print(f"  CFBD {endpoint} {params}: fetching...")
    key = _read_api_key()
    r = requests.get(
        f"{BASE}{endpoint}",
        headers={"Authorization": f"Bearer {key}"},
        params=params,
        timeout=60,
    )
    if r.status_code != 200:
        if verbose:
            print(f"  CFBD {endpoint} FAILED ({r.status_code}): {r.text[:200]}")
        return []
    data = r.json()
    cache_file.write_text(json.dumps(data))
    if verbose:
        print(f"  CFBD {endpoint}: {len(data)} rows cached")
    return data


def stats_player_season(year: int, verbose: bool = False) -> list:
    """Per-player counting stats for one season (long format)."""
    return get("/stats/player/season", {"year": year}, verbose=verbose)


def ppa_players_season(year: int, verbose: bool = False) -> list:
    """Per-player PPA (=EPA) for one season."""
    return get("/ppa/players/season", {"year": year}, verbose=verbose)


def player_usage(year: int, verbose: bool = False) -> list:
    """Per-player usage rates (target share, etc.) for one season."""
    return get("/player/usage", {"year": year}, verbose=verbose)


def plays(year: int, week: int, verbose: bool = False) -> list:
    """Play-by-play for a single (year, week). All FBS games."""
    return get("/plays", {"year": year, "week": week}, verbose=verbose)


def roster(year: int, verbose: bool = False) -> list:
    """All-FBS rosters for one season. Used to disambiguate F.LastName
    targets against full names."""
    return get("/roster", {"year": year}, verbose=verbose)

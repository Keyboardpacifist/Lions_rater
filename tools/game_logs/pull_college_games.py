#!/usr/bin/env python3
"""
College game-log pull — per-game player box scores from CollegeFootballData.

Hits CFBD's /games/players endpoint (one HTTP call per (season, season_type))
and flattens the deeply-nested category→type→athletes structure into one
row per player-game. CFBD returns stats as strings (e.g. "21/33") since
they preserve the original box score format; we keep them as strings here
and let downstream callers parse what they need.

Authenticates using CFBD_API_KEY from .streamlit/secrets.toml.

Files written:
  data/games/college_games_players.parquet  — one row per (game, player)

Usage:
    python tools/game_logs/pull_college_games.py
    python tools/game_logs/pull_college_games.py --seasons 2024
    python tools/game_logs/pull_college_games.py --seasons 2016-2025
"""
from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SECRETS_PATH = REPO_ROOT / ".streamlit" / "secrets.toml"
OUT_DIR = REPO_ROOT / "data" / "games"
OUT_FILE = OUT_DIR / "college_games_players.parquet"
DEFAULT_SEASONS = list(range(2016, 2026))
# CFBD requires week/team/conference per call. Iterate by week — wide
# enough to cover any season's full schedule (regular peaks at 17, post
# usually 1-3 chunks for bowls + championship).
SEASON_TYPE_WEEKS = {"regular": range(1, 18), "postseason": range(1, 4)}
CFBD_BASE = "https://api.collegefootballdata.com"
# Be polite — small inter-call delay so we don't trigger CFBD's
# burst limits during long pulls.
SLEEP_BETWEEN_CALLS = 0.15


def _load_api_key() -> str:
    if not SECRETS_PATH.exists():
        raise SystemExit(
            f"Missing secrets file at {SECRETS_PATH}. "
            "Add CFBD_API_KEY to .streamlit/secrets.toml."
        )
    with open(SECRETS_PATH, "rb") as f:
        secrets = tomllib.load(f)
    key = secrets.get("CFBD_API_KEY")
    if not key:
        raise SystemExit(
            "CFBD_API_KEY not found in .streamlit/secrets.toml. "
            "Sign up free at collegefootballdata.com and add the key there."
        )
    return key


def parse_seasons(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x.strip()) for x in s.split(",")]
    return [int(s)]


def _normalize_col(category: str, type_name: str) -> str:
    """CFBD type names are display-y ('C/ATT', 'YDS', 'TD'). Build a
    deterministic snake_case column name like 'passing_c_att'."""
    t = (type_name or "").lower()
    t = (t.replace("/", "_").replace(" ", "_")
           .replace(".", "").replace("%", "pct"))
    return f"{category}_{t}"


def fetch_games_players(api_key: str, year: int,
                        season_type: str, week: int) -> list[dict]:
    """One HTTP call returning every game's player stats for the
    (year, season_type, week). CFBD requires week/team/conference, so
    we paginate by week."""
    url = f"{CFBD_BASE}/games/players"
    params = {"year": year, "seasonType": season_type, "week": week}
    headers = {"Authorization": f"Bearer {api_key}",
               "Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=120)
    if r.status_code == 401:
        raise SystemExit("CFBD auth failed (401). Check CFBD_API_KEY.")
    if r.status_code == 429:
        raise SystemExit("CFBD rate-limited (429). Retry later.")
    r.raise_for_status()
    return r.json()


def flatten_games(games: Iterable[dict]) -> list[dict]:
    """One row per (game, school, player). Stats stored as strings to
    preserve CFBD's original formatting (e.g. '21/33' for completions)."""
    rows: list[dict] = []
    for game in games:
        gid = game.get("id")
        season = game.get("season")
        week = game.get("week")
        season_type = game.get("seasonType")

        teams = game.get("teams", []) or []
        # Pre-compute opponent for each team — only valid when len==2.
        opp = {}
        if len(teams) == 2:
            a, b = teams[0].get("school"), teams[1].get("school")
            opp = {a: b, b: a}

        for team in teams:
            school = team.get("school")
            conf = team.get("conference")
            home_away = team.get("homeAway")
            points = team.get("points")

            # Bucket per-player columns first, then emit one row per athlete
            # with all their categories merged.
            by_athlete: dict[tuple, dict] = {}
            for cat in (team.get("categories") or []):
                cat_name = (cat.get("name") or "").lower()
                for stype in (cat.get("types") or []):
                    type_name = stype.get("name") or ""
                    col = _normalize_col(cat_name, type_name)
                    for ath in (stype.get("athletes") or []):
                        aid = ath.get("id")
                        aname = ath.get("name")
                        stat = ath.get("stat")
                        key = (aid, aname)
                        bucket = by_athlete.setdefault(key, {
                            "player_id": aid,
                            "player_name": aname,
                        })
                        bucket[col] = stat

            for stats in by_athlete.values():
                rows.append({
                    "game_id": gid,
                    "season": season,
                    "week": week,
                    "season_type": season_type,
                    "school": school,
                    "opponent": opp.get(school),
                    "home_away": home_away,
                    "conference": conf,
                    "team_points": points,
                    **stats,
                })
    return rows


def main(seasons: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = _load_api_key()

    print(f"Pulling College game logs for seasons: "
          f"{seasons[0]}–{seasons[-1]}")
    print(f"Output: {OUT_FILE}\n")

    t0 = time.time()
    all_rows: list[dict] = []

    for year in seasons:
        for st, weeks in SEASON_TYPE_WEEKS.items():
            year_games = 0
            year_rows = 0
            for wk in weeks:
                try:
                    games = fetch_games_players(api_key, year, st, wk)
                except requests.HTTPError as e:
                    print(f"  {year} {st} wk{wk:<2} ⚠️  HTTP {e}")
                    continue
                if not games:
                    # Week beyond the schedule's end — silently skip.
                    continue
                rows = flatten_games(games)
                all_rows.extend(rows)
                year_games += len(games)
                year_rows += len(rows)
                time.sleep(SLEEP_BETWEEN_CALLS)
            print(f"  {year} {st:<11} {year_games:>4} games · "
                  f"{year_rows:>6,} player-game rows")

    if not all_rows:
        print("\n⚠️  no rows collected — aborting before writing.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    # Stable column order: identity first, then sorted stat columns.
    id_cols = ["game_id", "season", "week", "season_type", "school",
               "opponent", "home_away", "conference", "team_points",
               "player_id", "player_name"]
    other = sorted(c for c in df.columns if c not in id_cols)
    df = df[[c for c in id_cols if c in df.columns] + other]

    df.to_parquet(OUT_FILE, index=False)
    print(f"\n✅ wrote {len(df):,} rows × {len(df.columns)} cols "
          f"in {time.time() - t0:.1f}s")


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

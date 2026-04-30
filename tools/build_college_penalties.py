#!/usr/bin/env python3
"""Build per-player college penalty counts from CFBD play-by-play.

Output: data/draft_college_penalties.parquet
  Columns: player, school, season, penalties, types_json, pen_yards

Strategy:
  1. Read the consensus board → unique (school, season) pairs we care
     about (the schools our 400 prospects played for).
  2. For each pair, hit CFBD's /plays endpoint, filter penalty plays.
  3. Regex-extract the penalized player from play_text. CFBD describes
     plays like "Penalty on JOHN DOE, Defensive Holding, 5 yards" or
     "Holding (Defense) on Notre Dame, John Doe (Notre Dame), 10 yards".
  4. Aggregate per (player, school, season) → count + breakdown.
  5. Save parquet for the page to z-score and rank.

Setup:
  CFBD_API_KEY — free at https://collegefootballdata.com/key

Usage:
  python tools/build_college_penalties.py             # all years
  python tools/build_college_penalties.py --years 2024 2025
  python tools/build_college_penalties.py --resume    # skip already
                                                       # cached
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
CONSENSUS = REPO / "data" / "draft_2027_consensus.parquet"
OUT = REPO / "data" / "draft_college_penalties.parquet"
CACHE_DIR = REPO / ".data_cache" / "cfbd_pbp"
SECRETS = REPO / ".streamlit" / "secrets.toml"

CFBD_PLAYS_URL = "https://api.collegefootballdata.com/plays"
DEFAULT_YEARS = (2023, 2024, 2025)


def _load_api_key() -> str | None:
    key = os.environ.get("CFBD_API_KEY")
    if key:
        return key
    if SECRETS.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(SECRETS, "rb") as f:
            cfg = tomllib.load(f)
        return cfg.get("CFBD_API_KEY")
    return None


# ── Penalty extraction ──────────────────────────────────────────
# Real CFBD format observed:
#   "... Alabama Penalty, Offensive Holding (Gage Larvadain) to SC 9"
#   "... Penalty, Personal Foul (Jalen Milroe) to the ALA 28"
#
# Pattern: <school> Penalty, <PENALTY TYPE> (<PLAYER NAME>) to ...
_PENALTY_RE = re.compile(
    r"([A-Z][A-Za-z .&'\-]+?)\s+"            # team
    r"Penalty,\s*"
    r"([A-Za-z .'\-/]+?)\s*"                  # penalty type
    r"\(([^()]+?)\)",                         # player or note
    re.IGNORECASE,
)

# Reject parenthesized values that aren't names (yardage notes, etc.)
_NOT_NAME = re.compile(r"\d|yards|declined|offset|no play",
                         re.IGNORECASE)


def parse_penalty(play_text: str, school: str
                   ) -> tuple[str, str] | None:
    """Returns (player_name, penalty_type) or None if the penalty
    can't be cleanly attributed to a player on `school`."""
    if not play_text:
        return None
    school_l = school.lower()
    for m in _PENALTY_RE.finditer(play_text):
        team, ptype, who = (m.group(1).strip(),
                              m.group(2).strip().lower(),
                              m.group(3).strip())
        # Only count penalties on this team's player
        if school_l not in team.lower() and team.lower() not in school_l:
            continue
        if _NOT_NAME.search(who):
            continue
        # Player name should be 2+ tokens of letters
        toks = who.split()
        if len(toks) < 2 or any(ch.isdigit() for ch in who):
            continue
        return who, ptype
    return None


def is_penalty_play(play: dict) -> bool:
    # CFBD /plays returns camelCase keys.
    pt = (play.get("playType") or play.get("play_type") or "").lower()
    if "penalty" in pt:
        return True
    text = (play.get("playText") or play.get("play_text") or "").lower()
    return ("penalty" in text or "(defense) on" in text
            or "(offense) on" in text)


def fetch_plays(year: int, school: str, api_key: str) -> list[dict]:
    """CFBD /plays for one (year, team). Cached on disk to make
    re-runs fast and free."""
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", school)
    cache_path = CACHE_DIR / f"{year}_{safe}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    headers = {"Authorization": f"Bearer {api_key}"}
    # CFBD's /plays requires a week — loop regular season weeks 1-15
    # plus a couple postseason weeks. Empty weeks return [] cheaply.
    plays = []
    for season_type, weeks in (("regular", range(1, 16)),
                                 ("postseason", range(1, 3))):
        for week in weeks:
            try:
                resp = requests.get(
                    CFBD_PLAYS_URL, headers=headers,
                    params={"year": year, "team": school, "week": week,
                             "seasonType": season_type,
                             "classification": "fbs"},
                    timeout=30,
                )
            except Exception as e:
                print(f"  [{year} {school} w{week}] HTTP error: {e}")
                continue
            if resp.status_code != 200:
                if resp.status_code in (400, 404):
                    continue
                print(f"  [{year} {school} w{week}] status "
                      f"{resp.status_code}: {resp.text[:160]}")
                continue
            plays.extend(resp.json() or [])
            time.sleep(0.05)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(plays, f)
    return plays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int,
                          default=list(DEFAULT_YEARS),
                          help="Seasons to pull (default 2023 2024 2025)")
    parser.add_argument("--sleep", type=float, default=0.4,
                          help="Seconds between CFBD calls")
    args = parser.parse_args()

    api_key = _load_api_key()
    if not api_key:
        print("ERROR: CFBD_API_KEY not set (env var or "
              ".streamlit/secrets.toml).")
        print("Get a free key at https://collegefootballdata.com/key")
        sys.exit(1)

    if not CONSENSUS.exists():
        print(f"ERROR: missing {CONSENSUS}")
        sys.exit(1)

    consensus = pd.read_parquet(CONSENSUS)
    schools = sorted(set(consensus["school"].dropna().astype(str)))
    print(f"Pulling plays for {len(schools)} schools × "
          f"{len(args.years)} seasons "
          f"({len(schools)*len(args.years)} CFBD calls)")

    # Aggregator: (player_normalized, school, season) → counts
    counts: dict = defaultdict(lambda: {"penalties": 0,
                                          "types": Counter(),
                                          "pen_yards": 0})

    for i, school in enumerate(schools, start=1):
        for year in args.years:
            print(f"[{i}/{len(schools)}] {school} {year} ...",
                  end=" ", flush=True)
            plays = fetch_plays(year, school, api_key)
            n_pen = 0
            for play in plays:
                if not is_penalty_play(play):
                    continue
                text = (play.get("playText")
                        or play.get("play_text") or "")
                parsed = parse_penalty(text, school)
                if not parsed:
                    continue
                player, ptype = parsed
                yards = abs(int(play.get("yardsGained")
                                  or play.get("yards_gained") or 0))
                key = (player, school, year)
                counts[key]["penalties"] += 1
                counts[key]["types"][ptype] += 1
                counts[key]["pen_yards"] += yards
                n_pen += 1
            print(f"{len(plays)} plays · {n_pen} penalty events")
            time.sleep(args.sleep)

    rows = []
    for (player, school, year), agg in counts.items():
        rows.append({
            "player": player,
            "school": school,
            "season": year,
            "penalties": agg["penalties"],
            "pen_yards": agg["pen_yards"],
            "types_json": json.dumps(dict(agg["types"])),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print("No penalty events extracted — check parsing patterns.")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n✓ wrote {OUT.relative_to(REPO)}")
    print(f"  {len(df)} player-season rows · "
          f"{df['penalties'].sum()} total penalty events")
    print(f"  Cache dir: {CACHE_DIR.relative_to(REPO)} "
          f"(re-runs are instant)")


if __name__ == "__main__":
    main()

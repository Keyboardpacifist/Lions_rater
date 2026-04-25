"""
Pull college OL rosters from CFBD's /roster endpoint and write a flat
parquet that the app can use as the OL identity source.

CFBD doesn't publish per-player offensive-line stats (no PFF-style grades
in the free API), so this file is identity-only — name, team, season,
position, height, weight, class year. Combined with the recruiting
parquet + combine parquet that already exist, the app can show a useful
OL profile (recruiting stars, measurables, draft info) even without
play-by-play stats.

Output: data/college/college_ol_roster.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import client

OUTPUT_PATH = (Path(__file__).resolve().parent.parent.parent
               / "data" / "college" / "college_ol_roster.parquet")

OL_POSITIONS = {"OT", "G", "C", "OL", "OG", "IOL", "T", "OC"}


def run(seasons: list[int] | None = None, verbose: bool = True) -> None:
    seasons = seasons or list(range(2014, 2026))
    rows = []
    for year in seasons:
        if verbose:
            print(f"[{year}] fetching FBS roster...")
        roster = client.roster(year, verbose=verbose)
        if not roster:
            if verbose:
                print(f"  no roster data for {year}")
            continue
        for r in roster:
            pos = (r.get("position") or "").strip().upper()
            if pos not in OL_POSITIONS:
                continue
            # CFBD uses camelCase (firstName, lastName, homeCity, ...).
            first = (r.get("firstName") or r.get("first_name") or "").strip()
            last = (r.get("lastName") or r.get("last_name") or "").strip()
            name = f"{first} {last}".strip()
            if not name:
                continue
            rows.append({
                "player": name,
                "team": (r.get("team") or "").strip(),
                "season": int(year),
                "position": pos,
                "height": r.get("height"),  # inches
                "weight": r.get("weight"),  # lbs
                "jersey": r.get("jersey"),
                "class_year": r.get("year"),  # roster's "year" = season, NOT class
                "home_city": (r.get("homeCity") or r.get("home_city") or "").strip(),
                "home_state": (r.get("homeState") or r.get("home_state") or "").strip(),
                "player_id": r.get("id"),
            })
        if verbose:
            print(f"  {year}: {sum(1 for x in rows if x['season'] == year)} OL kept")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No OL rows pulled — aborting (CFBD likely returned empty).")
        return
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    if verbose:
        print(
            f"Wrote {OUTPUT_PATH.name}: {len(df)} rows, "
            f"{df['player'].nunique()} unique OL across {df['season'].nunique()} seasons"
        )
        print("Position breakdown:", df["position"].value_counts().to_dict())


if __name__ == "__main__":
    run(verbose=True)

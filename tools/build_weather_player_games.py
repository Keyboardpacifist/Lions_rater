"""Join weekly player stats to game weather → per-player-game weather table.

Output: data/player_games_weather.parquet

For each player-game, attaches temp / wind / roof / surface so we can
do cohort matching: "find this player's historical games at temp ≤ 40°F
and wind ≥ 15 mph" → empirical distribution of stat outcomes.

The weather slider in Feature 4.5 queries this table.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
SCHEDULES = REPO / "data" / "nfl_schedules.parquet"
OUT = REPO / "data" / "player_games_weather.parquet"


def main() -> None:
    print("→ loading player stats + schedules...")
    ps = pd.read_parquet(PLAYER_STATS)
    sch = pd.read_parquet(SCHEDULES)
    print(f"  player stats: {len(ps):,}  schedules: {len(sch):,}")

    # Build (season, week, team) → game weather lookup. Each game has
    # two team rows (one for home, one for away) so the player stat
    # row's `team` joins to whichever side matches.
    home = sch[["season", "week", "home_team", "temp", "wind", "roof",
                "surface", "stadium", "stadium_id", "spread_line",
                "total_line", "home_coach", "away_coach"]].copy()
    home = home.rename(columns={
        "home_team": "team",
        "home_coach": "team_coach",
        "away_coach": "opp_coach",
    })
    home["is_home"] = True

    away = sch[["season", "week", "away_team", "temp", "wind", "roof",
                "surface", "stadium", "stadium_id", "spread_line",
                "total_line", "home_coach", "away_coach"]].copy()
    away = away.rename(columns={
        "away_team": "team",
        "away_coach": "team_coach",
        "home_coach": "opp_coach",
    })
    away["is_home"] = False
    # Spread is from home's perspective; flip for away
    away["spread_line"] = -away["spread_line"]

    weather = pd.concat([home, away], ignore_index=True)

    merged = ps.merge(weather, on=["season", "week", "team"], how="left")
    print(f"  joined: {len(merged):,}")
    print(f"  weather coverage: temp {merged['temp'].notna().mean():.0%}, "
          f"wind {merged['wind'].notna().mean():.0%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

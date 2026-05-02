"""Pull historical NFL schedules with weather + closing odds + coaches.

Output: data/nfl_schedules.parquet

This single table is the foundation for FOUR gambling features:
  • 4.2 game-script simulator (coach + game-script context)
  • 4.3 books-vs-model (closing spread/total/moneyline → outcome)
  • 4.4 smart alerts (matchup + bet-context fusion)
  • 4.5 weather production window (temp/wind/roof/surface)

Per-game columns:
  weather:  temp, wind, roof, surface, stadium, stadium_id
  odds:     spread_line, total_line, away/home_moneyline,
            away/home_spread_odds, over_odds, under_odds
  context:  home_coach, away_coach, referee, home_qb, away_qb,
            home_rest, away_rest, div_game
  result:   home_score, away_score, result, total, overtime
"""
from __future__ import annotations

from pathlib import Path

import nflreadpy

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "nfl_schedules.parquet"


def main() -> None:
    print("→ pulling NFL schedules...")
    df = nflreadpy.load_schedules(seasons=True)
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    print(f"  rows: {len(df):,}")
    if "season" in df.columns:
        seasons = sorted(df["season"].dropna().unique().tolist())
        print(f"  seasons: {int(seasons[0])}–{int(seasons[-1])}")
        # Coverage check on the new fields
        for col in ("temp", "wind", "spread_line", "total_line",
                    "home_coach", "referee"):
            if col in df.columns:
                cov = df[col].notna().mean()
                print(f"  {col}: {cov:.0%} coverage")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

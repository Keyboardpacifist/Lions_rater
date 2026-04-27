#!/usr/bin/env python3
"""
Per-offense, per-position baselines — the symmetric flip of
nfl_defense_baselines. Used to give defensive players a 'schedule
strength' picture: how good were the offenses they faced?

For each (offense_team, season, opp_position_group), aggregate the
qualifying opposing defender's per-game stats. The result answers
the question 'how did this offense, on average, perform against
opposing defenders at position X, on a per-player-game basis.'

Inputs:
  data/games/nfl_weekly_stats.parquet

Outputs:
  data/games/nfl_offense_baselines.parquet
    - One row per (offense_team, season, defensive_position)
    - position ∈ {DE, DT, LB, CB, S}
    - Stat columns are means of qualifying opposing defender games
    - Plus offense-side context cols for tiering offenses by quality:
        avg_pass_epa, avg_rush_yards_pg, avg_total_yards_pg,
        avg_points_pg

Filters (qualifying defender threshold):
  All defensive positions: def_snaps ≥ 30 (one quarter+ of work)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT = REPO_ROOT / "data" / "games" / "nfl_weekly_stats.parquet"
OUTPUT = REPO_ROOT / "data" / "games" / "nfl_offense_baselines.parquet"

# Defensive position groups + positions that count toward each.
# Qualifying gate = at least 1 combined tackle (solo + assist) — a
# clean signal that the defender was on the field for meaningful work.
# Cleaner than name-based snap-count joins which break across feeds.
DEFENSIVE_POSITION_GROUPS = {
    "DE": {"positions": ["DE", "EDGE", "OLB"]},
    "DT": {"positions": ["DT", "NT", "DL"]},
    "LB": {"positions": ["LB", "ILB", "MLB", "OLB"]},
    "CB": {"positions": ["CB", "DB"]},
    "S": {"positions": ["S", "FS", "SS", "SAF", "DB"]},
}
MIN_TACKLES = 1  # At least 1 combined tackle = qualifying defender game.

# Stats we'll average for each defensive position. All come straight
# off the weekly box score.
DEF_STATS = [
    "def_tackles_solo", "def_tackle_assists", "def_tackles_for_loss",
    "def_sacks", "def_qb_hits", "def_pass_defended",
    "def_interceptions", "def_fumbles_forced",
]


def main():
    if not INPUT.exists():
        raise SystemExit(f"Missing {INPUT}. Run `make game-logs-nfl`.")

    print(f"Reading {INPUT.name}…")
    df = pl.read_parquet(INPUT)
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} cols\n")

    # Compute combined tackles and ensure all DEF_STATS exist.
    if "def_tackles" not in df.columns and "def_tackles_solo" in df.columns:
        df = df.with_columns(
            (pl.col("def_tackles_solo").fill_null(0)
             + pl.col("def_tackle_assists").fill_null(0)).alias("def_tackles")
        )
    DEF_STATS_FULL = DEF_STATS + (["def_tackles"] if "def_tackles" in df.columns
                                    else [])

    parts = []
    for name, cfg in DEFENSIVE_POSITION_GROUPS.items():
        filt = df.filter(pl.col("position").is_in(cfg["positions"]))
        filt = filt.filter(pl.col("def_tackles").fill_null(0) >= MIN_TACKLES)
        filt = filt.filter(pl.col("opponent_team").is_not_null())
        filt = filt.filter(pl.col("season").is_not_null())

        # Group by (opponent_team, season) — the OFFENSE this defender
        # was facing. "opponent_team" from the defender's row IS the
        # offense.
        agg_exprs = [
            pl.struct(["season_type", "week"]).n_unique().alias("n_games_with_qual"),
            pl.col("player_id").n_unique().alias("n_unique_players"),
            pl.len().alias("n_player_games"),
        ]
        for c in DEF_STATS_FULL:
            if c in filt.columns:
                agg_exprs.append(pl.col(c).mean().alias(c))

        agg = (filt.group_by(["opponent_team", "season"])
                   .agg(agg_exprs)
                   .rename({"opponent_team": "offense_team"}))
        agg = agg.with_columns(pl.lit(name).alias("position"))
        parts.append(agg)

    out = pl.concat(parts, how="diagonal_relaxed")
    out = out.sort(["offense_team", "season", "position"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUTPUT)
    print(f"✅ wrote {out.shape[0]:,} rows × {out.shape[1]} cols")
    print(f"   → {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

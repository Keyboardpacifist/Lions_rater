#!/usr/bin/env python3
"""
Build per-defense, per-position baselines from existing game logs.

For each (defense_team, season, opponent_position), aggregate the
qualifying opponent player-game stats. The result answers the question
"how did this defense, on average, perform against opposing position X
on a per-player-game basis."

These baselines are the foundation for player opponent-adjustment:
a player's individual game can be compared to the average qualifying
player-game his opponent's defense allowed.

Inputs (must already exist — run `make game-logs` first):
  data/games/nfl_weekly_stats.parquet

Outputs:
  data/games/nfl_defense_baselines.parquet
    - One row per (defense_team, season, position_group)
    - position_group ∈ {QB, RB, WR, TE}
    - Stat columns are means of qualifying opp player-games

Filters applied per position group (drops noise from 1-snap cameos):
  QB:  attempts ≥ 10
  RB:  carries  ≥ 5     (RB + FB collapsed into "RB")
  WR:  targets  ≥ 2
  TE:  targets  ≥ 2

Usage:
    python tools/game_logs/build_defense_baselines.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT = REPO_ROOT / "data" / "games" / "nfl_weekly_stats.parquet"
OUTPUT = REPO_ROOT / "data" / "games" / "nfl_defense_baselines.parquet"

# Per-position-group config: which `position` values count, which volume
# column gates inclusion, and the volume threshold. Thresholds are
# deliberately permissive — they cut out 1-snap cameos but keep RB2s,
# slot WRs, and TE2s in the baseline.
POSITION_GROUPS = {
    "QB": {"positions": ["QB"], "gate": "attempts", "min": 10},
    "RB": {"positions": ["RB", "FB"], "gate": "carries", "min": 5},
    "WR": {"positions": ["WR"], "gate": "targets", "min": 2},
    "TE": {"positions": ["TE"], "gate": "targets", "min": 2},
}

# Stats we want averaged for each group. Per-attempt rates are computed
# from totals (ypa = sum_yds / sum_att), not as a mean of per-game rates,
# to weight by volume correctly.
MEAN_COLS_BY_GROUP = {
    "QB": [
        "attempts", "completions", "passing_yards", "passing_tds",
        "passing_interceptions", "sacks_suffered",
        "passing_epa", "passing_cpoe",
        "carries", "rushing_yards",
    ],
    "RB": [
        "carries", "rushing_yards", "rushing_tds", "rushing_epa",
        "targets", "receptions", "receiving_yards", "receiving_epa",
    ],
    "WR": [
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "receiving_epa", "target_share",
    ],
    "TE": [
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "receiving_epa", "target_share",
    ],
}

# Rate stats we compute from totals after the group-by.
RATE_DEFS_BY_GROUP = {
    "QB": [
        ("yards_per_attempt", "passing_yards", "attempts"),
        ("completion_pct", "completions", "attempts"),
    ],
    "RB": [
        ("yards_per_carry", "rushing_yards", "carries"),
        ("yards_per_target", "receiving_yards", "targets"),
    ],
    "WR": [
        ("yards_per_target", "receiving_yards", "targets"),
        ("catch_rate", "receptions", "targets"),
    ],
    "TE": [
        ("yards_per_target", "receiving_yards", "targets"),
        ("catch_rate", "receptions", "targets"),
    ],
}


def build_for_group(df: pl.DataFrame, name: str, cfg: dict) -> pl.DataFrame:
    mean_cols = MEAN_COLS_BY_GROUP[name]
    rate_defs = RATE_DEFS_BY_GROUP[name]

    # Filter to qualifying opp player-games for this position group.
    filtered = (
        df.filter(pl.col("position").is_in(cfg["positions"]))
          .filter(pl.col(cfg["gate"]).fill_null(0) >= cfg["min"])
          .filter(pl.col("opponent_team").is_not_null())
          .filter(pl.col("season").is_not_null())
    )

    # Build per-(defense, season) aggregates. game_id is NULL in this
    # source, so use (season_type, week) as the per-game key — defense
    # plays at most one game per (season_type, week).
    agg_exprs = [
        pl.struct(["season_type", "week"]).n_unique().alias("n_games_with_qual"),
        pl.col("player_id").n_unique().alias("n_unique_players"),
        pl.len().alias("n_player_games"),
    ]
    for c in mean_cols:
        agg_exprs.append(pl.col(c).mean().alias(c))
    # Also keep volume sums so we can derive rate stats afterwards.
    sum_cols_needed = {col for _, num, den in rate_defs for col in (num, den)}
    for c in sum_cols_needed:
        agg_exprs.append(pl.col(c).sum().alias(f"_sum_{c}"))

    agg = (
        filtered
        .group_by(["opponent_team", "season"])
        .agg(agg_exprs)
        .rename({"opponent_team": "defense_team"})
    )

    # Derive rate stats from the volume sums, then drop the helpers.
    for new_col, num, den in rate_defs:
        agg = agg.with_columns(
            pl.when(pl.col(f"_sum_{den}") > 0)
              .then(pl.col(f"_sum_{num}") / pl.col(f"_sum_{den}"))
              .otherwise(None)
              .alias(new_col)
        )
    drop_cols = [f"_sum_{c}" for c in sum_cols_needed]
    agg = agg.drop(drop_cols)

    # Tag with the position group label.
    agg = agg.with_columns(pl.lit(name).alias("position"))

    # Reorder: identity columns first, then stats.
    id_cols = ["defense_team", "season", "position",
               "n_games_with_qual", "n_unique_players", "n_player_games"]
    other = [c for c in agg.columns if c not in id_cols]
    return agg.select(id_cols + other)


def main():
    if not INPUT.exists():
        raise SystemExit(
            f"Missing {INPUT}. Run `make game-logs-nfl` first."
        )

    print(f"Reading {INPUT.name}…")
    df = pl.read_parquet(INPUT)
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} cols\n")

    parts = []
    for name, cfg in POSITION_GROUPS.items():
        t0 = time.time()
        out = build_for_group(df, name, cfg)
        print(f"  [{name}] {out.shape[0]:>4} (defense, season) rows · "
              f"{(time.time()-t0):.2f}s")
        parts.append(out)

    # Concat — schemas don't perfectly align across position groups
    # because rate stats and mean stats differ. Use diagonal concat so
    # missing columns become NULL automatically.
    combined = pl.concat(parts, how="diagonal_relaxed")
    combined = combined.sort(["defense_team", "season", "position"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUTPUT)
    print(f"\n✅ wrote {combined.shape[0]:,} rows × {combined.shape[1]} cols")
    print(f"   → {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

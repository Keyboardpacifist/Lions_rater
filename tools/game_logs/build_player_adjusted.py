#!/usr/bin/env python3
"""
Per player-game opponent-adjusted stats — NFL.

For every NFL skill-position player-game, looks up the opposing defense's
season baseline (from nfl_defense_baselines.parquet) and computes:
  - <stat>_expected   the defense's per-qualifying-player-game average
  - <stat>_delta      actual − expected (positive = beat the schedule)

Position scope: QB / RB / FB / WR / TE. Other positions are dropped from
this output — we don't have offensive-style baselines for them.

Inputs (must already exist):
  data/games/nfl_weekly_stats.parquet
  data/games/nfl_defense_baselines.parquet  (run build_defense_baselines.py first)

Output:
  data/games/nfl_weekly_adjusted.parquet
    - One row per qualifying player-game
    - Identity cols + actual stat cols + _expected + _delta columns

Usage:
    python tools/game_logs/build_player_adjusted.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_PATH = REPO_ROOT / "data" / "games" / "nfl_weekly_stats.parquet"
BASELINES_PATH = REPO_ROOT / "data" / "games" / "nfl_defense_baselines.parquet"
OUTPUT = REPO_ROOT / "data" / "games" / "nfl_weekly_adjusted.parquet"

# Position-group mapping — same shape as the baseline builder so the
# join key (defense_team, season, position) lines up cleanly.
POSITION_TO_GROUP = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "WR": "WR",
    "TE": "TE",
}

# Stats to compute expected + delta columns for, per position group.
# These mirror the baseline parquet's stat columns.
ADJUSTED_STATS_BY_GROUP = {
    "QB": [
        "attempts", "completions", "passing_yards", "passing_tds",
        "passing_interceptions", "sacks_suffered",
        "passing_epa", "passing_cpoe",
        "yards_per_attempt", "completion_pct",
        "carries", "rushing_yards",
    ],
    "RB": [
        "carries", "rushing_yards", "rushing_tds", "rushing_epa",
        "yards_per_carry",
        "targets", "receptions", "receiving_yards",
    ],
    "WR": [
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "receiving_epa", "target_share",
        "yards_per_target", "catch_rate",
    ],
    "TE": [
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "receiving_epa", "target_share",
        "yards_per_target", "catch_rate",
    ],
}


def main():
    if not STATS_PATH.exists():
        raise SystemExit(f"Missing {STATS_PATH}. Run `make game-logs-nfl`.")
    if not BASELINES_PATH.exists():
        raise SystemExit(
            f"Missing {BASELINES_PATH}. "
            "Run `python tools/game_logs/build_defense_baselines.py` first."
        )

    print("Loading inputs…")
    stats = pl.read_parquet(STATS_PATH)
    bases = pl.read_parquet(BASELINES_PATH)
    print(f"  stats: {stats.shape[0]:,} rows × {stats.shape[1]} cols")
    print(f"  baselines: {bases.shape[0]:,} rows × {bases.shape[1]} cols\n")

    # Filter player-games to skill positions and add a position_group col
    # that matches the baseline's `position` field for the join.
    pos_map_expr = pl.col("position").replace_strict(
        POSITION_TO_GROUP, default=None,
        return_dtype=pl.String,
    )
    skill = (
        stats
        .with_columns(pos_map_expr.alias("position_group"))
        .filter(pl.col("position_group").is_not_null())
        .filter(pl.col("opponent_team").is_not_null())
        .filter(pl.col("season").is_not_null())
    )
    print(f"Skill-position player-games: {skill.shape[0]:,}\n")

    # Compute derived rate stats on the player-game side so they can be
    # delta'd against the baseline rate. Done at row level (per game).
    skill = skill.with_columns([
        pl.when(pl.col("attempts") > 0)
          .then(pl.col("passing_yards") / pl.col("attempts"))
          .otherwise(None).alias("yards_per_attempt"),
        pl.when(pl.col("attempts") > 0)
          .then(pl.col("completions") / pl.col("attempts"))
          .otherwise(None).alias("completion_pct"),
        pl.when(pl.col("carries") > 0)
          .then(pl.col("rushing_yards") / pl.col("carries"))
          .otherwise(None).alias("yards_per_carry"),
        pl.when(pl.col("targets") > 0)
          .then(pl.col("receiving_yards") / pl.col("targets"))
          .otherwise(None).alias("yards_per_target"),
        pl.when(pl.col("targets") > 0)
          .then(pl.col("receptions") / pl.col("targets"))
          .otherwise(None).alias("catch_rate"),
    ])

    # Prepare baselines for the join: rename the stat cols to *_expected
    # so they don't collide with the player-game columns.
    all_stats = sorted({s for stats_list in ADJUSTED_STATS_BY_GROUP.values()
                        for s in stats_list})
    bases_for_join = bases.rename({c: f"{c}_expected"
                                    for c in all_stats if c in bases.columns})
    bases_for_join = bases_for_join.rename({"position": "position_group"})

    # Identity baseline cols we want to keep alongside the expected stats
    # — useful for sample-size warnings on the consumer side.
    base_id_cols = ["defense_team", "season", "position_group",
                    "n_games_with_qual", "n_unique_players",
                    "n_player_games"]
    bases_for_join = bases_for_join.rename({
        "n_games_with_qual": "_def_n_games_with_qual",
        "n_unique_players": "_def_n_unique_players",
        "n_player_games": "_def_n_player_games",
    })
    bases_for_join = bases_for_join.rename({"defense_team": "opponent_team"})

    # Join: player_game.opponent_team == baseline.defense_team,
    # plus season + position_group.
    joined = skill.join(
        bases_for_join,
        on=["opponent_team", "season", "position_group"],
        how="left",
    )

    # Compute delta = actual − expected, one column per unique stat.
    # A given stat (e.g. `carries`) can belong to multiple position
    # groups (QB rushing + RB), so we pre-build {stat: [groups...]}
    # and gate the delta on the player belonging to ANY of those groups.
    stat_groups: dict[str, list[str]] = {}
    for group, stat_list in ADJUSTED_STATS_BY_GROUP.items():
        for stat in stat_list:
            stat_groups.setdefault(stat, []).append(group)

    delta_exprs = []
    for stat, groups in stat_groups.items():
        exp_col = f"{stat}_expected"
        delta_col = f"{stat}_delta"
        if exp_col not in joined.columns or stat not in joined.columns:
            continue
        delta_exprs.append(
            pl.when(
                pl.col("position_group").is_in(groups)
                & pl.col(stat).is_not_null()
                & pl.col(exp_col).is_not_null()
            )
            .then(pl.col(stat) - pl.col(exp_col))
            .otherwise(None)
            .alias(delta_col)
        )
    joined = joined.with_columns(delta_exprs)

    # Slim down: keep identity + actual stat cols + _expected + _delta.
    # (The baseline parquet has all stats, but a row for QB shouldn't
    # carry e.g. WR target_share_expected.)
    keep_id = [
        "season", "season_type", "week", "team", "opponent_team",
        "player_id", "player_name", "player_display_name", "position",
        "position_group",
        "_def_n_games_with_qual", "_def_n_unique_players",
        "_def_n_player_games",
    ]
    actual_cols = sorted({s for stat_list in ADJUSTED_STATS_BY_GROUP.values()
                          for s in stat_list})
    expected_cols = [f"{s}_expected" for s in actual_cols
                     if f"{s}_expected" in joined.columns]
    delta_cols = [f"{s}_delta" for s in actual_cols
                  if f"{s}_delta" in joined.columns]

    final_cols = (
        [c for c in keep_id if c in joined.columns]
        + [c for c in actual_cols if c in joined.columns]
        + expected_cols
        + delta_cols
    )
    out = joined.select(final_cols)
    out = out.sort(["season", "week", "team", "position_group",
                    "player_display_name"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUTPUT)

    print(f"✅ wrote {out.shape[0]:,} rows × {out.shape[1]} cols")
    print(f"   → {OUTPUT}\n")

    # Per-group sanity report
    print("Rows by position group:")
    print(out.group_by("position_group").agg(pl.len().alias("rows"))
             .sort("rows", descending=True).to_pandas().to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

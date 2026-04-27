#!/usr/bin/env python3
"""
Per-defensive-player game adjusted stats — symmetric to
build_player_adjusted but for defenders.

For every NFL defensive player-game (DE/DT/LB/CB/S), looks up the
opposing offense's per-defender baseline and computes:
  - <stat>_expected   the offense's per-qualifying-defender-game average
  - <stat>_delta      actual − expected (positive = beat the schedule)

Position scope: DE / DT / LB / CB / S. Positions on the front line
that bridge groups (e.g. EDGE → DE, OLB → LB) are mapped sensibly.

Inputs (must already exist):
  data/games/nfl_weekly_stats.parquet
  data/games/nfl_offense_baselines.parquet  (run build_offense_baselines first)

Output:
  data/games/nfl_defensive_player_adjusted.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATS_PATH = REPO_ROOT / "data" / "games" / "nfl_weekly_stats.parquet"
BASELINES_PATH = REPO_ROOT / "data" / "games" / "nfl_offense_baselines.parquet"
OUTPUT = REPO_ROOT / "data" / "games" / "nfl_defensive_player_adjusted.parquet"

# Map a player's recorded position to a position group used for
# baseline lookup. EDGEs and OLBs are tagged into both "DE" and "LB"
# pools by the offense baselines build, so the "primary" mapping
# below picks the more natural fit:
POSITION_TO_GROUP = {
    "DE": "DE",
    "EDGE": "DE",
    "DT": "DT",
    "NT": "DT",
    "DL": "DT",
    "LB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "OLB": "LB",  # OLB is more often coverage/run support — group with LB
    "CB": "CB",
    "DB": "CB",
    "S": "S",
    "FS": "S",
    "SS": "S",
    "SAF": "S",
}

ADJUSTED_STATS = [
    "def_tackles_solo", "def_tackle_assists", "def_tackles_for_loss",
    "def_sacks", "def_qb_hits", "def_pass_defended",
    "def_interceptions", "def_fumbles_forced", "def_tackles",
]


def main():
    if not STATS_PATH.exists():
        raise SystemExit(f"Missing {STATS_PATH}. Run `make game-logs-nfl`.")
    if not BASELINES_PATH.exists():
        raise SystemExit(
            f"Missing {BASELINES_PATH}. Run "
            "`python tools/game_logs/build_offense_baselines.py` first."
        )

    print("Loading inputs…")
    stats = pl.read_parquet(STATS_PATH)
    bases = pl.read_parquet(BASELINES_PATH)
    print(f"  stats: {stats.shape[0]:,} rows × {stats.shape[1]} cols")
    print(f"  baselines: {bases.shape[0]:,} rows × {bases.shape[1]} cols\n")

    # Compute combined tackles if missing.
    if "def_tackles" not in stats.columns and "def_tackles_solo" in stats.columns:
        stats = stats.with_columns(
            (pl.col("def_tackles_solo").fill_null(0)
             + pl.col("def_tackle_assists").fill_null(0)).alias("def_tackles")
        )

    # Map position → position_group for the join.
    pos_map_expr = pl.col("position").replace_strict(
        POSITION_TO_GROUP, default=None, return_dtype=pl.String,
    )
    defenders = (
        stats
        .with_columns(pos_map_expr.alias("position_group"))
        .filter(pl.col("position_group").is_not_null())
        .filter(pl.col("opponent_team").is_not_null())
        .filter(pl.col("season").is_not_null())
        .filter(pl.col("def_tackles").fill_null(0) >= 1)  # Same gate as baselines
    )
    print(f"Qualifying defender player-games: {defenders.shape[0]:,}\n")

    # Prepare baselines for the join: each opponent_team in the
    # defender's row IS the offense, so rename for clarity.
    bases_for_join = bases.rename({c: f"{c}_expected"
                                     for c in ADJUSTED_STATS
                                     if c in bases.columns})
    bases_for_join = bases_for_join.rename({"position": "position_group"})
    bases_for_join = bases_for_join.rename({
        "n_games_with_qual": "_off_n_games_with_qual",
        "n_unique_players": "_off_n_unique_players",
        "n_player_games": "_off_n_player_games",
    })
    bases_for_join = bases_for_join.rename({"offense_team": "opponent_team"})

    joined = defenders.join(
        bases_for_join,
        on=["opponent_team", "season", "position_group"],
        how="left",
    )

    # Compute deltas
    delta_exprs = []
    for stat in ADJUSTED_STATS:
        exp_col = f"{stat}_expected"
        delta_col = f"{stat}_delta"
        if stat in joined.columns and exp_col in joined.columns:
            delta_exprs.append(
                pl.when(pl.col(stat).is_not_null()
                         & pl.col(exp_col).is_not_null())
                .then(pl.col(stat) - pl.col(exp_col))
                .otherwise(None)
                .alias(delta_col)
            )
    joined = joined.with_columns(delta_exprs)

    # Slim down output schema
    keep_id = [
        "season", "season_type", "week", "team", "opponent_team",
        "player_id", "player_name", "player_display_name", "position",
        "position_group",
        "_off_n_games_with_qual", "_off_n_unique_players",
        "_off_n_player_games",
    ]
    expected_cols = [f"{s}_expected" for s in ADJUSTED_STATS
                     if f"{s}_expected" in joined.columns]
    delta_cols = [f"{s}_delta" for s in ADJUSTED_STATS
                  if f"{s}_delta" in joined.columns]
    final_cols = (
        [c for c in keep_id if c in joined.columns]
        + [c for c in ADJUSTED_STATS if c in joined.columns]
        + expected_cols + delta_cols
    )
    out = joined.select(final_cols)
    out = out.sort(["season", "week", "team", "position_group",
                    "player_display_name"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUTPUT)
    print(f"✅ wrote {out.shape[0]:,} rows × {out.shape[1]} cols")
    print(f"   → {OUTPUT.relative_to(REPO_ROOT)}\n")

    print("Rows by position group:")
    print(out.group_by("position_group").agg(pl.len().alias("rows"))
             .sort("rows", descending=True).to_pandas().to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

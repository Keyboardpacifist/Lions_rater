#!/usr/bin/env python3
"""
Per-(player, game) advanced-stat counts derived from play-by-play.

Extends the existing explosive-plays layer with offensive-skill metrics
that go beyond raw box scores: chunk completions, deep targets, YAC
chunks, red-zone targets, and first-down receptions. Each gets a
matching defense baseline so the splits expander can show actual /
expected / delta the same way it does for chunk runs.

QB stats (keyed by passer_player_id):
  • chunk_completions  — completed passes ≥20 yards
  • deep_attempts      — pass attempts with air_yards ≥20
  • td_long_passes     — pass TDs of ≥20 yards
  • scramble_first_downs — qb_scramble that converted a first down

Receiver stats (keyed by receiver_player_id — applies to WR / TE / RB):
  • yac_chunks         — completions with yards_after_catch ≥10
  • deep_targets       — pass plays targeting this player with air_yards ≥20
  • rz_targets         — pass plays targeting this player from yardline ≤20
  • first_down_recs    — completions that converted a first down

Outputs:
  data/games/nfl_advanced_player_games.parquet
    one row per (player_id, season, week, team), all stat columns above
    (zeros where a player wasn't involved in that side)

  data/games/nfl_advanced_def_baselines.parquet
    one row per (defense_team, season, position) with the per-qualifying-
    player-game average for each stat (mirrors nfl_defense_baselines)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PBP_PATH = REPO_ROOT / "data" / "games" / "nfl_pbp.parquet"
WSTATS_PATH = REPO_ROOT / "data" / "games" / "nfl_weekly_stats.parquet"
PLAYER_OUT = REPO_ROOT / "data" / "games" / "nfl_advanced_player_games.parquet"
DEF_OUT = REPO_ROOT / "data" / "games" / "nfl_advanced_def_baselines.parquet"

# Same qualifying thresholds the existing baselines use.
POSITION_GROUPS = {
    "QB": {"positions": ["QB"], "gate": "attempts", "min": 10},
    "RB": {"positions": ["RB", "FB"], "gate": "carries", "min": 5},
    "WR": {"positions": ["WR"], "gate": "targets", "min": 2},
    "TE": {"positions": ["TE"], "gate": "targets", "min": 2},
}

ADV_STATS = [
    "chunk_completions", "deep_attempts", "td_long_passes",
    "scramble_first_downs",
    "yac_chunks", "deep_targets", "rz_targets", "first_down_recs",
]


def main():
    if not PBP_PATH.exists():
        raise SystemExit(f"Missing {PBP_PATH}. Run `make game-logs-pbp`.")
    if not WSTATS_PATH.exists():
        raise SystemExit(f"Missing {WSTATS_PATH}. Run `make game-logs-nfl`.")

    print("Loading PBP…")
    pbp = pl.read_parquet(PBP_PATH)
    print(f"  {pbp.shape[0]:,} plays\n")

    # ── Passer-side stats (QB) ──
    print("Computing passer-side stats…")
    pass_plays = pbp.filter(pl.col("play_type") == "pass") \
                    .filter(pl.col("passer_player_id").is_not_null())
    qb_agg = (
        pass_plays
        .with_columns([
            ((pl.col("complete_pass") == 1)
             & (pl.col("yards_gained") >= 20)).cast(pl.Int64).alias("_chunk_cmp"),
            (pl.col("air_yards") >= 20).cast(pl.Int64).fill_null(0).alias("_deep_att"),
            ((pl.col("pass_touchdown") == 1)
             & (pl.col("yards_gained") >= 20)).cast(pl.Int64).alias("_td_long"),
        ])
        .group_by(["season", "week", "posteam", "passer_player_id"])
        .agg([
            pl.col("_chunk_cmp").sum().alias("chunk_completions"),
            pl.col("_deep_att").sum().alias("deep_attempts"),
            pl.col("_td_long").sum().alias("td_long_passes"),
        ])
        .rename({"posteam": "team", "passer_player_id": "player_id"})
    )

    # ── QB scramble first downs (rusher_player_id, qb_scramble==1) ──
    scrambles = pbp.filter(pl.col("qb_scramble") == 1) \
                   .filter(pl.col("rusher_player_id").is_not_null())
    scram_agg = (
        scrambles
        .with_columns(
            (pl.col("first_down") == 1).cast(pl.Int64).alias("_sfd")
        )
        .group_by(["season", "week", "posteam", "rusher_player_id"])
        .agg(pl.col("_sfd").sum().alias("scramble_first_downs"))
        .rename({"posteam": "team", "rusher_player_id": "player_id"})
    )

    qb_full = qb_agg.join(scram_agg,
                           on=["season", "week", "team", "player_id"],
                           how="full", coalesce=True)
    print(f"  passer rows: {qb_full.shape[0]:,}")

    # ── Receiver-side stats (WR / TE / RB) ──
    print("\nComputing receiver-side stats…")
    rec_pool = pbp.filter(pl.col("play_type") == "pass") \
                  .filter(pl.col("receiver_player_id").is_not_null())
    rec_agg = (
        rec_pool
        .with_columns([
            ((pl.col("complete_pass") == 1)
             & (pl.col("yards_after_catch") >= 10))
             .cast(pl.Int64).alias("_yac_chunk"),
            (pl.col("air_yards") >= 20).cast(pl.Int64).fill_null(0).alias("_deep_tgt"),
            (pl.col("yardline_100") <= 20).cast(pl.Int64).fill_null(0).alias("_rz_tgt"),
            ((pl.col("complete_pass") == 1)
             & (pl.col("first_down") == 1)).cast(pl.Int64).alias("_fd_rec"),
        ])
        .group_by(["season", "week", "posteam", "receiver_player_id"])
        .agg([
            pl.col("_yac_chunk").sum().alias("yac_chunks"),
            pl.col("_deep_tgt").sum().alias("deep_targets"),
            pl.col("_rz_tgt").sum().alias("rz_targets"),
            pl.col("_fd_rec").sum().alias("first_down_recs"),
        ])
        .rename({"posteam": "team", "receiver_player_id": "player_id"})
    )
    print(f"  receiver rows: {rec_agg.shape[0]:,}")

    # ── Outer-join into a unified player-game table ──
    player_games = qb_full.join(
        rec_agg, on=["season", "week", "team", "player_id"],
        how="full", coalesce=True,
    )
    # Replace nulls with 0 — a row appearing in only one side simply
    # wasn't involved on the other.
    fill_zero_cols = [c for c in player_games.columns
                      if c not in ("season", "week", "team", "player_id")]
    player_games = player_games.with_columns([
        pl.col(c).fill_null(0).cast(pl.Int64).alias(c) for c in fill_zero_cols
    ])
    print(f"\nMerged: {player_games.shape[0]:,} (player, game) rows")

    PLAYER_OUT.parent.mkdir(parents=True, exist_ok=True)
    player_games.write_parquet(PLAYER_OUT)
    print(f"  ✓ {PLAYER_OUT.relative_to(REPO_ROOT)}")

    # ── Defense baselines: avg per qualifying player-game by position ──
    print("\nBuilding defense baselines…")
    t0 = time.time()
    wstats = pl.read_parquet(WSTATS_PATH).select([
        "player_id", "season", "week", "team", "opponent_team",
        "position", "attempts", "carries", "targets",
    ])
    enriched = wstats.join(
        player_games, on=["season", "week", "team", "player_id"],
        how="left",
    )
    for c in ADV_STATS:
        if c in enriched.columns:
            enriched = enriched.with_columns(pl.col(c).fill_null(0))

    parts = []
    for group_name, cfg in POSITION_GROUPS.items():
        qual = enriched.filter(pl.col("position").is_in(cfg["positions"]))
        qual = qual.filter(pl.col(cfg["gate"]).fill_null(0) >= cfg["min"])
        qual = qual.filter(pl.col("opponent_team").is_not_null())
        agg = (
            qual.group_by(["opponent_team", "season"])
                .agg([
                    pl.col(c).mean().alias(f"avg_{c}") for c in ADV_STATS
                    if c in qual.columns
                ] + [pl.len().alias("n_player_games")])
                .with_columns(pl.lit(group_name).alias("position"))
                .rename({"opponent_team": "defense_team"})
        )
        parts.append(agg)

    defense_baselines = pl.concat(parts, how="diagonal_relaxed")
    defense_baselines = defense_baselines.sort(
        ["defense_team", "season", "position"]
    )
    DEF_OUT.parent.mkdir(parents=True, exist_ok=True)
    defense_baselines.write_parquet(DEF_OUT)
    print(f"  ✓ {DEF_OUT.relative_to(REPO_ROOT)} "
          f"({defense_baselines.shape[0]:,} rows · {(time.time()-t0):.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

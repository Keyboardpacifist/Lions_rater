#!/usr/bin/env python3
"""
Per-targeted-play feed for receivers — joins PBP + participation into
the smallest unit that lets the splits expander build a route ×
coverage matchup profile per receiver.

For every pass play with a non-null receiver_player_id, we pull:
  • route (NGS): GO / HITCH / SLANT / OUT / etc.
  • man-zone label: MAN_COVERAGE / ZONE_COVERAGE
  • coverage shell: COVER_0 / COVER_1 / 2_MAN / etc.
  • result: complete_pass, yards_gained, air_yards,
            yards_after_catch, epa, pass_touchdown
  • game context: season, week, posteam, defteam
  • situation: down, ydstogo, score_differential, shotgun
  • alignment: offense_formation, offense_personnel,
                defense_personnel, defense_bucket
  • blitz signals: number_of_pass_rushers, blitz_bucket,
                    defenders_in_box

Output:
  data/games/nfl_targeted_plays.parquet
    one row per pass play that targeted someone
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PBP_PATH = REPO_ROOT / "data" / "games" / "nfl_pbp.parquet"
PART_PATH = REPO_ROOT / "data" / "games" / "nfl_participation.parquet"
OUT = REPO_ROOT / "data" / "games" / "nfl_targeted_plays.parquet"


def main():
    if not PBP_PATH.exists():
        raise SystemExit(f"Missing {PBP_PATH}.")
    if not PART_PATH.exists():
        raise SystemExit(f"Missing {PART_PATH}.")

    print("Loading inputs…")
    pbp = pl.read_parquet(PBP_PATH)
    part = pl.read_parquet(PART_PATH)

    pbp_slim = pbp.select([
        "game_id", "play_id", "season", "week", "season_type",
        "posteam", "defteam",
        "receiver_player_id", "receiver_player_name",
        "complete_pass", "yards_gained", "air_yards",
        "yards_after_catch", "epa", "pass_touchdown",
        # Situation columns — let the route tree filter by down/distance.
        "down", "ydstogo", "score_differential", "shotgun",
    ]).filter(pl.col("receiver_player_id").is_not_null())

    part_slim = part.select([
        pl.col("nflverse_game_id").alias("game_id"),
        "play_id", "route",
        "defense_man_zone_type", "defense_coverage_type",
        # Alignment + blitz context
        "offense_formation", "offense_personnel",
        "defense_personnel", "defenders_in_box",
        "number_of_pass_rushers",
    ])
    out = pbp_slim.join(part_slim, on=["game_id", "play_id"], how="inner")

    # Normalize coverage labels to friendly names that match what
    # lib_splits already uses elsewhere.
    cov_map = {
        "COVER_0": "Cover-0", "COVER_1": "Cover-1",
        "COVER_2": "Cover-2", "COVER_3": "Cover-3",
        "COVER_4": "Cover-4", "COVER_6": "Cover-6",
        "COVER_9": "Cover-9", "2_MAN": "2-Man",
        "COMBO": "Combo", "PREVENT": "Prevent",
        "BLOWN": None,
    }
    out = out.with_columns(
        pl.col("defense_coverage_type").replace_strict(
            cov_map, default=None, return_dtype=pl.String
        ).alias("coverage_shell")
    )

    # Bucket man/zone for compactness
    mz_map = {"MAN_COVERAGE": "Man", "ZONE_COVERAGE": "Zone"}
    out = out.with_columns(
        pl.col("defense_man_zone_type").replace_strict(
            mz_map, default=None, return_dtype=pl.String
        ).alias("man_zone")
    )

    out = out.rename({"receiver_player_id": "player_id",
                       "posteam": "team",
                       "defteam": "opponent_team"})

    # Derive defense_bucket from defense_personnel (DB count drives it)
    def _defense_bucket(p: str | None) -> str | None:
        if p is None or p == "":
            return None
        try:
            for chunk in str(p).split(","):
                chunk = chunk.strip()
                if chunk.endswith("DB"):
                    n = int(chunk.split(" ", 1)[0])
                    if n >= 7: return "Quarter (7+ DB)"
                    if n == 6: return "Dime (6 DB)"
                    if n == 5: return "Nickel (5 DB)"
                    if n == 4: return "Base (4 DB)"
                    if n == 3: return "Heavy (3 DB)"
                    return f"{n} DB"
        except (ValueError, IndexError):
            pass
        return None

    out = out.with_columns(
        pl.col("defense_personnel").map_elements(
            _defense_bucket, return_dtype=pl.String
        ).alias("defense_bucket")
    )

    # Derive blitz_bucket — 4 rushers = standard, 5 = blitz, 6+ = heavy.
    out = out.with_columns(
        pl.when(pl.col("number_of_pass_rushers").is_null())
          .then(None)
          .when(pl.col("number_of_pass_rushers") <= 3)
            .then(pl.lit("Drop (3 or fewer)"))
          .when(pl.col("number_of_pass_rushers") == 4)
            .then(pl.lit("Standard rush (4)"))
          .when(pl.col("number_of_pass_rushers") == 5)
            .then(pl.lit("Blitz (5)"))
          .otherwise(pl.lit("Heavy blitz (6+)"))
          .alias("blitz_bucket")
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)
    print(f"✅ wrote {out.shape[0]:,} rows × {out.shape[1]} cols")
    print(f"   → {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

#!/usr/bin/env python3
"""
Slim per-play feed for the team game-summary modal.

The summary modal needs every play in a game (runs + passes + kicks
+ penalties) with enough context to render the WP arc, scoring
plays, line score, box score, team stats, and the offensive +
defensive counterfactuals.

Pulling raw pbp + participation live via nflreadpy on every modal
open is slow (~10-15 sec per season on cold cache). This script
pre-builds a slim parquet that loads in milliseconds.

Output: data/game_pbp.parquet (~50 cols × all plays 2016-2025)

Run:
    python tools/build_game_pbp.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "data" / "game_pbp.parquet"

SEASONS = list(range(2016, 2026))

PBP_COLS = [
    # Identification
    "game_id", "play_id", "season", "week", "season_type",
    "home_team", "away_team", "posteam", "defteam",
    # Time
    "qtr", "quarter_seconds_remaining", "game_seconds_remaining",
    # Down / distance / situation
    "down", "ydstogo", "yardline_100", "score_differential",
    "drive", "goal_to_go", "shotgun", "no_huddle", "play_clock",
    # Score progression (for line score)
    "home_score", "away_score",
    "total_home_score", "total_away_score",
    # WP / EPA / success
    "home_wp", "away_wp", "wpa", "epa", "success",
    # Play description + types
    "desc", "play_type",
    "pass_attempt", "rush_attempt", "sack", "qb_scramble", "qb_hit",
    # Pass-specific
    "complete_pass", "interception", "fumble_lost",
    "pass_location", "pass_length", "air_yards",
    "yards_after_catch", "passing_yards", "pass_touchdown",
    "passer_player_name", "passer_player_id",
    "receiver_player_name", "receiver_player_id",
    "cpoe",
    # Rush-specific
    "run_location", "run_gap",
    "rushing_yards", "rush_touchdown",
    "rusher_player_name", "rusher_player_id",
    "first_down", "first_down_pass", "first_down_rush",
    "third_down_converted", "third_down_failed",
    # Scoring
    "touchdown", "td_team",
    "field_goal_result", "safety", "two_point_conv_result",
    # Penalties
    "penalty", "penalty_yards", "penalty_team",
    # Receiving yards (for box score)
    "receiving_yards",
]

PARTICIPATION_COLS = [
    "nflverse_game_id", "play_id",
    "defense_coverage_type", "defense_man_zone_type",
    "number_of_pass_rushers", "defenders_in_box",
    "offense_personnel", "offense_formation",
    "time_to_throw", "was_pressure", "route",
]


def main() -> None:
    import nflreadpy as nfl

    print(f"Pulling pbp + participation for {SEASONS[0]}-{SEASONS[-1]}…")
    pbp_frames = []
    for s in SEASONS:
        print(f"  Season {s}…", end=" ", flush=True)
        try:
            pbp_s = nfl.load_pbp([s]).to_pandas()
            keep = [c for c in PBP_COLS if c in pbp_s.columns]
            pbp_s = pbp_s[keep]
            try:
                part_s = nfl.load_participation([s]).to_pandas()
                part_keep = [c for c in PARTICIPATION_COLS if c in part_s.columns]
                if part_keep:
                    part_slim = part_s[part_keep].rename(
                        columns={"nflverse_game_id": "game_id"}
                    )
                    pbp_s = pbp_s.merge(part_slim,
                                          on=["game_id", "play_id"],
                                          how="left")
            except Exception as e:
                print(f"(participation pull failed: {e}) ", end="")
            pbp_frames.append(pbp_s)
            print(f"{len(pbp_s):,} plays")
        except Exception as e:
            print(f"FAILED: {e}")

    if not pbp_frames:
        raise SystemExit("No pbp data loaded.")
    out = pd.concat(pbp_frames, ignore_index=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # zstd compression keeps the file commit-friendly
    out.to_parquet(OUTPUT, index=False, compression="zstd")
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\n✓ wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(out):,} rows × {out.shape[1]} cols ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

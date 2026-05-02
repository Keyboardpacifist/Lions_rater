"""Build per-defense allowance tables (Defense vs. Position).

Output: data/dvp.parquet

For each (defense team, season, position_group) computes per-game
allowances in receiving yards, receptions, targets, TDs, plus rushing
yards and attempts allowed (RB only). Includes a league-relative
delta column for each metric.

Position groups: WR, TE, RB. (QB DvP is unusual — left out for v1.)

Methodology
-----------
• Each receiving play's `receiver_player_id` is joined to the rosters
  table (season-level) to attach a position.
• Each rush play's `rusher_player_id` is similarly joined.
• Aggregate per (defteam, season, position_group) and divide by games
  played by that defense to get per-game allowances.
• Subtract season league average to produce `*_delta`.

This is the bread-and-butter table behind every fantasy/prop bet that
asks "how friendly is this matchup for [position]?"
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
ROSTERS = REPO / "data" / "nfl_rosters.parquet"
OUT = REPO / "data" / "dvp.parquet"


# Map roster `position` codes to broad groups.
POS_GROUP = {
    "WR": "WR",
    "TE": "TE",
    "RB": "RB", "FB": "RB", "HB": "RB",
}


def _games_played(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (defteam, season): number of distinct game_ids."""
    g = (pbp.dropna(subset=["defteam", "game_id"])
            .groupby(["defteam", "season"])["game_id"]
            .nunique()
            .reset_index()
            .rename(columns={"game_id": "games"}))
    return g


def main() -> None:
    print("→ loading pbp + rosters...")
    pbp = pd.read_parquet(PBP)
    rosters = pd.read_parquet(ROSTERS)
    print(f"  pbp rows: {len(pbp):,}  rosters rows: {len(rosters):,}")

    # Build a (gsis_id, season) -> position_group lookup.
    rosters = rosters[rosters["gsis_id"].notna()].copy()
    rosters["pos_group"] = rosters["position"].map(POS_GROUP)
    rosters = rosters.dropna(subset=["pos_group"])
    # If a player has multiple roster entries in a season (traded), use first.
    pos_lookup = (rosters.drop_duplicates(["gsis_id", "season"])[
        ["gsis_id", "season", "pos_group"]
    ])
    print(f"  position lookup rows: {len(pos_lookup):,}")

    # ── Receiving side ──
    rec = pbp[(pbp["play_type"] == "pass")
              & (pbp["receiver_player_id"].notna())].copy()
    rec = rec.merge(
        pos_lookup,
        left_on=["receiver_player_id", "season"],
        right_on=["gsis_id", "season"],
        how="left",
    )
    rec = rec[rec["pos_group"].notna()]
    print(f"  receiving plays w/ pos: {len(rec):,}")

    rec_agg = (rec.groupby(["defteam", "season", "pos_group"])
               .agg(rec_targets=("pass_attempt", "size"),
                    rec_completions=("complete_pass", "sum"),
                    rec_yards=("receiving_yards", "sum"),
                    rec_tds=("pass_touchdown", "sum"),
                    rec_air_yards=("air_yards", "sum"),
                    rec_yac=("yards_after_catch", "sum"))
               .reset_index())

    # ── Rushing side (RB only — others are noise) ──
    rsh = pbp[(pbp["play_type"] == "run")
              & (pbp["rusher_player_id"].notna())].copy()
    rsh = rsh.merge(
        pos_lookup.rename(columns={"pos_group": "rush_pos"}),
        left_on=["rusher_player_id", "season"],
        right_on=["gsis_id", "season"],
        how="left",
    )
    rsh = rsh[rsh["rush_pos"] == "RB"]
    rsh_agg = (rsh.groupby(["defteam", "season"])
               .agg(rush_attempts=("rush_attempt", "size"),
                    rush_yards=("rushing_yards", "sum"),
                    rush_tds=("rush_touchdown", "sum"))
               .reset_index())
    rsh_agg["pos_group"] = "RB"

    # Merge rushing into receiving by (defteam, season, RB row)
    merged = rec_agg.merge(rsh_agg,
                           on=["defteam", "season", "pos_group"],
                           how="outer")

    # Attach games played for per-game normalization
    games = _games_played(pbp)
    merged = merged.merge(games, on=["defteam", "season"], how="left")
    merged["games"] = merged["games"].fillna(0)

    # Per-game columns
    for col in ["rec_targets", "rec_completions", "rec_yards", "rec_tds",
                "rec_air_yards", "rec_yac",
                "rush_attempts", "rush_yards", "rush_tds"]:
        if col in merged.columns:
            pg = (merged[col].fillna(0) / merged["games"].clip(lower=1))
            merged[f"{col}_pg"] = pg

    # League-relative delta per (season, pos_group)
    pg_cols = [c for c in merged.columns if c.endswith("_pg")]
    for col in pg_cols:
        means = merged.groupby(["season", "pos_group"])[col].transform("mean")
        merged[f"{col}_delta"] = merged[col] - means

    # Drop the gsis_id artifact column from the merge
    merged = merged.drop(columns=[c for c in ["gsis_id", "gsis_id_x",
                                              "gsis_id_y"]
                                  if c in merged.columns])

    merged = merged.sort_values(["season", "pos_group", "defteam"]).reset_index(drop=True)

    print(f"  rows produced: {len(merged):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== 2024 DvP — easiest matchups for WRs (most rec yards "
          "allowed per game) ===")
    sample = (merged[(merged["season"] == 2024) & (merged["pos_group"] == "WR")]
              .sort_values("rec_yards_pg", ascending=False)
              .head(8))
    print(sample[["defteam", "rec_yards_pg", "rec_yards_pg_delta",
                  "rec_targets_pg", "rec_tds_pg"]].to_string())


if __name__ == "__main__":
    main()

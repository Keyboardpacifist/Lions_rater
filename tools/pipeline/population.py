"""
Population selection — snap filtering, top-N selection, roster matching.

Extracted from the identical pattern in wr_data_pull.py:80-117 and
rb_data_pull.py:77-109. One function that handles any position.
"""
from __future__ import annotations

import pandas as pd


def select_population(
    snaps: pd.DataFrame,
    rosters: pd.DataFrame,
    positions: list[str],
    top_n: dict[str, int],
    min_games: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """Select the top-N players by offensive snaps for each position.

    Steps:
      1. Filter snap counts to target positions
      2. Aggregate to player-season totals (off_snaps, games_played)
      3. Apply minimum-games floor
      4. Select top-N per position by snaps
      5. Match to gsis_id via rosters (for PBP joins)

    Args:
        snaps: Raw per-game snap count data from nflverse.
        rosters: Roster data for gsis_id / pfr_id matching.
        positions: e.g., ["WR", "TE"] or ["RB"].
        top_n: e.g., {"WR": 64, "TE": 32}.
        min_games: Minimum games played to be eligible.
        verbose: Print progress.

    Returns:
        DataFrame with columns:
          player, pfr_player_id, position, team, off_snaps,
          games_played, gsis_id, full_name
    """
    # Step 1: Filter to target positions
    pos_snaps = snaps[snaps["position"].isin(positions)].copy()

    # Step 2: Aggregate to player-season totals
    # Some positions may not have 'position' in groupby (e.g., RB where
    # all are one position). Include it when multiple positions are pooled.
    group_cols = ["player", "pfr_player_id", "position", "team"]
    # Only group by columns that exist
    group_cols = [c for c in group_cols if c in pos_snaps.columns]

    snap_totals = (
        pos_snaps.groupby(group_cols, as_index=False)
        .agg(
            off_snaps=("offense_snaps", "sum"),
            games_played=("game_id", "nunique"),
        )
    )
    if verbose:
        print(f"  Total player-seasons: {len(snap_totals)}")

    # Step 3: Minimum-games floor
    eligible = snap_totals[snap_totals["games_played"] >= min_games].copy()
    if verbose:
        print(f"  After {min_games}-game floor: {len(eligible)}")

    # Step 4: Top-N per position
    parts = []
    for pos, n in top_n.items():
        pos_df = eligible[eligible["position"] == pos] if "position" in eligible.columns else eligible
        selected = pos_df.nlargest(n, "off_snaps")
        parts.append(selected)
        if verbose:
            print(f"  Top {n} {pos}: {len(selected)} selected")

    population = pd.concat(parts, ignore_index=True)

    # Step 5: Match to gsis_id via rosters
    roster_slim = (
        rosters[["gsis_id", "pfr_id", "full_name", "position"]]
        .dropna(subset=["gsis_id"])
        .rename(columns={"pfr_id": "pfr_player_id"})
    )
    # Only merge on pfr_player_id + gsis_id + full_name (avoid position clash)
    population = population.merge(
        roster_slim[["pfr_player_id", "gsis_id", "full_name"]],
        on="pfr_player_id",
        how="left",
    )

    # Drop unmatched players
    missing = population["gsis_id"].isna().sum()
    if missing:
        if verbose:
            print(f"  Dropping {missing} players unmatched to PBP gsis_id")
        population = population.dropna(subset=["gsis_id"]).copy()

    if verbose:
        print(f"  Final population: {len(population)} players")

    return population

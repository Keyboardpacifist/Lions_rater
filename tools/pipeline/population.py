"""
Population selection — snap aggregation, qualification, roster matching.

Aggregates snaps to per-player-season totals (across all team stints),
applies a snap floor or top-N, and emits one row per player-season
attributed to the team where the player took the most snaps.

This shape — one row per player-season — matches how nflverse
`load_player_stats(summary_level='reg')` aggregates traded players,
so the per-team-stint counting bug from earlier (Davante Adams etc.)
is sidestepped entirely.
"""
from __future__ import annotations

import pandas as pd


def select_population(
    snaps: pd.DataFrame,
    rosters: pd.DataFrame,
    positions: list[str],
    min_games: int,
    top_n: dict[str, int] | None = None,
    snap_floor: dict[str, int] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Select qualifying players for a season.

    Steps:
      1. Filter snap counts to target positions
      2. Aggregate to per-player-season totals (sum across team stints)
      3. Apply minimum-games floor
      4. Apply snap_floor (preferred) OR top_n selection per position
      5. Attribute each player to the team with their most snaps
      6. Match to gsis_id via rosters

    Args:
        snaps: Raw per-game snap count data from nflverse.
        rosters: Roster data for gsis_id / pfr_id matching.
        positions: e.g., ["WR", "TE"] or ["RB"].
        min_games: Minimum games played to be eligible.
        top_n: Legacy. {"WR": 64, "TE": 32} = top N by total snaps per position.
        snap_floor: Preferred. {"WR": 100, "TE": 100} = all players with >=N total snaps.
        verbose: Print progress.

    Returns:
        DataFrame, one row per qualifying player-season, with columns:
          player, pfr_player_id, position, team, off_snaps,
          games_played, gsis_id, full_name
        where `team` = team with the most snaps for that player-season.
    """
    if (top_n is None) == (snap_floor is None):
        raise ValueError(
            "Provide exactly one of top_n or snap_floor (got "
            f"top_n={top_n}, snap_floor={snap_floor})"
        )

    # Step 1: Filter to target positions
    pos_snaps = snaps[snaps["position"].isin(positions)].copy()

    # Step 2a: Per-team-stint totals (we need these to find each player's
    # primary team and to know per-team snap distribution)
    per_stint = (
        pos_snaps.groupby(
            ["player", "pfr_player_id", "position", "team"],
            as_index=False,
        )
        .agg(
            stint_snaps=("offense_snaps", "sum"),
            stint_games=("game_id", "nunique"),
        )
    )

    # Step 2b: Aggregate to per-player-season totals (across all stints)
    per_player = (
        per_stint.groupby(
            ["player", "pfr_player_id", "position"],
            as_index=False,
        )
        .agg(
            off_snaps=("stint_snaps", "sum"),
            games_played=("stint_games", "sum"),
        )
    )
    if verbose:
        print(f"  Total player-seasons: {len(per_player)}")

    # Step 3: Minimum-games floor
    eligible = per_player[per_player["games_played"] >= min_games].copy()
    if verbose:
        print(f"  After {min_games}-game floor: {len(eligible)}")

    # Step 4: Position-by-position selection
    parts = []
    for pos in positions:
        pos_df = eligible[eligible["position"] == pos]
        if snap_floor is not None:
            floor = snap_floor.get(pos)
            if floor is None:
                if verbose:
                    print(f"  No snap_floor entry for {pos}; skipping")
                continue
            selected = pos_df[pos_df["off_snaps"] >= floor]
            if verbose:
                print(f"  {pos} >= {floor} snaps: {len(selected)} selected")
        else:
            n = top_n.get(pos)
            if n is None:
                if verbose:
                    print(f"  No top_n entry for {pos}; skipping")
                continue
            selected = pos_df.nlargest(n, "off_snaps")
            if verbose:
                print(f"  Top {n} {pos}: {len(selected)} selected")
        parts.append(selected)

    if not parts:
        return pd.DataFrame()

    population = pd.concat(parts, ignore_index=True)

    # Step 5: Pick each player's primary team (the one with most snaps)
    primary_team = (
        per_stint.sort_values("stint_snaps", ascending=False)
        .drop_duplicates(subset=["player", "pfr_player_id", "position"])
        [["player", "pfr_player_id", "position", "team"]]
    )
    population = population.merge(
        primary_team,
        on=["player", "pfr_player_id", "position"],
        how="left",
    )

    # Step 6: Match to gsis_id via rosters
    roster_slim = (
        rosters[["gsis_id", "pfr_id", "full_name", "position"]]
        .dropna(subset=["gsis_id"])
        .rename(columns={"pfr_id": "pfr_player_id"})
    )
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

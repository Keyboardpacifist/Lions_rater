"""
Population selection — snap aggregation, qualification, per-team-stint output.

A player qualifies based on TOTAL snaps across all team stints (so traded
players are not silently dropped). The output emits one row per
(player, team) stint, so each team's page accurately reflects what that
player did for that team.

Example: Davante Adams 2024 had 178 snaps for LV and 614 snaps for NYJ
(792 total). With snap_floor={"WR": 100}, he qualifies on the 792 total
and produces two output rows — one for LV and one for NYJ — each with
that team's per-stint snap count.
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
    """Select qualifying players, emit one row per (player, team) stint.

    Steps:
      1. Filter snap counts to target positions
      2. Aggregate per (player, team) stint and per player total
      3. Apply minimum-games floor on TOTAL games
      4. Apply snap_floor (preferred) OR top_n on TOTAL snaps per position
      5. Restrict per-stint table to qualifying players
      6. Match to gsis_id via rosters

    Args:
        snaps: Raw per-game snap count data from nflverse.
        rosters: Roster data for gsis_id / pfr_id matching.
        positions: e.g., ["WR", "TE"] or ["RB"].
        min_games: Minimum total games played to be eligible.
        top_n: Legacy. {"WR": 64, "TE": 32} = top N players by total snaps per position.
        snap_floor: Preferred. {"WR": 100, "TE": 100} = all players with >=N total snaps.
        verbose: Print progress.

    Returns:
        DataFrame, one row per qualifying player-team-stint, with columns:
          player, pfr_player_id, position, team,
          off_snaps (stint snaps), games_played (stint games),
          total_snaps, total_games (full-season context),
          gsis_id, full_name
    """
    if (top_n is None) == (snap_floor is None):
        raise ValueError(
            "Provide exactly one of top_n or snap_floor (got "
            f"top_n={top_n}, snap_floor={snap_floor})"
        )

    # Step 1: Filter to target positions
    pos_snaps = snaps[snaps["position"].isin(positions)].copy()

    # Step 2a: Per-team-stint totals
    per_stint = (
        pos_snaps.groupby(
            ["player", "pfr_player_id", "position", "team"],
            as_index=False,
        )
        .agg(
            off_snaps=("offense_snaps", "sum"),
            games_played=("game_id", "nunique"),
        )
    )

    # Step 2b: Per-player totals (for qualification)
    per_player_total = (
        per_stint.groupby(
            ["player", "pfr_player_id", "position"],
            as_index=False,
        )
        .agg(
            total_snaps=("off_snaps", "sum"),
            total_games=("games_played", "sum"),
        )
    )
    if verbose:
        print(f"  Total player-seasons: {len(per_player_total)}")

    # Step 3: Minimum-games floor (on total games)
    eligible = per_player_total[per_player_total["total_games"] >= min_games].copy()
    if verbose:
        print(f"  After {min_games}-game floor: {len(eligible)}")

    # Step 4: Position-by-position selection on TOTAL snaps
    qualifying_parts = []
    for pos in positions:
        pos_df = eligible[eligible["position"] == pos]
        if snap_floor is not None:
            floor = snap_floor.get(pos)
            if floor is None:
                if verbose:
                    print(f"  No snap_floor entry for {pos}; skipping")
                continue
            kept = pos_df[pos_df["total_snaps"] >= floor]
            if verbose:
                print(f"  {pos} >= {floor} total snaps: {len(kept)} qualify")
        else:
            n = top_n.get(pos)
            if n is None:
                if verbose:
                    print(f"  No top_n entry for {pos}; skipping")
                continue
            kept = pos_df.nlargest(n, "total_snaps")
            if verbose:
                print(f"  Top {n} {pos} by total snaps: {len(kept)} qualify")
        qualifying_parts.append(kept)

    if not qualifying_parts:
        return pd.DataFrame()

    qualifying = pd.concat(qualifying_parts, ignore_index=True)

    # Step 5: Restrict per-stint table to qualifying players
    # Inner join on (player, pfr_player_id, position) — preserves per-team stint rows.
    population = per_stint.merge(
        qualifying[
            ["player", "pfr_player_id", "position", "total_snaps", "total_games"]
        ],
        on=["player", "pfr_player_id", "position"],
        how="inner",
    )

    if verbose:
        n_traded = (
            population.groupby(["player", "pfr_player_id", "position"])
            .size()
            .gt(1)
            .sum()
        )
        print(
            f"  Per-stint rows: {len(population)} "
            f"({n_traded} qualifying players had multiple team stints)"
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
            print(f"  Dropping {missing} stint rows unmatched to PBP gsis_id")
        population = population.dropna(subset=["gsis_id"]).copy()

    if verbose:
        print(f"  Final population: {len(population)} stint rows")

    return population

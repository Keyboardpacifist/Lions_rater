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
    snap_column: str = "offense_snaps",
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

    # Step 0: Augment snap_counts with rosters' depth_chart_position to
    # disambiguate generic codes. Match primarily on pfr_id (most reliable
    # — survives name format quirks like "DJ Reader" vs "D.J. Reader"),
    # then fall back to (name, team).
    if "depth_chart_position" in rosters.columns:
        roster_dcp = rosters[
            [c for c in ["full_name", "team", "pfr_id", "depth_chart_position"]
             if c in rosters.columns]
        ].dropna(subset=["depth_chart_position"])

        # Primary: by pfr_id when both sides have it
        if "pfr_id" in roster_dcp.columns and "pfr_player_id" in snaps.columns:
            by_pfr = (
                roster_dcp.dropna(subset=["pfr_id"])[["pfr_id", "depth_chart_position"]]
                .drop_duplicates(subset=["pfr_id"])
                .rename(columns={"depth_chart_position": "_dcp_pfr"})
            )
            snaps = snaps.merge(
                by_pfr, left_on="pfr_player_id", right_on="pfr_id", how="left"
            )
            snaps = snaps.drop(columns=[c for c in ["pfr_id"] if c in snaps.columns])
        else:
            snaps["_dcp_pfr"] = pd.NA

        # Fallback: by (name, team) for rows the pfr_id match missed
        by_name = (
            roster_dcp.drop_duplicates(subset=["full_name", "team"])
            [["full_name", "team", "depth_chart_position"]]
            .rename(columns={"depth_chart_position": "_dcp_name"})
        )
        snaps = snaps.merge(
            by_name, left_on=["player", "team"], right_on=["full_name", "team"], how="left"
        )
        snaps["effective_position"] = (
            snaps["_dcp_pfr"]
            .fillna(snaps.get("_dcp_name"))
            .fillna(snaps["position"])
        )
        snaps = snaps.drop(columns=[c for c in ["_dcp_pfr", "_dcp_name", "full_name"]
                                     if c in snaps.columns])
    else:
        snaps["effective_position"] = snaps["position"]

    # Step 1: Filter to target positions (using effective_position)
    pos_snaps = snaps[snaps["effective_position"].isin(positions)].copy()
    # Use effective_position as the canonical position from here forward
    pos_snaps["position"] = pos_snaps["effective_position"]

    # Step 2a: Per-team-stint totals
    if snap_column not in pos_snaps.columns:
        if verbose:
            print(f"  WARNING: snap column '{snap_column}' missing from snap_counts; "
                  f"falling back to offense_snaps")
        snap_column = "offense_snaps"
    per_stint = (
        pos_snaps.groupby(
            ["player", "pfr_player_id", "position", "team"],
            as_index=False,
        )
        .agg(
            off_snaps=(snap_column, "sum"),
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

    # Step 6: Match to gsis_id via rosters.
    # Primary: join on pfr_player_id (most reliable when present).
    # Fallback: for rows where rosters lacks pfr_id (common for punters
    # and other special-teamers), join by (player name, team).
    roster_slim = (
        rosters[["gsis_id", "pfr_id", "full_name", "position", "team"]]
        .dropna(subset=["gsis_id"])
        .rename(columns={"pfr_id": "pfr_player_id"})
    )
    by_pfr = (
        roster_slim.dropna(subset=["pfr_player_id"])
        [["pfr_player_id", "gsis_id", "full_name"]]
    )
    population = population.merge(by_pfr, on="pfr_player_id", how="left")

    # Name+team fallback for unmatched rows
    if population["gsis_id"].isna().any():
        by_name = (
            roster_slim[["full_name", "team", "gsis_id"]]
            .drop_duplicates(subset=["full_name", "team"])
            .rename(columns={"gsis_id": "_fb_gsis_id", "full_name": "_fb_full_name"})
        )
        population = population.merge(
            by_name,
            left_on=["player", "team"],
            right_on=["_fb_full_name", "team"],
            how="left",
        )
        # Use fallback gsis_id where primary is null
        population["gsis_id"] = population["gsis_id"].fillna(population["_fb_gsis_id"])
        population["full_name"] = population["full_name"].fillna(population["_fb_full_name"])
        population = population.drop(columns=["_fb_gsis_id", "_fb_full_name"])

    # Drop unmatched players
    missing = population["gsis_id"].isna().sum()
    if missing:
        if verbose:
            print(f"  Dropping {missing} stint rows unmatched to PBP gsis_id")
        population = population.dropna(subset=["gsis_id"]).copy()

    if verbose:
        print(f"  Final population: {len(population)} stint rows")

    return population

"""
Season-loop orchestrator.

For each season: pull data → build population → merge pre-aggregated
player_stats (counting stats) → aggregate PBP advanced stats → merge
NGS/PFR sources → compute derived stats → z-score → stack.

After all seasons are processed, write the combined output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import PositionConfig
from .output import write_metadata, write_parquet
from .population import select_population
from .sources import (
    load_ngs,
    load_pbp,
    load_pfr,
    load_player_stats,
    load_rosters,
    load_snap_counts,
)
from .zscore import zscore_stats


def run_season(
    config: PositionConfig,
    season: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the pipeline for a single season."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Season {season}")
        print(f"{'='*60}")

    # ── Pull data ────────────────────────────────────────────────
    if verbose:
        print("\nPulling data...")
    pbp = load_pbp(season, verbose=verbose)
    snaps = load_snap_counts(season, verbose=verbose)
    rosters = load_rosters(season, verbose=verbose)

    if pbp.empty or snaps.empty or rosters.empty:
        if verbose:
            print(f"  SKIPPING season {season}: missing core data")
        return pd.DataFrame()

    # ── Build population ─────────────────────────────────────────
    if verbose:
        print("\nBuilding population...")
    population = select_population(
        snaps=snaps,
        rosters=rosters,
        positions=config.snap_positions,
        min_games=config.min_games,
        top_n=config.top_n,
        snap_floor=config.snap_floor,
        verbose=verbose,
    )

    if population.empty:
        if verbose:
            print(f"  SKIPPING season {season}: empty population")
        return pd.DataFrame()

    # ── Merge pre-aggregated player_stats per (player, team) stint ──
    if config.use_player_stats and config.player_stats_col_map:
        if verbose:
            print("\nMerging player_stats per team stint...")
        # Pull WEEK-level so we can re-aggregate per (player, team) stint.
        # summary_level='reg' would already collapse traded players to one row.
        ps_week = load_player_stats(season, summary_level="week", verbose=verbose)
        if not ps_week.empty and "player_id" in ps_week.columns and "team" in ps_week.columns:
            # Filter to configured season types (REG by default)
            if "season_type" in ps_week.columns:
                ps_week = ps_week[ps_week["season_type"].isin(config.pbp_season_types)]

            # Counting stats: sum across weeks per (player, team) stint
            counting_in_map = {
                "targets", "receptions", "receiving_yards", "receiving_tds",
                "receiving_first_downs", "receiving_air_yards",
                "receiving_yards_after_catch", "receiving_2pt_conversions",
                "receiving_fumbles", "receiving_fumbles_lost",
                "carries", "rushing_yards", "rushing_tds", "rushing_first_downs",
                "rushing_fumbles", "rushing_fumbles_lost",
                "completions", "attempts", "passing_yards", "passing_tds",
                "passing_interceptions", "sacks_suffered", "sack_yards_lost",
                "passing_first_downs", "passing_air_yards",
                "passing_yards_after_catch",
            }
            sum_cols = [c for c in config.player_stats_col_map if c in counting_in_map and c in ps_week.columns]
            agg_dict = {c: (c, "sum") for c in sum_cols}
            # Always track first/last week per stint for chronological ordering
            # (used by the career arc viz to draw trades in the correct order)
            if "week" in ps_week.columns:
                agg_dict["first_week"] = ("week", "min")
                agg_dict["last_week"] = ("week", "max")
            ps_stint = ps_week.groupby(["player_id", "team"], as_index=False).agg(**agg_dict) if agg_dict else ps_week.groupby(["player_id", "team"], as_index=False).first()

            # Per-stint team-level totals (for rate stats like target_share).
            # team_targets_in_stint = sum of all team targets across the weeks
            # this player was on this team. Same for air_yards.
            if "targets" in ps_week.columns or "receiving_air_yards" in ps_week.columns:
                team_week_agg = {}
                if "targets" in ps_week.columns:
                    team_week_agg["team_week_targets"] = ("targets", "sum")
                if "receiving_air_yards" in ps_week.columns:
                    team_week_agg["team_week_air_yards"] = ("receiving_air_yards", "sum")
                team_week = ps_week.groupby(["team", "week"], as_index=False).agg(**team_week_agg)

                # For each player-team stint, find weeks they were on that team
                stint_weeks = ps_week[["player_id", "team", "week"]].drop_duplicates()
                stint_with_team_totals = stint_weeks.merge(team_week, on=["team", "week"])
                stint_team_totals = stint_with_team_totals.groupby(
                    ["player_id", "team"], as_index=False
                ).agg(
                    team_targets_in_stint=("team_week_targets", "sum") if "team_week_targets" in team_week.columns else ("week", "first"),
                    team_air_yards_in_stint=("team_week_air_yards", "sum") if "team_week_air_yards" in team_week.columns else ("week", "first"),
                )
                ps_stint = ps_stint.merge(
                    stint_team_totals, on=["player_id", "team"], how="left"
                )

                # Per-stint rate stats — recomputed from sums (correct, not approximated)
                if "targets" in ps_stint.columns and "team_targets_in_stint" in ps_stint.columns:
                    ps_stint["target_share"] = (
                        ps_stint["targets"] / ps_stint["team_targets_in_stint"].replace(0, np.nan)
                    )
                if (
                    "receiving_air_yards" in ps_stint.columns
                    and "team_air_yards_in_stint" in ps_stint.columns
                ):
                    ps_stint["air_yards_share"] = (
                        ps_stint["receiving_air_yards"]
                        / ps_stint["team_air_yards_in_stint"].replace(0, np.nan)
                    )
                if (
                    "receiving_yards" in ps_stint.columns
                    and "receiving_air_yards" in ps_stint.columns
                ):
                    ps_stint["racr"] = (
                        ps_stint["receiving_yards"]
                        / ps_stint["receiving_air_yards"].replace(0, np.nan)
                    )
                if "target_share" in ps_stint.columns and "air_yards_share" in ps_stint.columns:
                    ps_stint["wopr"] = 1.5 * ps_stint["target_share"] + 0.7 * ps_stint["air_yards_share"]

            # Select + rename to output columns and merge on (gsis_id, team)
            available = {
                k: v for k, v in config.player_stats_col_map.items() if k in ps_stint.columns
            }
            cols_to_select = ["player_id", "team"] + list(available.keys())
            ps_slim = ps_stint[cols_to_select].rename(
                columns={"player_id": "gsis_id", **available}
            )
            population = population.merge(ps_slim, on=["gsis_id", "team"], how="left")
            if verbose:
                print(f"  Merged player_stats columns per stint: {list(available.values())}")
        elif verbose:
            print("  player_stats unavailable for this season")

    # ── Filter PBP: play types + season types (Bug 3 fix) ────────
    pbp_filtered = pbp[
        (pbp["play_type"].isin(config.pbp_play_types))
        & (pbp["season_type"].isin(config.pbp_season_types))
    ].copy()
    if verbose:
        print(
            f"\nPBP filter: play_type in {config.pbp_play_types}, "
            f"season_type in {config.pbp_season_types} → {len(pbp_filtered):,} plays"
        )

    # ── Aggregate PBP advanced stats ─────────────────────────────
    if config.aggregate_stats:
        if verbose:
            print("\nAggregating PBP advanced stats...")
        for agg_fn in config.aggregate_stats:
            stats_df = agg_fn(pbp_filtered, population)
            if not stats_df.empty and "gsis_id" in stats_df.columns:
                merge_keys = ["gsis_id"]
                if "team" in stats_df.columns and "team" in population.columns:
                    merge_keys.append("team")
                population = population.merge(stats_df, on=merge_keys, how="left")

    # ── Merge NGS stats ──────────────────────────────────────────
    if config.ngs_stat_type and config.ngs_col_map:
        if verbose:
            print("\nMerging NGS stats...")
        ngs = load_ngs(season, config.ngs_stat_type, verbose=verbose)
        if not ngs.empty and "player_gsis_id" in ngs.columns:
            available = {
                k: v for k, v in config.ngs_col_map.items() if k in ngs.columns
            }
            cols_to_select = ["player_gsis_id"] + [
                k for k in available if k != "player_gsis_id"
            ]
            ngs_slim = ngs[cols_to_select].rename(
                columns={"player_gsis_id": "gsis_id", **available}
            )
            population = population.merge(ngs_slim, on="gsis_id", how="left")
            if verbose:
                print(f"  Merged NGS columns: {list(available.values())}")
        else:
            for col in config.ngs_col_map.values():
                if col not in population.columns:
                    population[col] = np.nan

    # ── Merge PFR stats ──────────────────────────────────────────
    if config.pfr_stat_type and config.pfr_col_map:
        if verbose:
            print("\nMerging PFR stats...")
        pfr = load_pfr(season, config.pfr_stat_type, verbose=verbose)
        if not pfr.empty:
            pfr_id_col = (
                "pfr_player_id"
                if "pfr_player_id" in pfr.columns
                else "pfr_id" if "pfr_id" in pfr.columns else None
            )
            if pfr_id_col:
                available = {
                    k: v for k, v in config.pfr_col_map.items() if k in pfr.columns
                }
                cols_to_select = [pfr_id_col] + list(available.keys())
                cols_to_select = [c for c in cols_to_select if c in pfr.columns]
                pfr_slim = pfr[cols_to_select].rename(
                    columns={pfr_id_col: "pfr_player_id", **available}
                )
                population = population.merge(
                    pfr_slim, on="pfr_player_id", how="left"
                )
                if verbose:
                    print(f"  Merged PFR columns: {list(available.values())}")
        else:
            for col in config.pfr_col_map.values():
                if col not in population.columns:
                    population[col] = np.nan

    # ── Compute derived stats ────────────────────────────────────
    if verbose:
        print("\nComputing derived stats...")
    population = config.compute_derived(population)

    # ── Z-score (Decision 2: optionally per position group) ──────
    if verbose:
        print("\nZ-scoring...")
    if config.zscore_groups:
        groups = []
        for pos in config.zscore_groups:
            pos_df = population[population["position"] == pos].copy()
            if pos_df.empty:
                continue
            pos_df = zscore_stats(
                pos_df, stats=config.stats_to_zscore, invert=config.invert_stats
            )
            if verbose:
                print(f"  Z-scored {pos}: {len(pos_df)} players")
            groups.append(pos_df)
        # Include any players whose position falls outside zscore_groups
        # (shouldn't happen for current configs, but keep them)
        outside = population[~population["position"].isin(config.zscore_groups)]
        if not outside.empty:
            for stat in config.stats_to_zscore:
                outside[f"{stat}_z"] = np.nan
            groups.append(outside)
        population = pd.concat(groups, ignore_index=True)
    else:
        population = zscore_stats(
            population,
            stats=config.stats_to_zscore,
            invert=config.invert_stats,
        )

    # ── Normalize column names for pages ─────────────────────────
    population["player_display_name"] = population.get(
        "full_name", population.get("player", "")
    )
    population["player_id"] = population["gsis_id"]
    population["recent_team"] = population["team"]
    population["season_year"] = season
    population["games"] = population["games_played"]

    if verbose:
        print(f"\n  Season {season}: {len(population)} players processed")

    return population


def run_pipeline(
    config: PositionConfig,
    seasons: list[int],
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Run the full multi-season pipeline.

    For each season: pull → populate → merge → aggregate → derive →
    zscore. Then stack all seasons and write output.
    """
    if verbose:
        print(f"Pipeline: {config.key.upper()}")
        print(f"Seasons: {seasons}")
        print(f"Output: {output_dir}")
        if dry_run:
            print("DRY RUN — will not write files")

    all_seasons = []
    for season in seasons:
        result = run_season(config, season, verbose=verbose)
        if not result.empty:
            all_seasons.append(result)

    if not all_seasons:
        print("\nERROR: No seasons produced data. Nothing to write.")
        return

    combined = pd.concat(all_seasons, ignore_index=True)
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Combined: {len(combined)} rows across {len(all_seasons)} seasons")
        print(f"{'='*60}")

    if dry_run:
        print("\nDry run complete. Would write:")
        for f in config.output_filenames:
            print(f"  {output_dir / f}")
        print(f"  {output_dir / config.metadata_filename}")
        return

    if verbose:
        print("\nWriting outputs...")
    write_parquet(combined, config, output_dir, verbose=verbose)
    write_metadata(
        combined,
        config,
        output_dir,
        season=f"{min(seasons)}-{max(seasons)}",
        verbose=verbose,
    )

    if verbose:
        print("\nDone.")

"""
Season-loop orchestrator.

For each season: pull data → build population → aggregate PBP stats →
merge advanced sources → compute derived stats → z-score → stack.

After all seasons are processed, write the combined output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import PositionConfig
from .output import write_metadata, write_parquet
from .population import select_population
from .sources import load_ngs, load_pbp, load_pfr, load_rosters, load_snap_counts
from .zscore import zscore_stats


def run_season(
    config: PositionConfig,
    season: int,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the pipeline for a single season.

    Returns a DataFrame with all stats + z-scores for one season.
    """
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
        top_n=config.top_n,
        min_games=config.min_games,
        verbose=verbose,
    )

    if population.empty:
        if verbose:
            print(f"  SKIPPING season {season}: empty population")
        return pd.DataFrame()

    # ── Filter PBP to relevant play types ────────────────────────
    pbp_filtered = pbp[pbp["play_type"].isin(config.pbp_play_types)].copy()

    # ── Aggregate PBP stats ──────────────────────────────────────
    if verbose:
        print("\nAggregating PBP stats...")
    for agg_fn in config.aggregate_stats:
        stats_df = agg_fn(pbp_filtered, population)
        if not stats_df.empty and "gsis_id" in stats_df.columns:
            population = population.merge(stats_df, on="gsis_id", how="left")

    # ── Merge NGS stats ──────────────────────────────────────────
    if config.ngs_stat_type and config.ngs_col_map:
        if verbose:
            print("\nMerging NGS stats...")
        ngs = load_ngs(season, config.ngs_stat_type, verbose=verbose)
        if not ngs.empty and "player_gsis_id" in ngs.columns:
            available = {
                k: v for k, v in config.ngs_col_map.items() if k in ngs.columns
            }
            # Always need the join key
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
            # Fill NaN for expected NGS columns
            for col in config.ngs_col_map.values():
                if col not in population.columns:
                    population[col] = np.nan

    # ── Merge PFR stats ──────────────────────────────────────────
    if config.pfr_stat_type and config.pfr_col_map:
        if verbose:
            print("\nMerging PFR stats...")
        pfr = load_pfr(season, config.pfr_stat_type, verbose=verbose)
        if not pfr.empty:
            # PFR uses 'pfr_id' or 'pfr_player_id'
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

    # ── Z-score ──────────────────────────────────────────────────
    if verbose:
        print("\nZ-scoring...")
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

    For each season: pull → populate → aggregate → merge → derive →
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

    # Stack all seasons
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

    # Write outputs
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

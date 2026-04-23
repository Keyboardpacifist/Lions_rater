"""
PositionConfig — declarative definition of a position's data pipeline.

Each position provides:
- What to filter from snap counts
- How to aggregate PBP stats
- What advanced sources to merge
- What derived stats to compute
- What to z-score
- Metadata for the UI (tiers, labels, methodology)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass
class PositionConfig:
    # ── Identity ─────────────────────────────────────────────
    key: str                        # "wr", "rb", "qb", etc.
    output_filenames: list[str]     # ["league_wr_all_seasons.parquet", "league_te_all_seasons.parquet"]
    metadata_filename: str          # "wr_te_stat_metadata.json"

    # ── Population selection ─────────────────────────────────
    snap_positions: list[str]       # ["WR", "TE"] or ["RB"]
    min_games: int                  # 6
    # Provide exactly one selection mechanism:
    #   - top_n: legacy {"WR": 64, "TE": 32} — pick top N by total snaps
    #   - snap_floor: {"WR": 100, "TE": 100} — pick all players >= N total snaps
    # snap_floor is preferred (Decision 1, 2026-04-22). top_n kept for back-compat.
    top_n: dict[str, int] | None = None
    snap_floor: dict[str, int] | None = None

    # ── Data sources ─────────────────────────────────────────
    pbp_play_types: list[str] = field(default_factory=list)     # ["pass"] / ["run"] / ["run", "pass"]
    pbp_season_types: list[str] = field(default_factory=lambda: ["REG"])  # filter PBP to these season_type values
    ngs_stat_type: str | None = None        # "receiving", "rushing", or None
    pfr_stat_type: str | None = None        # "rush", "rec", "def", or None

    # ── Pre-aggregated player_stats (Decision 3 hybrid) ──────
    # When True, load nflverse player_stats and merge mapped columns onto
    # the per-team-stint population. Use for counting stats (targets,
    # receiving_yards, receiving_tds, etc.) so PBP's lateral-TD bug and
    # cross-team aggregation are sidestepped.
    use_player_stats: bool = False
    # Map: nflverse player_stats column name -> output column name.
    player_stats_col_map: dict[str, str] = field(default_factory=dict)

    # ── PBP aggregation ──────────────────────────────────────
    # Each function: (pbp_filtered: DataFrame, population: DataFrame) -> DataFrame
    # Returns a DataFrame with 'gsis_id' column + aggregated stats. Under
    # the hybrid model these compute *advanced* stats only (success_rate,
    # epa, first_downs, etc.) — counting stats come from player_stats.
    aggregate_stats: list[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = field(default_factory=list)

    # ── Advanced source merging ──────────────────────────────
    ngs_col_map: dict[str, str] = field(default_factory=dict)   # ngs_col -> output_col
    pfr_col_map: dict[str, str] = field(default_factory=dict)   # pfr_col -> output_col

    # ── Derived stats ────────────────────────────────────────
    # Function: (population: DataFrame) -> DataFrame
    compute_derived: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df

    # ── Z-scoring ────────────────────────────────────────────
    stats_to_zscore: list[str] = field(default_factory=list)
    invert_stats: set[str] = field(default_factory=set)
    # When set (e.g., ["WR", "TE"]), z-score each position group
    # independently before recombining (Decision 2: separate reference,
    # combined output). When None, z-score the whole population together.
    zscore_groups: list[str] | None = None

    # ── Output ───────────────────────────────────────────────
    output_columns: list[str] = field(default_factory=list)

    # ── Metadata ─────────────────────────────────────────────
    stat_tiers: dict[str, int] = field(default_factory=dict)
    stat_labels: dict[str, str] = field(default_factory=dict)
    stat_methodology: dict[str, dict] = field(default_factory=dict)

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
    top_n: dict[str, int]           # {"WR": 64, "TE": 32}
    min_games: int                  # 6

    # ── Data sources ─────────────────────────────────────────
    pbp_play_types: list[str]       # ["pass"] or ["run"] or ["run", "pass"]
    ngs_stat_type: str | None       # "receiving", "rushing", or None
    pfr_stat_type: str | None       # "rush", "rec", "def", or None

    # ── PBP aggregation ──────────────────────────────────────
    # Each function: (pbp_filtered: DataFrame, population: DataFrame) -> DataFrame
    # Returns a DataFrame with 'gsis_id' column + aggregated stats
    aggregate_stats: list[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]]

    # ── Advanced source merging ──────────────────────────────
    ngs_col_map: dict[str, str] = field(default_factory=dict)   # ngs_col -> output_col
    pfr_col_map: dict[str, str] = field(default_factory=dict)   # pfr_col -> output_col

    # ── Derived stats ────────────────────────────────────────
    # Function: (population: DataFrame) -> DataFrame
    compute_derived: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df

    # ── Z-scoring ────────────────────────────────────────────
    stats_to_zscore: list[str] = field(default_factory=list)
    invert_stats: set[str] = field(default_factory=set)

    # ── Output ───────────────────────────────────────────────
    output_columns: list[str] = field(default_factory=list)

    # ── Metadata ─────────────────────────────────────────────
    stat_tiers: dict[str, int] = field(default_factory=dict)
    stat_labels: dict[str, str] = field(default_factory=dict)
    stat_methodology: dict[str, dict] = field(default_factory=dict)

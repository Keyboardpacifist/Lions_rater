"""
TEMPLATE — Copy this file to add a new position to the pipeline.

Steps:
1. Copy this file to positions/{position}.py (e.g., positions/qb.py)
2. Fill in the aggregation function(s), derived stats, and config
3. Add to positions/__init__.py:
     from .qb import QB_CONFIG
     POSITIONS["qb"] = QB_CONFIG
4. Test: python tools/data_pull.py --position qb --seasons 2024 --verbose
5. Compare output columns to existing data/league_qb_all_seasons.parquet

Look at wr.py for a complete working example.
Look at rb.py for a more complex example (two aggregation functions + PFR + NGS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import PositionConfig


# ── PBP aggregation ──────────────────────────────────────────────────────────
# This function receives ALL plays matching your pbp_play_types (e.g., "pass",
# "run") and the population DataFrame. It should:
#   1. Filter to relevant plays (e.g., where rusher_player_id is not null)
#   2. Group by the player ID column (e.g., "rusher_player_id")
#   3. Compute per-player stats
#   4. Return a DataFrame with 'gsis_id' column + stat columns


def agg_POSITION(pbp: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player stats from PBP.

    TODO: Replace with your position's aggregation logic.
    """
    # Example for a rushing position:
    #
    # rush_plays = pbp[
    #     (pbp["play_type"] == "run") & (pbp["rusher_player_id"].notna())
    # ].copy()
    #
    # if rush_plays.empty:
    #     return pd.DataFrame()
    #
    # def _agg(group):
    #     return pd.Series({
    #         "rush_yards": group["yards_gained"].fillna(0).sum(),
    #         "carries": len(group),
    #     })
    #
    # stats = (
    #     rush_plays.groupby("rusher_player_id")
    #     .apply(_agg, include_groups=False)
    #     .reset_index()
    #     .rename(columns={"rusher_player_id": "gsis_id"})
    # )
    # return stats

    raise NotImplementedError("Fill in your aggregation logic")


# ── Derived stats ────────────────────────────────────────────────────────────


def compute_POSITION_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived rate stats.

    TODO: Add your position's derived stats (rates, per-game, per-snap).
    """
    # Example:
    # safe = lambda col: df[col].replace(0, np.nan)
    # df["yards_per_carry"] = df["rush_yards"] / safe("carries")
    return df


# ── Config ───────────────────────────────────────────────────────────────────

STATS_TO_ZSCORE = [
    # TODO: List all stats to z-score. Tiers help but aren't required here.
    # "rush_yards",
    # "yards_per_carry",
]

OUTPUT_COLUMNS = [
    # Always include these identity columns:
    "player_id",
    "player_display_name",
    "position",
    "recent_team",
    "season_year",
    "games",
    "off_snaps",
    # TODO: Add your raw stat columns
    # TODO: Add your z-score columns (stat_name + "_z")
]

STAT_TIERS = {
    # TODO: Map z-score column name -> tier (1=counted, 2=rate, 3=modeled, 4=inferred)
}

STAT_LABELS = {
    # TODO: Map z-score column name -> human-readable label
}

STAT_METHODOLOGY = {
    # TODO: Map z-score column name -> {"what": ..., "how": ..., "limits": ...}
}

# Uncomment and customize:
#
# POSITION_CONFIG = PositionConfig(
#     key="POSITION",                          # e.g., "qb"
#     output_filenames=["league_POSITION_all_seasons.parquet"],
#     metadata_filename="POSITION_stat_metadata.json",
#     snap_positions=["POSITION"],             # e.g., ["QB"]
#     top_n={"POSITION": 32},                  # e.g., {"QB": 32}
#     min_games=6,
#     pbp_play_types=["pass"],                 # or ["run"] or ["run", "pass"]
#     ngs_stat_type="passing",                 # or "rushing", "receiving", None
#     pfr_stat_type=None,                      # or "pass", "rush", "rec", "def"
#     aggregate_stats=[agg_POSITION],
#     ngs_col_map={},                          # NGS column -> output column
#     pfr_col_map={},                          # PFR column -> output column
#     compute_derived=compute_POSITION_derived,
#     stats_to_zscore=STATS_TO_ZSCORE,
#     invert_stats=set(),                      # Stats where lower is better
#     output_columns=OUTPUT_COLUMNS,
#     stat_tiers=STAT_TIERS,
#     stat_labels=STAT_LABELS,
#     stat_methodology=STAT_METHODOLOGY,
# )

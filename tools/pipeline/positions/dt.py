"""
DT (Defensive Tackle / Interior DL) position config.

Mirror of DE: same counting + advanced stats, just scoped to interior
defensive line via snap_positions=["DT", "NT"].
"""
from __future__ import annotations

from .de import (
    compute_de_derived as compute_dt_derived,
    STATS_TO_ZSCORE,
    OUTPUT_COLUMNS,
    STAT_TIERS,
    STAT_LABELS,
    STAT_METHODOLOGY,
)
from ..base import PositionConfig


DT_CONFIG = PositionConfig(
    key="dt",
    output_filenames=["league_dt_all_seasons.parquet"],
    metadata_filename="dt_stat_metadata.json",
    snap_positions=["DT", "NT"],
    snap_floor={"DT": 100, "NT": 100},
    min_games=6,
    snap_column="defense_snaps",
    pbp_play_types=["pass", "run"],
    pbp_season_types=["REG"],
    use_player_stats=True,
    player_stats_col_map={
        "def_sacks": "def_sacks",
        "def_sack_yards": "def_sack_yards",
        "def_qb_hits": "def_qb_hits",
        "def_tackles_solo": "def_tackles_solo",
        "def_tackle_assists": "def_tackle_assists",
        "def_tackles_with_assist": "def_tackles_with_assist",
        "def_tackles_for_loss": "def_tackles_for_loss",
        "def_tackles_for_loss_yards": "def_tackles_for_loss_yards",
        "def_pass_defended": "def_pass_defended",
        "def_interceptions": "def_interceptions",
        "def_interception_yards": "def_interception_yards",
        "def_fumbles_forced": "def_fumbles_forced",
        "def_tds": "def_tds",
        "first_week": "first_week",
        "last_week": "last_week",
    },
    compute_pass_exposure=True,
    ngs_stat_type=None,
    pfr_stat_type="def",
    aggregate_stats=[],
    ngs_col_map={},
    pfr_col_map={
        "prss": "pfr_pressures",
        "hrry": "pfr_hurries",
        "qbkd": "pfr_qb_knockdowns",
        "m_tkl_percent": "pfr_missed_tackle_pct",
    },
    compute_derived=compute_dt_derived,
    stats_to_zscore=STATS_TO_ZSCORE,
    invert_stats={"missed_tackle_pct"},  # lower is better
    zscore_groups=None,
    output_columns=OUTPUT_COLUMNS,
    stat_tiers=STAT_TIERS,
    stat_labels=STAT_LABELS,
    stat_methodology=STAT_METHODOLOGY,
)

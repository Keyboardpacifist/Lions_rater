"""
LB (Linebacker) position config.

Mix of run-stopping, blitzing, and pass coverage. Counting stats from
nflverse player_stats; pressures + missed tackles from PFR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import PositionConfig


def compute_lb_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stint LB rate stats."""
    safe = lambda col: df[col].replace(0, np.nan) if col in df.columns else np.nan

    df["sacks_per_game"] = df.get("def_sacks", 0) / safe("games_played")
    df["qb_hits_per_game"] = df.get("def_qb_hits", 0) / safe("games_played")
    df["tfl_per_game"] = df.get("def_tackles_for_loss", 0) / safe("games_played")
    df["forced_fumbles_per_game"] = df.get("def_fumbles_forced", 0) / safe("games_played")
    df["passes_defended_per_game"] = df.get("def_pass_defended", 0) / safe("games_played")
    df["interceptions_per_game"] = df.get("def_interceptions", 0) / safe("games_played")

    total_tackles = (
        df.get("def_tackles_solo", pd.Series(0)).fillna(0)
        + df.get("def_tackle_assists", pd.Series(0)).fillna(0)
    )
    df["tackles"] = total_tackles
    df["tackles_per_game"] = total_tackles / safe("games_played")
    df["tackles_per_snap"] = total_tackles / safe("off_snaps")
    df["solo_tackle_rate"] = df.get("def_tackles_solo", 0) / (
        df.get("def_tackles_solo", pd.Series(0)).fillna(0)
        + df.get("def_tackle_assists", pd.Series(0)).fillna(0)
    ).replace(0, np.nan)

    # Convenience aliases for the leaderboard
    df["sacks"] = df.get("def_sacks", np.nan)
    df["tfl"] = df.get("def_tackles_for_loss", np.nan)
    df["interceptions"] = df.get("def_interceptions", np.nan)
    df["passes_defended"] = df.get("def_pass_defended", np.nan)
    df["missed_tackle_pct"] = df.get("pfr_missed_tackle_pct", np.nan)

    return df


STATS_TO_ZSCORE = [
    "tackles_per_game",
    "solo_tackle_rate",
    "tackles_per_snap",
    "tfl_per_game",
    "sacks_per_game",
    "qb_hits_per_game",
    "forced_fumbles_per_game",
    "passes_defended_per_game",
    "interceptions_per_game",
]

INVERT_STATS = set()  # missed_tackle_pct would be inverted if z-scored, but
                      # it's surfaced as a raw column on the leaderboard.

OUTPUT_COLUMNS = [
    "player_id",
    "player_display_name",
    "player_name",  # legacy alias
    "position",
    "recent_team",
    "season_year",
    "games",
    "off_snaps",
    "def_snaps",  # legacy alias
    "first_week",
    "last_week",
    "def_sacks",
    "def_qb_hits",
    "def_tackles_for_loss",
    "def_tackles_solo",
    "def_tackle_assists",
    "def_pass_defended",
    "def_interceptions",
    "def_fumbles_forced",
    "tackles",
    "sacks",
    "tfl",
    "interceptions",
    "passes_defended",
    "missed_tackle_pct",
    "tackles_per_game",
    "tackles_per_snap",
    "solo_tackle_rate",
    "tfl_per_game",
    "sacks_per_game",
    "qb_hits_per_game",
    "forced_fumbles_per_game",
    "passes_defended_per_game",
    "interceptions_per_game",
    "pfr_pressures",
    "pfr_missed_tackle_pct",
    "tackles_per_game_z",
    "solo_tackle_rate_z",
    "tackles_per_snap_z",
    "tfl_per_game_z",
    "sacks_per_game_z",
    "qb_hits_per_game_z",
    "forced_fumbles_per_game_z",
    "passes_defended_per_game_z",
    "interceptions_per_game_z",
]

STAT_TIERS = {
    "tackles_per_game_z": 1,
    "solo_tackle_rate_z": 2,
    "tackles_per_snap_z": 2,
    "tfl_per_game_z": 1,
    "sacks_per_game_z": 1,
    "qb_hits_per_game_z": 2,
    "forced_fumbles_per_game_z": 2,
    "passes_defended_per_game_z": 2,
    "interceptions_per_game_z": 2,
}

STAT_LABELS = {
    "tackles_per_game_z": "Tackles per game",
    "solo_tackle_rate_z": "Solo tackle rate",
    "tackles_per_snap_z": "Tackles per snap",
    "tfl_per_game_z": "Tackles for loss per game",
    "sacks_per_game_z": "Sacks per game",
    "qb_hits_per_game_z": "QB hits per game",
    "forced_fumbles_per_game_z": "Forced fumbles per game",
    "passes_defended_per_game_z": "Passes defended per game",
    "interceptions_per_game_z": "Interceptions per game",
}

STAT_METHODOLOGY = {
    "tackles_per_game_z": {"what": "Total tackles per game.", "how": "(solo + assists) / games.", "limits": "Volume opportunity, not skill."},
    "solo_tackle_rate_z": {"what": "Solo tackles as a share of total tackles.", "how": "solo / (solo + assists).", "limits": "Scorekeeper subjectivity."},
    "tackles_per_snap_z": {"what": "Total tackles per defensive snap.", "how": "(solo + assists) / off_snaps.", "limits": "Heavy run-down LBs accumulate more."},
    "tfl_per_game_z": {"what": "Tackles for loss per game.", "how": "From player_stats per stint.", "limits": "Run-down opportunity."},
    "sacks_per_game_z": {"what": "Sacks per game.", "how": "From player_stats per stint.", "limits": "Blitz LBs accumulate more."},
    "qb_hits_per_game_z": {"what": "QB hits per game.", "how": "From player_stats per stint.", "limits": "Tracks contact, not effectiveness."},
    "forced_fumbles_per_game_z": {"what": "Forced fumbles per game.", "how": "From player_stats per stint.", "limits": "Rare event — small samples noisy."},
    "passes_defended_per_game_z": {"what": "Passes broken up per game.", "how": "From player_stats per stint.", "limits": "Coverage LBs accumulate more."},
    "interceptions_per_game_z": {"what": "Interceptions per game.", "how": "From player_stats per stint.", "limits": "Rare event."},
}


LB_CONFIG = PositionConfig(
    key="lb",
    output_filenames=["league_lb_all_seasons.parquet"],
    metadata_filename="lb_stat_metadata.json",
    snap_positions=["LB", "ILB", "OLB", "MLB"],
    snap_floor={"LB": 100, "ILB": 100, "OLB": 100, "MLB": 100},
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
    compute_pass_exposure=False,  # LB pressure rate isn't surfaced on leaderboard
    ngs_stat_type=None,
    pfr_stat_type="def",
    aggregate_stats=[],
    ngs_col_map={},
    pfr_col_map={
        "prss": "pfr_pressures",
        "m_tkl_percent": "pfr_missed_tackle_pct",
    },
    compute_derived=compute_lb_derived,
    stats_to_zscore=STATS_TO_ZSCORE,
    invert_stats=INVERT_STATS,
    zscore_groups=None,
    output_columns=OUTPUT_COLUMNS,
    stat_tiers=STAT_TIERS,
    stat_labels=STAT_LABELS,
    stat_methodology=STAT_METHODOLOGY,
)

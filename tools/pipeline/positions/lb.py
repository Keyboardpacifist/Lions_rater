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

    # PFR pass-rush stats — pressures, hurries, knockdowns
    df["pressures_per_game"] = df.get("pfr_pressures", 0) / safe("games_played")
    df["hurries_per_game"] = df.get("pfr_hurries", 0) / safe("games_played")
    df["qb_knockdowns_per_game"] = df.get("pfr_qb_knockdowns", 0) / safe("games_played")

    # PFR coverage stats — when LBs drop into coverage, the same PFR
    # advanced-stats record carries targets / completions / yards /
    # passer rating allowed. A coverage LB ends up tagged on these.
    df["coverage_targets_per_game"] = df.get("pfr_coverage_targets", 0) / safe("games_played")
    df["completion_pct_allowed"] = df.get("pfr_completion_pct_allowed", np.nan)
    df["yards_per_target_allowed"] = df.get("pfr_yards_per_target_allowed", np.nan)
    df["passer_rating_allowed"] = df.get("pfr_passer_rating_allowed", np.nan)

    # Missed tackle %
    df["missed_tackle_pct"] = df.get("pfr_missed_tackle_pct", np.nan)

    # Convenience aliases for the leaderboard
    df["sacks"] = df.get("def_sacks", np.nan)
    df["tfl"] = df.get("def_tackles_for_loss", np.nan)
    df["interceptions"] = df.get("def_interceptions", np.nan)
    df["passes_defended"] = df.get("def_pass_defended", np.nan)
    df["pressures"] = df.get("pfr_pressures", np.nan)
    df["hurries"] = df.get("pfr_hurries", np.nan)
    df["qb_knockdowns"] = df.get("pfr_qb_knockdowns", np.nan)

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
    # Pass rush
    "pressures_per_game",
    "hurries_per_game",
    "qb_knockdowns_per_game",
    "missed_tackle_pct",
    # Coverage (LBs in pass coverage)
    "coverage_targets_per_game",
    "completion_pct_allowed",
    "yards_per_target_allowed",
    "passer_rating_allowed",
]

# Lower-is-better stats — flip the z-score so positive = good.
INVERT_STATS = {
    "missed_tackle_pct",
    "completion_pct_allowed",
    "yards_per_target_allowed",
    "passer_rating_allowed",
}

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
    "pressures",
    "hurries",
    "qb_knockdowns",
    "tackles_per_game",
    "tackles_per_snap",
    "solo_tackle_rate",
    "tfl_per_game",
    "sacks_per_game",
    "qb_hits_per_game",
    "forced_fumbles_per_game",
    "passes_defended_per_game",
    "interceptions_per_game",
    # New pass-rush + coverage stats
    "pressures_per_game",
    "hurries_per_game",
    "qb_knockdowns_per_game",
    "missed_tackle_pct",
    "coverage_targets_per_game",
    "completion_pct_allowed",
    "yards_per_target_allowed",
    "passer_rating_allowed",
    # PFR raw passthroughs
    "pfr_pressures",
    "pfr_hurries",
    "pfr_qb_knockdowns",
    "pfr_missed_tackle_pct",
    "pfr_coverage_targets",
    "pfr_completion_pct_allowed",
    "pfr_yards_per_target_allowed",
    "pfr_passer_rating_allowed",
    # Z-scores
    "tackles_per_game_z",
    "solo_tackle_rate_z",
    "tackles_per_snap_z",
    "tfl_per_game_z",
    "sacks_per_game_z",
    "qb_hits_per_game_z",
    "forced_fumbles_per_game_z",
    "passes_defended_per_game_z",
    "interceptions_per_game_z",
    "pressures_per_game_z",
    "hurries_per_game_z",
    "qb_knockdowns_per_game_z",
    "missed_tackle_pct_z",
    "coverage_targets_per_game_z",
    "completion_pct_allowed_z",
    "yards_per_target_allowed_z",
    "passer_rating_allowed_z",
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
    # Pass rush
    "pressures_per_game_z": 1,
    "hurries_per_game_z": 2,
    "qb_knockdowns_per_game_z": 2,
    "missed_tackle_pct_z": 2,
    # Coverage (LBs in pass coverage — PFR's coverage stats)
    "coverage_targets_per_game_z": 2,
    "completion_pct_allowed_z": 2,
    "yards_per_target_allowed_z": 2,
    "passer_rating_allowed_z": 2,
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
    "pressures_per_game_z": "Pressures per game",
    "hurries_per_game_z": "Hurries per game",
    "qb_knockdowns_per_game_z": "QB knockdowns per game",
    "missed_tackle_pct_z": "Missed tackle % (lower is better)",
    "coverage_targets_per_game_z": "Coverage targets per game",
    "completion_pct_allowed_z": "Completion % allowed (lower is better)",
    "yards_per_target_allowed_z": "Yards per target allowed (lower is better)",
    "passer_rating_allowed_z": "Passer rating allowed (lower is better)",
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
    "pressures_per_game_z": {"what": "Total pressures per game (sacks + hits + hurries).", "how": "PFR pressures / games.", "limits": "PFR's chart, not PFF's."},
    "hurries_per_game_z": {"what": "Hurries per game (pressure short of sack/knockdown).", "how": "PFR hurries / games.", "limits": "Subset of pressures."},
    "qb_knockdowns_per_game_z": {"what": "QB knockdowns per game.", "how": "PFR qb_knockdowns / games.", "limits": "Different from QB hits in nflverse player_stats."},
    "missed_tackle_pct_z": {"what": "% of tackle attempts that were missed. Lower is better.", "how": "PFR missed_tackle_pct, z-score inverted so positive = reliable tackler.", "limits": "PFR charting subjectivity."},
    "coverage_targets_per_game_z": {"what": "Times targeted in coverage per game.", "how": "PFR def_targets / games. Higher = more action when in coverage.", "limits": "Volume — could mean trusted matchup OR weak link picked on."},
    "completion_pct_allowed_z": {"what": "% of targets completed against this LB. Lower is better.", "how": "PFR cmp_percent allowed, z-score inverted.", "limits": "Heavily tied to who the LB is matched on."},
    "yards_per_target_allowed_z": {"what": "Yards allowed per target. Lower is better.", "how": "PFR yds_tgt allowed, z-score inverted.", "limits": "RB/TE matchups skew this for some LBs."},
    "passer_rating_allowed_z": {"what": "QB rating when targeting this LB. Lower is better.", "how": "PFR passer rating allowed, z-score inverted.", "limits": "The classic CB/LB coverage stat."},
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
        # Pass rush
        "prss": "pfr_pressures",
        "hrry": "pfr_hurries",
        "qbkd": "pfr_qb_knockdowns",
        "m_tkl_percent": "pfr_missed_tackle_pct",
        "m_tkl": "pfr_missed_tackles",
        # Coverage (LBs in coverage)
        "tgt": "pfr_coverage_targets",
        "cmp": "pfr_coverage_completions",
        "cmp_percent": "pfr_completion_pct_allowed",
        "yds_tgt": "pfr_yards_per_target_allowed",
        "rat": "pfr_passer_rating_allowed",
        "dadot": "pfr_avg_depth_of_target",
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

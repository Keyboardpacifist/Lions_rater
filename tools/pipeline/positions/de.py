"""
DE (Defensive End / Edge) position config.

Population: DEs with >= 100 total defensive snaps for the season.
Counting stats from nflverse player_stats. Pressure data from PFR.
Pressure rate uses an estimated "pass plays defended while on the field"
denominator (team_total_pass_plays × player_def_snap_share).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import PositionConfig


def compute_de_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stint defensive rate stats."""
    safe = lambda col: df[col].replace(0, np.nan) if col in df.columns else np.nan

    # Per-game volume rates
    df["sacks_per_game"] = df.get("def_sacks", 0) / safe("games_played")
    df["qb_hits_per_game"] = df.get("def_qb_hits", 0) / safe("games_played")
    df["tfl_per_game"] = df.get("def_tackles_for_loss", 0) / safe("games_played")
    df["forced_fumbles_per_game"] = df.get("def_fumbles_forced", 0) / safe("games_played")
    df["passes_defended_per_game"] = df.get("def_pass_defended", 0) / safe("games_played")
    df["interceptions_per_game"] = df.get("def_interceptions", 0) / safe("games_played")

    # Tackles per snap
    total_tackles = (
        df.get("def_tackles_solo", pd.Series(0)).fillna(0)
        + df.get("def_tackle_assists", pd.Series(0)).fillna(0)
    )
    df["tackles_per_snap"] = total_tackles / safe("off_snaps")
    # Solo tackle rate = solo / (solo + assists)
    df["solo_tackle_rate"] = df.get("def_tackles_solo", 0) / (
        df.get("def_tackles_solo", pd.Series(0)).fillna(0)
        + df.get("def_tackle_assists", pd.Series(0)).fillna(0)
    ).replace(0, np.nan)

    # Pressure rate = PFR pressures / estimated pass plays defended on field.
    if "pfr_pressures" in df.columns and "pass_plays_exposure" in df.columns:
        df["pressure_rate"] = df["pfr_pressures"] / df["pass_plays_exposure"].replace(0, np.nan)
    else:
        df["pressure_rate"] = np.nan

    # Per-game versions of the previously-hidden PFR fields. These are
    # the headline pass-rush quality stats — pressures (incl. hurries +
    # knockdowns + sacks) and hurries are more honest than sacks alone.
    df["pressures_per_game"] = df.get("pfr_pressures", 0) / safe("games_played")
    df["hurries_per_game"] = df.get("pfr_hurries", 0) / safe("games_played")
    df["qb_knockdowns_per_game"] = df.get("pfr_qb_knockdowns", 0) / safe("games_played")
    # Missed tackle % is already a percentage 0-100 in PFR data — keep as-is.
    df["missed_tackle_pct"] = df.get("pfr_missed_tackle_pct", np.nan)

    # Convenience raw columns for the leaderboard
    df["pressures"] = df.get("pfr_pressures", np.nan)
    df["sacks"] = df.get("def_sacks", np.nan)
    df["qb_hits"] = df.get("def_qb_hits", np.nan)
    df["tfl"] = df.get("def_tackles_for_loss", np.nan)
    df["tackles"] = total_tackles
    df["hurries"] = df.get("pfr_hurries", np.nan)
    df["qb_knockdowns"] = df.get("pfr_qb_knockdowns", np.nan)

    return df


STATS_TO_ZSCORE = [
    "sacks_per_game",
    "qb_hits_per_game",
    "pressure_rate",
    "pressures_per_game",
    "hurries_per_game",
    "qb_knockdowns_per_game",
    "missed_tackle_pct",
    "tfl_per_game",
    "solo_tackle_rate",
    "tackles_per_snap",
    "forced_fumbles_per_game",
    "passes_defended_per_game",
    "interceptions_per_game",
]

OUTPUT_COLUMNS = [
    "player_id",
    "player_display_name",
    "player_name",  # legacy alias for older pages
    "position",
    "recent_team",
    "season_year",
    "games",
    "off_snaps",
    "def_snaps",  # legacy alias
    "first_week",
    "last_week",
    # Counting (from player_stats)
    "def_sacks",
    "def_qb_hits",
    "def_tackles_for_loss",
    "def_tackles_solo",
    "def_tackle_assists",
    "def_pass_defended",
    "def_interceptions",
    "def_fumbles_forced",
    # Convenience aliases for the page
    "sacks",
    "qb_hits",
    "tfl",
    "tackles",
    "pressures",
    "hurries",
    "qb_knockdowns",
    # Derived per-game / per-snap rates
    "sacks_per_game",
    "qb_hits_per_game",
    "tfl_per_game",
    "tackles_per_snap",
    "solo_tackle_rate",
    "forced_fumbles_per_game",
    "passes_defended_per_game",
    "interceptions_per_game",
    # Pressure rate (the headline modern stat)
    "pressure_rate",
    "pass_plays_exposure",
    "team_pass_plays_defended",
    # New PFR-derived stats (Phase 2.5 defensive depth)
    "pressures_per_game",
    "hurries_per_game",
    "qb_knockdowns_per_game",
    "missed_tackle_pct",
    # PFR raw fields exposed for the leaderboard / detail
    "pfr_pressures",
    "pfr_hurries",
    "pfr_qb_knockdowns",
    "pfr_missed_tackle_pct",
    # Z-scores (match what pages/DE.py expects via RAW_COL_MAP)
    "sacks_per_game_z",
    "qb_hits_per_game_z",
    "pressure_rate_z",
    "pressures_per_game_z",
    "hurries_per_game_z",
    "qb_knockdowns_per_game_z",
    "missed_tackle_pct_z",
    "tfl_per_game_z",
    "solo_tackle_rate_z",
    "tackles_per_snap_z",
    "forced_fumbles_per_game_z",
    "passes_defended_per_game_z",
    "interceptions_per_game_z",
]

STAT_TIERS = {
    "sacks_per_game_z": 1,
    "qb_hits_per_game_z": 2,
    "pressure_rate_z": 3,
    "pressures_per_game_z": 1,
    "hurries_per_game_z": 2,
    "qb_knockdowns_per_game_z": 2,
    "missed_tackle_pct_z": 2,
    "tfl_per_game_z": 1,
    "solo_tackle_rate_z": 2,
    "tackles_per_snap_z": 2,
    "forced_fumbles_per_game_z": 2,
    "passes_defended_per_game_z": 2,
    "interceptions_per_game_z": 2,
}

STAT_LABELS = {
    "sacks_per_game_z": "Sacks per game",
    "qb_hits_per_game_z": "QB hits per game",
    "pressure_rate_z": "Pressure rate",
    "pressures_per_game_z": "Pressures per game",
    "hurries_per_game_z": "Hurries per game",
    "qb_knockdowns_per_game_z": "QB knockdowns per game",
    "missed_tackle_pct_z": "Missed tackle % (lower is better)",
    "tfl_per_game_z": "Tackles for loss per game",
    "solo_tackle_rate_z": "Solo tackle rate",
    "tackles_per_snap_z": "Tackles per snap",
    "forced_fumbles_per_game_z": "Forced fumbles per game",
    "passes_defended_per_game_z": "Passes defended per game",
    "interceptions_per_game_z": "Interceptions per game",
}

STAT_METHODOLOGY = {
    "sacks_per_game_z": {
        "what": "Sacks per game.",
        "how": "From player_stats. Per-stint sacks / per-stint games.",
        "limits": "Volume stat. Boom-or-bust play type.",
    },
    "qb_hits_per_game_z": {
        "what": "QB hits per game (sack + non-sack pressure that contacted the QB).",
        "how": "From player_stats. Per-stint count / games.",
        "limits": "Tracks contact, not effectiveness.",
    },
    "pressure_rate_z": {
        "what": "Pressures per pass play defended while on the field.",
        "how": "PFR pressures / estimated pass plays defended (team pass plays × this player's defensive snap share).",
        "limits": "Snap-share approximation; not a true PFF/NGS pass-rush snap denominator.",
    },
    "tfl_per_game_z": {
        "what": "Tackles for loss per game.",
        "how": "From player_stats. Per-stint TFLs / games.",
        "limits": "Run-down opportunity dependent.",
    },
    "solo_tackle_rate_z": {
        "what": "Solo tackles as a share of total tackles.",
        "how": "solo / (solo + assists).",
        "limits": "Scorekeeper subjectivity on solo vs assist.",
    },
    "tackles_per_snap_z": {
        "what": "Total tackles per defensive snap.",
        "how": "(solo + assists) / off_snaps.",
        "limits": "Volume opportunity, not skill.",
    },
    "forced_fumbles_per_game_z": {
        "what": "Forced fumbles per game.",
        "how": "From player_stats. Per-stint count / games.",
        "limits": "Rare event — small samples noisy.",
    },
    "passes_defended_per_game_z": {
        "what": "Passes broken up per game (DE deflections at the line + pass coverage on swings).",
        "how": "From player_stats. Per-stint count / games.",
        "limits": "Edge defenders see fewer of these than DBs.",
    },
    "interceptions_per_game_z": {
        "what": "Interceptions per game.",
        "how": "From player_stats. Per-stint count / games.",
        "limits": "Even rarer than forced fumbles for edge players.",
    },
    "pressures_per_game_z": {
        "what": "Total pressures per game (sacks + hits + hurries).",
        "how": "PFR pressures / games_played.",
        "limits": "PFR's chart, not PFF's. Methods differ slightly between sources.",
    },
    "hurries_per_game_z": {
        "what": "Hurries per game (pressure that didn't sack or knock down).",
        "how": "PFR hurries / games_played.",
        "limits": "Subset of total pressures. Sometimes overlaps with hits.",
    },
    "qb_knockdowns_per_game_z": {
        "what": "QB knockdowns per game (got the QB to the ground without a sack).",
        "how": "PFR qb_knockdowns / games_played.",
        "limits": "Different from QB hits in nflverse player_stats.",
    },
    "missed_tackle_pct_z": {
        "what": "% of tackle attempts that were missed. Lower is better — z-score is INVERTED so higher z = more reliable tackler.",
        "how": "PFR missed_tackle_pct, then z-scored with sign flipped so positive = good.",
        "limits": "PFR's charting subjectivity; not always the same as PFF's missed tackle count.",
    },
}


DE_CONFIG = PositionConfig(
    key="de",
    output_filenames=["league_de_all_seasons.parquet"],
    metadata_filename="de_stat_metadata.json",
    snap_positions=["DE", "DL"],
    snap_floor={"DE": 100, "DL": 100},
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
    compute_derived=compute_de_derived,
    stats_to_zscore=STATS_TO_ZSCORE,
    invert_stats={"missed_tackle_pct"},  # lower is better
    zscore_groups=None,
    output_columns=OUTPUT_COLUMNS,
    stat_tiers=STAT_TIERS,
    stat_labels=STAT_LABELS,
    stat_methodology=STAT_METHODOLOGY,
)

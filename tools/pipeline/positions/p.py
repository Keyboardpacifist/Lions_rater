"""
P (Punter) position config.

Population: punters with >= 30 ST snaps in a season (basically anyone
with enough work to evaluate). One row per (player, team) stint.

Counting + advanced stats derived from PBP punt plays. nflverse
player_stats doesn't expose punter-specific columns; everything we need
is in PBP (kick_distance, return_yards, touchback, punt_inside_twenty).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import PositionConfig


def agg_punter(pbp: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Per-(player, team) punter stats from PBP punt plays.

    Net yards per punt = gross - return - (touchback adjustment).
    nflverse PBP marks `touchback==1` for punts that go for a touchback.
    """
    punts = pbp[
        (pbp["play_type"] == "punt") & (pbp["punter_player_id"].notna())
    ].copy()
    if punts.empty:
        return pd.DataFrame()

    # Net yards convention: gross - returnYards - 20 if touchback (since
    # touchback gives ball at the 20). Punt blocked = 0 yards typically.
    punts["return_yards"] = punts.get("return_yards", pd.Series(0)).fillna(0)
    punts["touchback"] = punts.get("touchback", pd.Series(0)).fillna(0).astype(int)
    punts["punt_inside_twenty"] = punts.get("punt_inside_twenty", pd.Series(0)).fillna(0).astype(int)
    punts["punt_blocked"] = punts.get("punt_blocked", pd.Series(0)).fillna(0).astype(int)
    punts["punt_fair_catch"] = punts.get("punt_fair_catch", pd.Series(0)).fillna(0).astype(int)
    punts["punt_downed"] = punts.get("punt_downed", pd.Series(0)).fillna(0).astype(int)
    punts["punt_out_of_bounds"] = punts.get("punt_out_of_bounds", pd.Series(0)).fillna(0).astype(int)
    punts["kick_distance"] = punts.get("kick_distance", pd.Series(0)).fillna(0)
    punts["net_yards"] = punts["kick_distance"] - punts["return_yards"] - 20 * punts["touchback"]

    def _agg(group):
        n = len(group)
        return pd.Series({
            "punts": n,
            "punt_yards": group["kick_distance"].sum(),
            "punt_returns": int((group["return_yards"] > 0).sum()),
            "punt_return_yards_allowed": group["return_yards"].sum(),
            "touchbacks": int(group["touchback"].sum()),
            "inside_twenty": int(group["punt_inside_twenty"].sum()),
            "fair_catches": int(group["punt_fair_catch"].sum()),
            "downed": int(group["punt_downed"].sum()),
            "out_of_bounds": int(group["punt_out_of_bounds"].sum()),
            "blocked": int(group["punt_blocked"].sum()),
            "net_punt_yards": group["net_yards"].sum(),
            "punt_epa_total": group["epa"].sum() if "epa" in group.columns else np.nan,
        })

    stats = (
        punts.groupby(["punter_player_id", "posteam"])
        .apply(_agg, include_groups=False)
        .reset_index()
        .rename(columns={"punter_player_id": "gsis_id", "posteam": "team"})
    )
    return stats


def compute_p_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stint punter rate stats."""
    safe = lambda col: df[col].replace(0, np.nan) if col in df.columns else np.nan
    df["avg_distance"] = df.get("punt_yards", 0) / safe("punts")
    df["avg_net"] = df.get("net_punt_yards", 0) / safe("punts")
    df["inside_20_rate"] = df.get("inside_twenty", 0) / safe("punts")
    df["touchback_rate"] = df.get("touchbacks", 0) / safe("punts")
    df["fair_catch_rate"] = df.get("fair_catches", 0) / safe("punts")
    # Pin rate = total of inside-20 + downed + out-of-bounds (deep / placement punts)
    pin_count = (
        df.get("inside_twenty", pd.Series(0)).fillna(0)
        + df.get("downed", pd.Series(0)).fillna(0)
        + df.get("out_of_bounds", pd.Series(0)).fillna(0)
    )
    df["pin_rate"] = pin_count / safe("punts")
    df["punt_epa"] = df.get("punt_epa_total", 0) / safe("punts")
    # Legacy column the page uses
    df["punt_att"] = df.get("punts", np.nan)
    return df


STATS_TO_ZSCORE = [
    "avg_distance",
    "avg_net",
    "inside_20_rate",
    "touchback_rate",
    "fair_catch_rate",
    "pin_rate",
    "punt_epa",
]

INVERT_STATS = {"touchback_rate"}  # lower is better

OUTPUT_COLUMNS = [
    "player_id",
    "player_display_name",
    "player_name",  # legacy alias
    "position",
    "recent_team",
    "season_year",
    "games",
    "off_snaps",
    "st_snaps",  # legacy alias
    "first_week",
    "last_week",
    "punts",
    "punt_att",  # legacy alias
    "punt_yards",
    "net_punt_yards",
    "touchbacks",
    "inside_twenty",
    "fair_catches",
    "downed",
    "out_of_bounds",
    "blocked",
    "punt_returns",
    "punt_return_yards_allowed",
    "punt_epa_total",
    "avg_distance",
    "avg_net",
    "inside_20_rate",
    "touchback_rate",
    "fair_catch_rate",
    "pin_rate",
    "punt_epa",
    "avg_distance_z",
    "avg_net_z",
    "inside_20_rate_z",
    "touchback_rate_z",
    "fair_catch_rate_z",
    "pin_rate_z",
    "punt_epa_z",
]

STAT_TIERS = {
    "avg_distance_z": 1,
    "avg_net_z": 2,
    "inside_20_rate_z": 2,
    "touchback_rate_z": 2,
    "punt_epa_z": 3,
}

STAT_LABELS = {
    "avg_distance_z": "Average gross distance",
    "avg_net_z": "Average net yards",
    "inside_20_rate_z": "Inside-20 rate",
    "touchback_rate_z": "Touchback rate (lower is better)",
    "punt_epa_z": "EPA per punt",
}

STAT_METHODOLOGY = {
    "avg_distance_z": {
        "what": "Average gross yards per punt.",
        "how": "kick_distance summed across the player's punts, divided by punt count.",
        "limits": "Doesn't account for return or touchback.",
    },
    "avg_net_z": {
        "what": "Average net yards per punt — the modern punter stat.",
        "how": "(gross yards − return yards − 20 × touchbacks) / punts. Touchback penalty is the standard 20-yard correction.",
        "limits": "Coverage team is part of the credit.",
    },
    "inside_20_rate_z": {
        "what": "Share of punts downed inside the opponent's 20-yard line.",
        "how": "PBP punt_inside_twenty / total punts.",
        "limits": "Field position dependent.",
    },
    "touchback_rate_z": {
        "what": "Share of punts that resulted in a touchback. Lower is better.",
        "how": "PBP touchback / total punts. Z-score is inverted so positive = good.",
        "limits": "Aggressive distance can drag this up; tradeoff with avg distance.",
    },
    "punt_epa_z": {
        "what": "Expected Points Added per punt.",
        "how": "Mean of nflverse EPA across the player's punts.",
        "limits": "Punt EPA reflects field position swing — partly the punt-coverage unit's work.",
    },
}


P_CONFIG = PositionConfig(
    key="p",
    output_filenames=["league_p_all_seasons.parquet"],
    metadata_filename="punter_stat_metadata.json",
    snap_positions=["P"],
    snap_floor={"P": 30},  # ~half a season of ST snaps
    min_games=4,
    snap_column="st_snaps",
    pbp_play_types=["punt"],
    pbp_season_types=["REG"],
    use_player_stats=False,  # Punter stats not in player_stats; PBP only.
    player_stats_col_map={},
    compute_pass_exposure=False,
    ngs_stat_type=None,
    pfr_stat_type=None,
    aggregate_stats=[agg_punter],
    ngs_col_map={},
    pfr_col_map={},
    compute_derived=compute_p_derived,
    stats_to_zscore=STATS_TO_ZSCORE,
    invert_stats=INVERT_STATS,
    zscore_groups=None,
    output_columns=OUTPUT_COLUMNS,
    stat_tiers=STAT_TIERS,
    stat_labels=STAT_LABELS,
    stat_methodology=STAT_METHODOLOGY,
)

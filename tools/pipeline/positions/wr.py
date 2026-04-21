"""
WR + TE position config.

Translated from tools/wr_data_pull.py. WR and TE are z-scored together
as one combined "pass catcher" pool (top 64 WR + top 32 TE by snaps).
Both league_wr_all_seasons.parquet and league_te_all_seasons.parquet
are written from the same population — each page filters on position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import PositionConfig


# ── PBP aggregation ──────────────────────────────────────────────────────────


def agg_receiver(pbp: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player receiving stats from pass plays.

    Matches wr_data_pull.py:123-154.
    """
    # Filter to pass plays with a receiver
    passes = pbp[
        (pbp["play_type"] == "pass") & (pbp["receiver_player_id"].notna())
    ].copy()

    if passes.empty:
        return pd.DataFrame()

    def _agg(group):
        targets = len(group)
        receptions = group["complete_pass"].sum()
        rec_yards = group["receiving_yards"].fillna(0).sum()
        rec_tds = (
            group["pass_touchdown"].fillna(0).sum()
            if "pass_touchdown" in group.columns
            else group["touchdown"].fillna(0).sum()
        )
        rec_first_downs = (
            group["first_down"].fillna(0).sum()
            if "first_down" in group.columns
            else np.nan
        )
        air_yards = group["air_yards"].fillna(0).sum()
        epa_per_target = group["epa"].mean()
        success_rate = group["success"].mean()
        yac_sum = group["yards_after_catch"].fillna(0).sum()
        avg_cpoe = (
            group["cpoe"].mean() if "cpoe" in group.columns else np.nan
        )

        return pd.Series(
            {
                "targets": targets,
                "receptions": receptions,
                "rec_yards": rec_yards,
                "rec_tds": rec_tds,
                "rec_first_downs": rec_first_downs,
                "air_yards": air_yards,
                "epa_per_target": epa_per_target,
                "success_rate": success_rate,
                "yac": yac_sum,
                "avg_cpoe": avg_cpoe,
            }
        )

    stats = (
        passes.groupby("receiver_player_id")
        .apply(_agg, include_groups=False)
        .reset_index()
        .rename(columns={"receiver_player_id": "gsis_id"})
    )
    return stats


# ── Derived stats ────────────────────────────────────────────────────────────


def compute_wr_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived rate stats for WR/TE.

    Matches wr_data_pull.py:186-193.
    """
    safe = lambda col: df[col].replace(0, np.nan) if col in df.columns else np.nan

    df["yards_per_target"] = df.get("rec_yards", 0) / safe("targets")
    df["yards_per_snap"] = df.get("rec_yards", 0) / safe("off_snaps")
    df["catch_rate"] = df.get("receptions", 0) / safe("targets")
    df["targets_per_snap"] = df.get("targets", 0) / safe("off_snaps")
    df["first_down_rate"] = df.get("rec_first_downs", 0) / safe("targets")
    df["yac_per_reception"] = df.get("yac", 0) / safe("receptions")

    return df


# ── Stats to z-score ─────────────────────────────────────────────────────────

STATS_TO_ZSCORE = [
    # Tier 1 — raw counts
    "rec_yards",
    "receptions",
    "rec_tds",
    "targets",
    # Tier 2 — rates
    "catch_rate",
    "success_rate",
    "first_down_rate",
    "yards_per_target",
    "yac_per_reception",
    "targets_per_snap",
    "yards_per_snap",
    # Tier 3 — adjusted / modeled
    "epa_per_target",
    "avg_cpoe",
    "yac_above_exp",
    "avg_separation",
]

# ── Output columns ───────────────────────────────────────────────────────────
# These must match what pages/WR.py and pages/TE.py expect.

OUTPUT_COLUMNS = [
    # Identity
    "player_id",
    "player_display_name",
    "position",
    "recent_team",
    "season_year",
    "games",
    "off_snaps",
    # Raw counts
    "targets",
    "receptions",
    "rec_yards",
    "rec_tds",
    "rec_first_downs",
    "air_yards",
    "yac",
    # NGS (may be NaN for older seasons)
    "avg_cushion",
    "avg_separation",
    "avg_target_depth",
    # Derived rates
    "catch_rate",
    "success_rate",
    "first_down_rate",
    "yards_per_target",
    "yards_per_snap",
    "targets_per_snap",
    "yac_per_reception",
    "yac_above_exp",
    "epa_per_target",
    "avg_cpoe",
    # Z-scores
    "rec_yards_z",
    "receptions_z",
    "rec_tds_z",
    "targets_z",
    "catch_rate_z",
    "success_rate_z",
    "first_down_rate_z",
    "yards_per_target_z",
    "yards_per_snap_z",
    "targets_per_snap_z",
    "yac_per_reception_z",
    "yac_above_exp_z",
    "epa_per_target_z",
    "avg_cpoe_z",
    "avg_separation_z",
]

# ── Metadata ─────────────────────────────────────────────────────────────────

STAT_TIERS = {
    "rec_yards_z": 1,
    "receptions_z": 1,
    "rec_tds_z": 1,
    "targets_z": 1,
    "catch_rate_z": 2,
    "success_rate_z": 2,
    "first_down_rate_z": 2,
    "yards_per_target_z": 2,
    "yards_per_snap_z": 2,
    "targets_per_snap_z": 2,
    "yac_per_reception_z": 2,
    "epa_per_target_z": 3,
    "avg_cpoe_z": 3,
    "yac_above_exp_z": 3,
    "avg_separation_z": 3,
}

STAT_LABELS = {
    "rec_yards_z": "Receiving yards (raw)",
    "receptions_z": "Receptions (raw)",
    "rec_tds_z": "Receiving TDs (raw)",
    "targets_z": "Targets (raw)",
    "catch_rate_z": "Catch rate",
    "success_rate_z": "Success rate",
    "first_down_rate_z": "First-down rate",
    "yards_per_target_z": "Yards per target",
    "yards_per_snap_z": "Yards per snap",
    "targets_per_snap_z": "Targets per snap",
    "yac_per_reception_z": "YAC per reception",
    "epa_per_target_z": "EPA per target",
    "avg_cpoe_z": "CPOE",
    "yac_above_exp_z": "YAC over expected",
    "avg_separation_z": "Average separation",
}

STAT_METHODOLOGY = {
    "rec_yards_z": {
        "what": "Total raw receiving yards.",
        "how": "Sum of PBP receiving_yards, z-scored against the league population (top 64 WR + top 32 TE by snaps).",
        "limits": "Raw volume stat \u2014 rewards opportunity as much as skill.",
    },
    "receptions_z": {
        "what": "Total raw receptions.",
        "how": "Count of complete passes where this player was the receiver.",
        "limits": "Volume stat. High-volume possession receivers outrank efficient deep threats.",
    },
    "rec_tds_z": {
        "what": "Total raw receiving touchdowns.",
        "how": "Count of TDs on pass plays where this player was the receiver.",
        "limits": "Small integer samples are noisy.",
    },
    "targets_z": {
        "what": "Total raw targets.",
        "how": "Count of pass plays where this player was the intended receiver.",
        "limits": "Pure opportunity \u2014 not a skill measure.",
    },
    "catch_rate_z": {
        "what": "Percentage of targets caught.",
        "how": "receptions / targets.",
        "limits": "Doesn't account for target difficulty.",
    },
    "success_rate_z": {
        "what": "Percentage of targets that produced a successful play by EPA standards.",
        "how": "Mean of nflverse binary success flag across this player's targets.",
        "limits": "Binary cutoff hides near-misses and runaway successes.",
    },
    "first_down_rate_z": {
        "what": "Percentage of targets that gained a first down.",
        "how": "first_downs / targets.",
        "limits": "Depends on usage patterns (slot on 3rd-and-short inflates this).",
    },
    "yards_per_target_z": {
        "what": "Average yards per target (not per reception).",
        "how": "total receiving yards / total targets.",
        "limits": "Penalizes drops as zeros. Rewards big plays disproportionately.",
    },
    "yards_per_snap_z": {
        "what": "Receiving yards per offensive snap on the field.",
        "how": "total receiving yards / offensive snaps.",
        "limits": "Best available efficiency-of-role metric from free data.",
    },
    "targets_per_snap_z": {
        "what": "How often the QB looks your way per snap.",
        "how": "targets / offensive snaps.",
        "limits": "Measures role, not skill.",
    },
    "yac_per_reception_z": {
        "what": "Average yards gained after the catch, per reception.",
        "how": "total yards_after_catch / receptions.",
        "limits": "Credit shared between receiver, scheme, and blockers.",
    },
    "epa_per_target_z": {
        "what": "Expected Points Added per target.",
        "how": "Mean of nflverse EPA on this player's targets.",
        "limits": "Depends on trusting the EPA model.",
    },
    "avg_cpoe_z": {
        "what": "Completion Percentage Over Expected.",
        "how": "Actual completion - model-expected completion, averaged across targets.",
        "limits": "A model decides what 'expected' means.",
    },
    "yac_above_exp_z": {
        "what": "YAC vs. league-average receiver in the same situations.",
        "how": "NFL Next Gen Stats: actual YAC - expected YAC from tracking data.",
        "limits": "Requires NGS tracking data. Small samples may be unstable.",
    },
    "avg_separation_z": {
        "what": "Average yards of separation from nearest defender at catch point.",
        "how": "NFL Next Gen Stats tracking data, season average.",
        "limits": "Depth-blind \u2014 doesn't distinguish route-running from scheme.",
    },
}


# ── The config ───────────────────────────────────────────────────────────────

WR_CONFIG = PositionConfig(
    key="wr",
    output_filenames=["league_wr_all_seasons.parquet", "league_te_all_seasons.parquet"],
    metadata_filename="wr_te_stat_metadata.json",
    snap_positions=["WR", "TE"],
    top_n={"WR": 64, "TE": 32},
    min_games=6,
    pbp_play_types=["pass"],
    ngs_stat_type="receiving",
    pfr_stat_type=None,  # WR doesn't use PFR advanced stats
    aggregate_stats=[agg_receiver],
    ngs_col_map={
        "avg_separation": "avg_separation",
        "avg_cushion": "avg_cushion",
        "avg_intended_air_yards": "avg_target_depth",
        "avg_yac_above_expectation": "yac_above_exp",
    },
    pfr_col_map={},
    compute_derived=compute_wr_derived,
    stats_to_zscore=STATS_TO_ZSCORE,
    invert_stats=set(),  # All WR stats are higher-is-better
    output_columns=OUTPUT_COLUMNS,
    stat_tiers=STAT_TIERS,
    stat_labels=STAT_LABELS,
    stat_methodology=STAT_METHODOLOGY,
)

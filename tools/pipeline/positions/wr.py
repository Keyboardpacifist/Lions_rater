"""
WR + TE position config.

Population: WRs and TEs with >= 100 total offensive snaps for the season,
min 6 games. Each position is z-scored within its own pool (Decision 2:
separate reference, combined output).

Counting stats (targets, receptions, yards, TDs, etc.) come from
nflverse pre-aggregated player_stats — handles trades and lateral-TD
attribution correctly. PBP is used only for advanced stats that
nflverse doesn't pre-compute (success_rate, EPA per target).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import PositionConfig


# ── PBP aggregation: ADVANCED stats only ────────────────────────────────────
# Counting stats (targets, receptions, rec_yards, rec_tds, rec_first_downs,
# air_yards, yac) come from player_stats — see player_stats_col_map below.


def agg_receiver_advanced(pbp: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Compute per-(player, team) PBP-derived stats that aren't in player_stats.

    Grouping by posteam ensures traded players get separate advanced stats
    for each team stint (so Davante Adams' LV success_rate is distinct from
    his NYJ success_rate).

    Currently: success_rate, epa_per_target, avg_cpoe.
    """
    passes = pbp[
        (pbp["play_type"] == "pass") & (pbp["receiver_player_id"].notna())
    ].copy()

    if passes.empty:
        return pd.DataFrame()

    def _agg(group):
        return pd.Series(
            {
                "epa_per_target": group["epa"].mean(),
                "success_rate": group["success"].mean(),
                "avg_cpoe": (
                    group["cpoe"].mean() if "cpoe" in group.columns else np.nan
                ),
            }
        )

    stats = (
        passes.groupby(["receiver_player_id", "posteam"])
        .apply(_agg, include_groups=False)
        .reset_index()
        .rename(columns={"receiver_player_id": "gsis_id", "posteam": "team"})
    )
    return stats


# ── Derived stats (rate stats from counting stats) ──────────────────────────


def compute_wr_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived rate stats from counting stats."""
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
    # Tier 2 — opportunity / efficiency from player_stats
    "target_share",
    "air_yards_share",
    "racr",
    "wopr",
]

# ── Output columns ───────────────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    # Identity
    "player_id",
    "player_display_name",
    "position",
    "recent_team",
    "season_year",
    "games",
    "off_snaps",
    "first_week",
    "last_week",
    # Raw counts (from player_stats)
    "targets",
    "receptions",
    "rec_yards",
    "rec_tds",
    "rec_first_downs",
    "air_yards",
    "yac",
    "rec_2pt",
    "rec_fumbles",
    # NGS (may be NaN for older seasons)
    "avg_cushion",
    "avg_separation",
    "avg_target_depth",
    "yac_above_exp",
    # Opportunity (from player_stats)
    "target_share",
    "air_yards_share",
    "racr",
    "wopr",
    # Derived rates
    "catch_rate",
    "success_rate",
    "first_down_rate",
    "yards_per_target",
    "yards_per_snap",
    "targets_per_snap",
    "yac_per_reception",
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
    "target_share_z",
    "air_yards_share_z",
    "racr_z",
    "wopr_z",
]

# ── Metadata ─────────────────────────────────────────────────────────────────

STAT_TIERS = {
    "rec_yards_z": 1,
    "receptions_z": 1,
    "rec_tds_z": 1,
    "targets_z": 1,
    "target_share_z": 2,
    "air_yards_share_z": 2,
    "racr_z": 2,
    "wopr_z": 2,
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
    "target_share_z": "Target share",
    "air_yards_share_z": "Air-yards share",
    "racr_z": "RACR (yards / air yards)",
    "wopr_z": "WOPR (weighted opportunity)",
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
        "what": "Total raw receiving yards (full season, REG only).",
        "how": "From nflverse load_player_stats(summary_level='reg'). Z-scored within position group (WR vs WR, TE vs TE).",
        "limits": "Raw volume stat — rewards opportunity as much as skill.",
    },
    "receptions_z": {
        "what": "Total raw receptions.",
        "how": "From nflverse load_player_stats. Z-scored within position group.",
        "limits": "Volume stat. High-volume possession receivers outrank efficient deep threats.",
    },
    "rec_tds_z": {
        "what": "Total raw receiving touchdowns.",
        "how": "From nflverse load_player_stats — correctly attributed even on lateral plays (which PBP overcounts).",
        "limits": "Small integer samples are noisy.",
    },
    "targets_z": {
        "what": "Total raw targets.",
        "how": "From nflverse load_player_stats.",
        "limits": "Pure opportunity — not a skill measure.",
    },
    "target_share_z": {
        "what": "Player's share of team passing targets.",
        "how": "From nflverse load_player_stats. targets / team total targets.",
        "limits": "Team-context dependent. Top option on a low-volume team can outrank #2 on a heavy passing team.",
    },
    "air_yards_share_z": {
        "what": "Player's share of team intended air yards (downfield throws).",
        "how": "From nflverse load_player_stats.",
        "limits": "Rewards deep threats; possession receivers near zero by design.",
    },
    "racr_z": {
        "what": "Receiver Air Conversion Ratio. Yards gained per air yard thrown your way.",
        "how": "receiving_yards / receiving_air_yards from player_stats.",
        "limits": "Inflated by big YAC plays; small-sample players get extreme values.",
    },
    "wopr_z": {
        "what": "Weighted Opportunity Rating. Combines target share and air-yards share.",
        "how": "1.5 × target_share + 0.7 × air_yards_share. Standard fantasy/analytics opportunity metric.",
        "limits": "Opportunity, not skill. Two players with the same WOPR can have very different outcomes.",
    },
    "catch_rate_z": {
        "what": "Percentage of targets caught.",
        "how": "receptions / targets.",
        "limits": "Doesn't account for target difficulty.",
    },
    "success_rate_z": {
        "what": "Percentage of targets that produced a successful play by EPA standards.",
        "how": "Mean of nflverse binary success flag across this player's targets, from PBP.",
        "limits": "Binary cutoff hides near-misses and runaway successes.",
    },
    "first_down_rate_z": {
        "what": "Percentage of targets that gained a first down.",
        "how": "receiving_first_downs / targets, both from player_stats.",
        "limits": "Depends on usage patterns (slot on 3rd-and-short inflates this).",
    },
    "yards_per_target_z": {
        "what": "Average yards per target (not per reception).",
        "how": "receiving_yards / targets.",
        "limits": "Penalizes drops as zeros. Rewards big plays disproportionately.",
    },
    "yards_per_snap_z": {
        "what": "Receiving yards per offensive snap on the field.",
        "how": "receiving_yards / off_snaps.",
        "limits": "Best available efficiency-of-role metric from free data.",
    },
    "targets_per_snap_z": {
        "what": "How often the QB looks your way per snap.",
        "how": "targets / off_snaps.",
        "limits": "Measures role, not skill.",
    },
    "yac_per_reception_z": {
        "what": "Average yards gained after the catch, per reception.",
        "how": "receiving_yards_after_catch / receptions.",
        "limits": "Credit shared between receiver, scheme, and blockers.",
    },
    "epa_per_target_z": {
        "what": "Expected Points Added per target.",
        "how": "Mean of nflverse EPA on this player's targets, from PBP.",
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
        "limits": "Depth-blind — doesn't distinguish route-running from scheme.",
    },
}


# ── The config ───────────────────────────────────────────────────────────────

WR_CONFIG = PositionConfig(
    key="wr",
    output_filenames=["league_wr_all_seasons.parquet", "league_te_all_seasons.parquet"],
    metadata_filename="wr_te_stat_metadata.json",
    snap_positions=["WR", "TE"],
    snap_floor={"WR": 100, "TE": 100},  # Decision 1
    min_games=6,
    pbp_play_types=["pass"],
    pbp_season_types=["REG"],  # Bug 3 fix
    use_player_stats=True,  # Decision 3 (hybrid)
    player_stats_col_map={
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "rec_yards",
        "receiving_tds": "rec_tds",
        "receiving_first_downs": "rec_first_downs",
        "receiving_air_yards": "air_yards",
        "receiving_yards_after_catch": "yac",
        "receiving_2pt_conversions": "rec_2pt",
        "receiving_fumbles": "rec_fumbles",
        "target_share": "target_share",
        "air_yards_share": "air_yards_share",
        "racr": "racr",
        "wopr": "wopr",
        "first_week": "first_week",
        "last_week": "last_week",
    },
    ngs_stat_type="receiving",
    pfr_stat_type=None,
    aggregate_stats=[agg_receiver_advanced],
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
    zscore_groups=["WR", "TE"],  # Decision 2
    output_columns=OUTPUT_COLUMNS,
    stat_tiers=STAT_TIERS,
    stat_labels=STAT_LABELS,
    stat_methodology=STAT_METHODOLOGY,
)

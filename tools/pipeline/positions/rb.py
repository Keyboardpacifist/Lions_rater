"""
RB position config.

Population: RBs with >= 100 total offensive snaps for the season,
min 6 games. One row per (player, team) stint.

Counting stats (carries, rushing/receiving yards, TDs, etc.) come from
nflverse pre-aggregated player_stats — handles trades and lateral-TD
attribution correctly. PBP is used only for advanced/situational stats
that nflverse doesn't pre-compute (success_rate, EPA per rush, explosive
runs, red-zone usage, short-yardage conversions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import PositionConfig, fo_success_per_play


# ── PBP aggregation: rushing (advanced/situational only) ─────────────────────


def agg_rusher_advanced(pbp: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Per-(player, team) advanced rushing metrics from PBP.

    Counting stats (carries, rush_yards, rush_tds) come from player_stats.
    Here we compute: epa_per_rush, rush_success_rate, explosive run counts,
    and red-zone / goal-line / short-yardage situational stats.
    """
    rush_plays = pbp[
        (pbp["play_type"] == "run") & (pbp["rusher_player_id"].notna())
    ].copy()

    if rush_plays.empty:
        return pd.DataFrame()

    rush_plays["is_explosive_10"] = (rush_plays["yards_gained"] >= 10).astype(int)
    rush_plays["is_explosive_15"] = (rush_plays["yards_gained"] >= 15).astype(int)
    if "yardline_100" in rush_plays.columns:
        rush_plays["is_gl"] = (rush_plays["yardline_100"] <= 5).astype(int)
        rush_plays["is_rz"] = (rush_plays["yardline_100"] <= 20).astype(int)
    else:
        rush_plays["is_gl"] = 0
        rush_plays["is_rz"] = 0

    if "ydstogo" in rush_plays.columns and "down" in rush_plays.columns:
        rush_plays["is_sy"] = (
            (rush_plays["ydstogo"] <= 2) & (rush_plays["down"].isin([3, 4]))
        ).astype(int)
    else:
        rush_plays["is_sy"] = 0

    rush_plays["is_sy_converted"] = (
        (rush_plays["is_sy"] == 1)
        & (rush_plays.get("first_down", pd.Series(0)).fillna(0) == 1)
    ).astype(int)
    rush_plays["is_gl_td"] = (
        (rush_plays["is_gl"] == 1)
        & (rush_plays.get("rush_touchdown", pd.Series(0)).fillna(0) == 1)
    ).astype(int)
    # FO/PFR success rate — replaces nflverse's EPA-based success
    # so our numbers align with PFF / Pro-Football-Reference.
    rush_plays["fo_success"] = fo_success_per_play(rush_plays)

    def _agg(group):
        return pd.Series({
            "pbp_carries": len(group),
            "epa_per_rush": group["epa"].mean(),
            "rush_success_rate": group["fo_success"].mean(),
            "explosive_10_count": group["is_explosive_10"].sum(),
            "explosive_15_count": group["is_explosive_15"].sum(),
            "rz_carries": group["is_rz"].sum(),
            "gl_attempts": group["is_gl"].sum(),
            "gl_tds": group["is_gl_td"].sum(),
            "sy_attempts": group["is_sy"].sum(),
            "sy_conversions": group["is_sy_converted"].sum(),
        })

    stats = (
        rush_plays.groupby(["rusher_player_id", "posteam"])
        .apply(_agg, include_groups=False)
        .reset_index()
        .rename(columns={"rusher_player_id": "gsis_id", "posteam": "team"})
    )
    return stats


# ── PBP aggregation: RB receiving (advanced only) ───────────────────────────


def agg_rb_receiver_advanced(pbp: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Per-(player, team) advanced receiving metrics for RBs.

    Counting stats come from player_stats. Here: rec_epa_per_target,
    rec_success_rate.
    """
    passes = pbp[
        (pbp["play_type"] == "pass") & (pbp["receiver_player_id"].notna())
    ].copy()

    if passes.empty:
        return pd.DataFrame()

    # FO/PFR success rate — aligns with PFF / PFR conventions
    passes["fo_success"] = fo_success_per_play(passes)

    def _agg(group):
        return pd.Series({
            "rec_epa_per_target": group["epa"].mean(),
            "rec_success_rate": group["fo_success"].mean(),
        })

    stats = (
        passes.groupby(["receiver_player_id", "posteam"])
        .apply(_agg, include_groups=False)
        .reset_index()
        .rename(columns={"receiver_player_id": "gsis_id", "posteam": "team"})
    )
    return stats


# ── Derived stats ────────────────────────────────────────────────────────────


def compute_rb_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived rate stats for RB from counting stats + PBP situational counts."""
    safe = lambda col: df[col].replace(0, np.nan) if col in df.columns else np.nan

    df["yards_per_carry"] = df.get("rush_yards", 0) / safe("carries")
    df["carries_per_game"] = df.get("carries", 0) / safe("games_played")
    # 65 = approximate offensive snaps per game (a rough but stable scale factor)
    df["snap_share"] = df.get("off_snaps", 0) / (
        df.get("games_played", 1) * 65
    ).replace(0, np.nan)
    df["touches_per_game"] = (
        df.get("carries", pd.Series(0)).fillna(0)
        + df.get("receptions", pd.Series(0)).fillna(0)
    ) / safe("games_played")
    df["targets_per_game"] = df.get("targets", pd.Series(0)).fillna(0) / safe("games_played")

    # Explosive / situational rates use PBP per-stint counts (from agg_rusher_advanced)
    # divided by counting carries from player_stats.
    df["explosive_run_rate"] = df.get("explosive_10_count", 0) / safe("carries")
    df["explosive_15_rate"] = df.get("explosive_15_count", 0) / safe("carries")
    df["rz_carry_share"] = df.get("rz_carries", 0) / safe("carries")
    df["goal_line_td_rate"] = df.get("gl_tds", 0) / safe("gl_attempts")
    df["short_yardage_conv_rate"] = df.get("sy_conversions", 0) / safe("sy_attempts")

    df["rec_yards_per_target"] = df.get("rec_yards", 0) / safe("targets")
    df["yac_per_reception"] = df.get("yac", 0) / safe("receptions")

    # PFR-derived rates (per-(player, season) from PFR; same value applied to
    # both stints for traded players).
    if "yards_before_contact_total" in df.columns:
        pfr_carries = df.get("pfr_carries", df.get("carries", 0)).replace(0, np.nan)
        df["yards_before_contact_per_att"] = df["yards_before_contact_total"] / pfr_carries
        df["yards_after_contact_per_att"] = df["yards_after_contact_total"] / pfr_carries
        df["broken_tackles_per_att"] = df["broken_tackles_total"] / pfr_carries

    return df


# ── Stats to z-score ─────────────────────────────────────────────────────────

STATS_TO_ZSCORE = [
    # Tier 1 — raw counts
    "rush_yards",
    "rush_tds",
    "carries",
    "receptions",
    "rec_yards",
    "rec_tds",
    # Tier 2 — rates and situational
    "yards_per_carry",
    "rush_success_rate",
    "carries_per_game",
    "snap_share",
    "touches_per_game",
    "targets_per_game",
    "explosive_run_rate",
    "explosive_15_rate",
    "rz_carry_share",
    "goal_line_td_rate",
    "short_yardage_conv_rate",
    "rec_yards_per_target",
    "yac_per_reception",
    "broken_tackles_per_att",
    "yards_before_contact_per_att",
    "yards_after_contact_per_att",
    # Tier 3 — modeled / NGS
    "epa_per_rush",
    "rec_epa_per_target",
    "ryoe_per_att",
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
    "carries",
    "rush_yards",
    "rush_tds",
    "rush_first_downs",
    "rush_2pt",
    "receptions",
    "targets",
    "rec_yards",
    "rec_tds",
    "rec_first_downs",
    "yac",
    # Derived rates
    "yards_per_carry",
    "rush_success_rate",
    "carries_per_game",
    "snap_share",
    "touches_per_game",
    "targets_per_game",
    "explosive_run_rate",
    "explosive_15_rate",
    "rz_carry_share",
    "goal_line_td_rate",
    "short_yardage_conv_rate",
    "rec_yards_per_target",
    "yac_per_reception",
    # PBP situational counts (raw)
    "rz_carries",
    "gl_attempts",
    "gl_tds",
    "sy_attempts",
    "sy_conversions",
    "explosive_10_count",
    "explosive_15_count",
    # PFR (per-player-season; same on both stints if traded)
    "broken_tackles_per_att",
    "yards_before_contact_per_att",
    "yards_after_contact_per_att",
    # PBP advanced
    "epa_per_rush",
    "rec_epa_per_target",
    "rec_success_rate",
    # NGS rushing
    "ryoe_per_att",
    "avg_time_to_los",
    "efficiency",
    "stacked_box_rate",
    # Z-scores
    "rush_yards_z",
    "rush_tds_z",
    "carries_z",
    "receptions_z",
    "rec_yards_z",
    "rec_tds_z",
    "yards_per_carry_z",
    "rush_success_rate_z",
    "carries_per_game_z",
    "snap_share_z",
    "touches_per_game_z",
    "targets_per_game_z",
    "explosive_run_rate_z",
    "explosive_15_rate_z",
    "rz_carry_share_z",
    "goal_line_td_rate_z",
    "short_yardage_conv_rate_z",
    "rec_yards_per_target_z",
    "yac_per_reception_z",
    "broken_tackles_per_att_z",
    "yards_before_contact_per_att_z",
    "yards_after_contact_per_att_z",
    "epa_per_rush_z",
    "rec_epa_per_target_z",
    "ryoe_per_att_z",
]

STAT_TIERS = {
    "rush_yards_z": 1,
    "rush_tds_z": 1,
    "carries_z": 1,
    "receptions_z": 1,
    "rec_yards_z": 1,
    "rec_tds_z": 1,
    "yards_per_carry_z": 2,
    "rush_success_rate_z": 2,
    "carries_per_game_z": 2,
    "snap_share_z": 2,
    "touches_per_game_z": 2,
    "targets_per_game_z": 2,
    "explosive_run_rate_z": 2,
    "explosive_15_rate_z": 2,
    "rz_carry_share_z": 2,
    "goal_line_td_rate_z": 2,
    "short_yardage_conv_rate_z": 2,
    "rec_yards_per_target_z": 2,
    "yac_per_reception_z": 2,
    "broken_tackles_per_att_z": 2,
    "yards_before_contact_per_att_z": 2,
    "yards_after_contact_per_att_z": 2,
    "epa_per_rush_z": 3,
    "rec_epa_per_target_z": 3,
    "ryoe_per_att_z": 3,
}

STAT_LABELS = {
    "rush_yards_z": "Rushing yards (raw)",
    "rush_tds_z": "Rushing TDs (raw)",
    "carries_z": "Carries (raw)",
    "receptions_z": "Receptions (raw)",
    "rec_yards_z": "Receiving yards (raw)",
    "rec_tds_z": "Receiving TDs (raw)",
    "yards_per_carry_z": "Yards per carry",
    "rush_success_rate_z": "Rush success rate",
    "carries_per_game_z": "Carries per game",
    "snap_share_z": "Snap share",
    "touches_per_game_z": "Touches per game",
    "targets_per_game_z": "Targets per game",
    "explosive_run_rate_z": "Explosive run rate (10+ yd)",
    "explosive_15_rate_z": "15+ yard run rate",
    "rz_carry_share_z": "Red zone carry share",
    "goal_line_td_rate_z": "Goal line TD rate",
    "short_yardage_conv_rate_z": "Short-yardage conversion rate",
    "rec_yards_per_target_z": "Receiving yards per target",
    "yac_per_reception_z": "YAC per reception",
    "broken_tackles_per_att_z": "Broken tackles per attempt",
    "yards_before_contact_per_att_z": "Yards before contact per attempt",
    "yards_after_contact_per_att_z": "Yards after contact per attempt",
    "epa_per_rush_z": "EPA per rush",
    "rec_epa_per_target_z": "Receiving EPA per target",
    "ryoe_per_att_z": "Rush yards over expected per attempt",
}

STAT_METHODOLOGY = {
    "rush_yards_z": {
        "what": "Total raw rushing yards (full season, REG only).",
        "how": "From nflverse load_player_stats(summary_level='week') summed per (player, team).",
        "limits": "Volume stat — usage-driven.",
    },
    "rush_tds_z": {
        "what": "Total raw rushing touchdowns.",
        "how": "From nflverse load_player_stats — correctly attributed even on lateral plays.",
        "limits": "Small integer samples are noisy.",
    },
    "carries_z": {
        "what": "Total raw rushing attempts.",
        "how": "From nflverse load_player_stats per (player, team) stint.",
        "limits": "Pure opportunity — not a skill measure.",
    },
    "receptions_z": {
        "what": "Total raw receptions.",
        "how": "From nflverse load_player_stats per stint.",
        "limits": "Volume stat.",
    },
    "rec_yards_z": {
        "what": "Total raw receiving yards.",
        "how": "From nflverse load_player_stats per stint.",
        "limits": "Volume stat. Pass-catching role drives this.",
    },
    "rec_tds_z": {
        "what": "Total raw receiving touchdowns.",
        "how": "From nflverse load_player_stats per stint.",
        "limits": "Small integer samples are noisy.",
    },
    "yards_per_carry_z": {
        "what": "Average yards per rushing attempt.",
        "how": "rush_yards / carries.",
        "limits": "Inflated by long runs; one or two big gains can spike a low-volume back.",
    },
    "rush_success_rate_z": {
        "what": "Percentage of rushes that produced a successful play by EPA standards.",
        "how": "Mean of nflverse binary success flag across this player's carries (per stint).",
        "limits": "Binary cutoff; doesn't differentiate near-misses from runaway successes.",
    },
    "carries_per_game_z": {
        "what": "Carries per game played.",
        "how": "carries / games_played.",
        "limits": "Pure usage — not skill.",
    },
    "snap_share_z": {
        "what": "Fraction of estimated team offensive snaps the player was on the field for.",
        "how": "off_snaps / (games_played * 65). 65 is an approximate league-average snap count per offensive game.",
        "limits": "Uses a constant denominator; teams with low- or high-snap offenses are normalized away.",
    },
    "touches_per_game_z": {
        "what": "Combined carries + receptions per game.",
        "how": "(carries + receptions) / games_played.",
        "limits": "Volume / role indicator.",
    },
    "targets_per_game_z": {
        "what": "Pass targets per game.",
        "how": "targets / games_played.",
        "limits": "Role-driven; pass-catching backs vs. early-down only.",
    },
    "explosive_run_rate_z": {
        "what": "Percentage of carries gaining 10+ yards.",
        "how": "PBP plays gaining ≥10 yards / total carries.",
        "limits": "Often scheme-aided; low-volume backs have noisy rates.",
    },
    "explosive_15_rate_z": {
        "what": "Percentage of carries gaining 15+ yards.",
        "how": "Same as above with a 15-yard cutoff.",
        "limits": "Same caveats; harder threshold = noisier.",
    },
    "rz_carry_share_z": {
        "what": "Share of the player's carries that came inside the opponent's 20.",
        "how": "rz_carries / carries.",
        "limits": "Usage signal: trust in scoring position.",
    },
    "goal_line_td_rate_z": {
        "what": "TD rate on carries inside the 5.",
        "how": "gl_tds / gl_attempts.",
        "limits": "Tiny samples — extreme values for backs with few goal-line carries.",
    },
    "short_yardage_conv_rate_z": {
        "what": "Conversion rate on 3rd/4th-and-2-or-shorter carries.",
        "how": "sy_conversions / sy_attempts.",
        "limits": "Small samples; counted as a first down only when the run gained one.",
    },
    "rec_yards_per_target_z": {
        "what": "Receiving yards per target.",
        "how": "rec_yards / targets.",
        "limits": "Penalizes drops as zeros.",
    },
    "yac_per_reception_z": {
        "what": "Average yards after the catch per reception.",
        "how": "yac / receptions.",
        "limits": "Credit shared with scheme and blockers.",
    },
    "broken_tackles_per_att_z": {
        "what": "Broken tackles per rushing attempt.",
        "how": "PFR data: broken tackles total / PFR carries.",
        "limits": "Subjective tracking by PFR; methodology can drift.",
    },
    "yards_before_contact_per_att_z": {
        "what": "Yards gained before being contacted, per attempt.",
        "how": "PFR data: yards before contact / PFR carries.",
        "limits": "More about offensive line + scheme than the runner.",
    },
    "yards_after_contact_per_att_z": {
        "what": "Yards gained after first contact, per attempt.",
        "how": "PFR data: yards after contact / PFR carries.",
        "limits": "Best single-stat playmaker measure for RBs.",
    },
    "epa_per_rush_z": {
        "what": "Expected Points Added per rush.",
        "how": "Mean of nflverse EPA on this player's carries (per stint).",
        "limits": "Depends on trusting the EPA model.",
    },
    "rec_epa_per_target_z": {
        "what": "Expected Points Added per target on receiving plays.",
        "how": "Mean of nflverse EPA on the player's targets (per stint).",
        "limits": "Same caveats as EPA generally.",
    },
    "ryoe_per_att_z": {
        "what": "Rush Yards Over Expected per attempt.",
        "how": "NFL Next Gen Stats: actual rush yards − model-expected, per attempt.",
        "limits": "Requires NGS tracking data; older seasons missing or partial.",
    },
}


# ── The config ───────────────────────────────────────────────────────────────

RB_CONFIG = PositionConfig(
    key="rb",
    output_filenames=["league_rb_all_seasons.parquet"],
    metadata_filename="rb_stat_metadata.json",
    snap_positions=["RB"],
    snap_floor={"RB": 100},  # Decision 1: 100+ snap floor
    min_games=6,
    pbp_play_types=["run", "pass"],
    pbp_season_types=["REG"],  # Bug 3 fix: regular season only
    use_player_stats=True,  # Decision 3: hybrid data source
    player_stats_col_map={
        "carries": "carries",
        "rushing_yards": "rush_yards",
        "rushing_tds": "rush_tds",
        "rushing_first_downs": "rush_first_downs",
        "rushing_2pt_conversions": "rush_2pt",
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "rec_yards",
        "receiving_tds": "rec_tds",
        "receiving_first_downs": "rec_first_downs",
        "receiving_air_yards": "air_yards",
        "receiving_yards_after_catch": "yac",
        "first_week": "first_week",
        "last_week": "last_week",
    },
    ngs_stat_type="rushing",
    pfr_stat_type="rush",
    aggregate_stats=[agg_rusher_advanced, agg_rb_receiver_advanced],
    ngs_col_map={
        "rush_yards_over_expected_per_att": "ryoe_per_att",
        "avg_time_to_los": "avg_time_to_los",
        "efficiency": "efficiency",
        "percent_attempts_gte_eight_defenders": "stacked_box_rate",
    },
    pfr_col_map={
        "att": "pfr_carries",
        "ybc": "yards_before_contact_total",
        "yac": "yards_after_contact_total",
        "brk_tkl": "broken_tackles_total",
    },
    compute_derived=compute_rb_derived,
    stats_to_zscore=STATS_TO_ZSCORE,
    invert_stats=set(),  # All higher-is-better
    zscore_groups=None,  # Single-position pool — no per-group split needed
    output_columns=OUTPUT_COLUMNS,
    stat_tiers=STAT_TIERS,
    stat_labels=STAT_LABELS,
    stat_methodology=STAT_METHODOLOGY,
)

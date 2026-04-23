"""
QB position config.

Population: QBs with >= 100 total offensive snaps for the season,
min 6 games. One row per (player, team) stint.

Counting stats from nflverse player_stats. PBP used for success rate
and per-play EPA on dropbacks (passing + scrambles). Career arc and
radar gain the same starter benchmarks as WR / RB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import PositionConfig


# ── PBP aggregation: passing advanced ────────────────────────────────────────


def agg_passing_advanced(pbp: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Per-(player, team) advanced passing metrics from PBP.

    Computes pass_success_rate (mean success flag on dropbacks) and
    pbp_dropbacks (count of attempts + sacks). Used to recompute
    pass_epa_per_play correctly per stint.
    """
    drops = pbp[
        (pbp["play_type"].isin(["pass", "qb_spike", "qb_kneel"]))
        & (pbp["passer_player_id"].notna())
    ].copy()

    if drops.empty:
        return pd.DataFrame()

    def _agg(group):
        return pd.Series({
            "pbp_dropbacks": len(group),
            "pass_success_rate": group["success"].mean(),
            "pbp_pass_epa_per_play": group["epa"].mean(),
            # CPOE per stint: mean across this QB's pass attempts on this team.
            # PBP cpoe is per-play; mean is the right per-stint aggregate.
            "passing_cpoe": (
                group.loc[group["play_type"] == "pass", "cpoe"].mean()
                if "cpoe" in group.columns else np.nan
            ),
        })

    stats = (
        drops.groupby(["passer_player_id", "posteam"])
        .apply(_agg, include_groups=False)
        .reset_index()
        .rename(columns={"passer_player_id": "gsis_id", "posteam": "team"})
    )
    return stats


# ── PBP aggregation: QB rushing (scrambles + designed) ───────────────────────


def agg_qb_rushing_advanced(pbp: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Per-(player, team) advanced rushing metrics for QBs."""
    rushes = pbp[
        (pbp["play_type"] == "run") & (pbp["rusher_player_id"].notna())
    ].copy()

    if rushes.empty:
        return pd.DataFrame()

    def _agg(group):
        return pd.Series({
            "rush_epa_per_carry": group["epa"].mean(),
            "rush_success_rate": group["success"].mean(),
        })

    stats = (
        rushes.groupby(["rusher_player_id", "posteam"])
        .apply(_agg, include_groups=False)
        .reset_index()
        .rename(columns={"rusher_player_id": "gsis_id", "posteam": "team"})
    )
    return stats


# ── Derived stats ────────────────────────────────────────────────────────────


def compute_qb_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stint derived rate stats from counting stats."""
    safe = lambda col: df[col].replace(0, np.nan) if col in df.columns else np.nan

    df["yards_per_attempt"] = df.get("passing_yards", 0) / safe("attempts")
    df["completion_pct"] = df.get("completions", 0) / safe("attempts")
    df["td_rate"] = df.get("passing_tds", 0) / safe("attempts")
    df["int_rate"] = df.get("passing_interceptions", 0) / safe("attempts")
    df["first_down_rate"] = df.get("passing_first_downs", 0) / safe("attempts")
    df["air_yards_per_attempt"] = df.get("passing_air_yards", 0) / safe("attempts")
    df["yac_per_completion"] = df.get("passing_yards_after_catch", 0) / safe("completions")

    # Sack rate uses dropbacks (attempts + sacks_suffered) as denominator
    if "attempts" in df.columns and "sacks_suffered" in df.columns:
        dropbacks = df["attempts"].fillna(0) + df["sacks_suffered"].fillna(0)
        df["sack_rate"] = df["sacks_suffered"] / dropbacks.replace(0, np.nan)
    else:
        df["sack_rate"] = np.nan

    # Turnover rate: (INTs + lost fumbles) / dropbacks
    fumbles_lost = df.get("sack_fumbles_lost", pd.Series(0)).fillna(0)
    if "passing_interceptions" in df.columns and "attempts" in df.columns:
        sacks = df.get("sacks_suffered", pd.Series(0)).fillna(0)
        dropbacks = df["attempts"].fillna(0) + sacks
        df["turnover_rate"] = (
            df["passing_interceptions"].fillna(0) + fumbles_lost
        ) / dropbacks.replace(0, np.nan)
    else:
        df["turnover_rate"] = np.nan

    # Pass EPA per play — prefer PBP per-stint value (correctly per-team).
    # Fall back to passing_epa / dropbacks from player_stats if PBP missing.
    if "pbp_pass_epa_per_play" in df.columns:
        df["pass_epa_per_play"] = df["pbp_pass_epa_per_play"]
    elif "passing_epa" in df.columns and "attempts" in df.columns:
        sacks = df.get("sacks_suffered", pd.Series(0)).fillna(0)
        dropbacks = df["attempts"].fillna(0) + sacks
        df["pass_epa_per_play"] = df["passing_epa"] / dropbacks.replace(0, np.nan)
    else:
        df["pass_epa_per_play"] = np.nan

    # Per-game rates use the stint's games_played (not season total)
    df["passing_yards_per_game"] = df.get("passing_yards", 0) / safe("games_played")
    df["passing_tds_per_game"] = df.get("passing_tds", 0) / safe("games_played")
    df["rush_yards_per_game"] = df.get("rushing_yards", 0) / safe("games_played")

    return df


# ── Stats to z-score ─────────────────────────────────────────────────────────

STATS_TO_ZSCORE = [
    # Tier 1 — per-game volume
    "passing_yards_per_game",
    "passing_tds_per_game",
    "rush_yards_per_game",
    # Tier 2 — rate stats
    "yards_per_attempt",
    "completion_pct",
    "td_rate",
    "int_rate",
    "first_down_rate",
    "air_yards_per_attempt",
    "yac_per_completion",
    "sack_rate",
    "turnover_rate",
    # Tier 3 — modeled
    "pass_epa_per_play",
    "passing_cpoe",
    "rush_epa_per_carry",
    "pass_success_rate",
]

INVERT_STATS = {
    # Lower is better for these
    "int_rate",
    "sack_rate",
    "turnover_rate",
}

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
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "sacks_suffered",
    "passing_first_downs",
    "passing_air_yards",
    "passing_yards_after_catch",
    "carries",
    "rushing_yards",
    "rushing_tds",
    "rushing_first_downs",
    # Derived rates
    "yards_per_attempt",
    "completion_pct",
    "td_rate",
    "int_rate",
    "first_down_rate",
    "air_yards_per_attempt",
    "yac_per_completion",
    "sack_rate",
    "turnover_rate",
    "passing_yards_per_game",
    "passing_tds_per_game",
    "rush_yards_per_game",
    # Modeled / advanced
    "pass_epa_per_play",
    "passing_cpoe",
    "rush_epa_per_carry",
    "pass_success_rate",
    "rush_success_rate",
    # Z-scores (matches what pages/QB.py expects via RAW_COL_MAP)
    "passing_yards_per_game_z",
    "passing_tds_per_game_z",
    "rush_yards_per_game_z",
    "yards_per_attempt_z",
    "completion_pct_z",
    "td_rate_z",
    "int_rate_z",
    "first_down_rate_z",
    "air_yards_per_attempt_z",
    "yac_per_completion_z",
    "sack_rate_z",
    "turnover_rate_z",
    "pass_epa_per_play_z",
    "passing_cpoe_z",
    "rush_epa_per_carry_z",
    "pass_success_rate_z",
]

# ── Metadata ─────────────────────────────────────────────────────────────────

STAT_TIERS = {
    "passing_yards_per_game_z": 1,
    "passing_tds_per_game_z": 1,
    "rush_yards_per_game_z": 1,
    "yards_per_attempt_z": 2,
    "completion_pct_z": 2,
    "td_rate_z": 2,
    "int_rate_z": 2,
    "first_down_rate_z": 2,
    "air_yards_per_attempt_z": 2,
    "yac_per_completion_z": 2,
    "sack_rate_z": 2,
    "turnover_rate_z": 2,
    "pass_epa_per_play_z": 3,
    "passing_cpoe_z": 3,
    "rush_epa_per_carry_z": 3,
    "pass_success_rate_z": 3,
}

STAT_LABELS = {
    "passing_yards_per_game_z": "Passing yards per game",
    "passing_tds_per_game_z": "Passing TDs per game",
    "rush_yards_per_game_z": "Rushing yards per game",
    "yards_per_attempt_z": "Yards per attempt",
    "completion_pct_z": "Completion %",
    "td_rate_z": "TD rate",
    "int_rate_z": "INT rate (lower is better)",
    "first_down_rate_z": "First-down rate",
    "air_yards_per_attempt_z": "Air yards per attempt",
    "yac_per_completion_z": "YAC per completion",
    "sack_rate_z": "Sack rate (lower is better)",
    "turnover_rate_z": "Turnover rate (lower is better)",
    "pass_epa_per_play_z": "Pass EPA per play",
    "passing_cpoe_z": "Completion % over expected",
    "rush_epa_per_carry_z": "Rush EPA per carry",
    "pass_success_rate_z": "Pass success rate",
}

STAT_METHODOLOGY = {
    "passing_yards_per_game_z": {
        "what": "Passing yards per game played.",
        "how": "From player_stats per stint, divided by stint games.",
        "limits": "Volume metric — opportunity-driven.",
    },
    "passing_tds_per_game_z": {
        "what": "Passing TDs per game played.",
        "how": "From player_stats per stint.",
        "limits": "Small samples noisy.",
    },
    "rush_yards_per_game_z": {
        "what": "Rushing yards per game (QB rushing only).",
        "how": "From player_stats rushing per stint.",
        "limits": "Pocket passers near zero by design.",
    },
    "yards_per_attempt_z": {
        "what": "Average yards per pass attempt.",
        "how": "passing_yards / attempts.",
        "limits": "Doesn't account for sacks.",
    },
    "completion_pct_z": {
        "what": "Percentage of attempts completed.",
        "how": "completions / attempts.",
        "limits": "Doesn't account for throw difficulty (CPOE does).",
    },
    "td_rate_z": {
        "what": "Touchdown passes per attempt.",
        "how": "passing_tds / attempts.",
        "limits": "Red-zone usage drives it.",
    },
    "int_rate_z": {
        "what": "Interceptions per attempt — lower is better.",
        "how": "passing_interceptions / attempts. Z inverted so positive = good.",
        "limits": "Tipped balls and bad luck inflate this.",
    },
    "first_down_rate_z": {
        "what": "First downs per pass attempt.",
        "how": "passing_first_downs / attempts.",
        "limits": "Down-and-distance situational.",
    },
    "air_yards_per_attempt_z": {
        "what": "Average intended air yards per attempt (downfield aggression).",
        "how": "passing_air_yards / attempts.",
        "limits": "Doesn't reward dink-and-dunk efficiency.",
    },
    "yac_per_completion_z": {
        "what": "Yards after the catch per completion.",
        "how": "passing_yards_after_catch / completions.",
        "limits": "Heavy receiver / scheme dependence.",
    },
    "sack_rate_z": {
        "what": "Sacks per dropback — lower is better.",
        "how": "sacks_suffered / (attempts + sacks_suffered). Z inverted.",
        "limits": "Offensive line quality is a major factor.",
    },
    "turnover_rate_z": {
        "what": "INTs + lost fumbles per dropback — lower is better.",
        "how": "(passing_interceptions + sack_fumbles_lost) / dropbacks. Z inverted.",
        "limits": "Conflates skill and bad luck.",
    },
    "pass_epa_per_play_z": {
        "what": "Expected Points Added per dropback.",
        "how": "Mean of nflverse EPA on this player's dropbacks (per stint).",
        "limits": "Depends on trusting the EPA model.",
    },
    "passing_cpoe_z": {
        "what": "Completion percentage over expected.",
        "how": "Actual completion rate − model-expected, weighted across attempts.",
        "limits": "A model decides what 'expected' means.",
    },
    "rush_epa_per_carry_z": {
        "what": "Expected Points Added per QB rush.",
        "how": "Mean EPA on the QB's carries (per stint).",
        "limits": "Few carries → noisy values.",
    },
    "pass_success_rate_z": {
        "what": "Percentage of dropbacks that produced a successful play by EPA standards.",
        "how": "Mean of nflverse binary success flag across this player's dropbacks (per stint).",
        "limits": "Binary cutoff hides near-misses.",
    },
}


# ── The config ───────────────────────────────────────────────────────────────

QB_CONFIG = PositionConfig(
    key="qb",
    output_filenames=["league_qb_all_seasons.parquet"],
    metadata_filename="qb_stat_metadata.json",
    snap_positions=["QB"],
    snap_floor={"QB": 100},  # Decision 1
    min_games=6,
    pbp_play_types=["pass", "run", "qb_spike", "qb_kneel"],
    pbp_season_types=["REG"],  # Bug 3
    use_player_stats=True,  # Decision 3 (hybrid)
    player_stats_col_map={
        # Counting stats — sum cleanly across the player's weeks on a team
        "attempts": "attempts",
        "completions": "completions",
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        "passing_interceptions": "passing_interceptions",
        "sacks_suffered": "sacks_suffered",
        "sack_yards_lost": "sack_yards_lost",
        "sack_fumbles": "sack_fumbles",
        "sack_fumbles_lost": "sack_fumbles_lost",
        "passing_first_downs": "passing_first_downs",
        "passing_air_yards": "passing_air_yards",
        "passing_yards_after_catch": "passing_yards_after_catch",
        "passing_2pt_conversions": "passing_2pt_conversions",
        "passing_epa": "passing_epa",
        "carries": "carries",
        "rushing_yards": "rushing_yards",
        "rushing_tds": "rushing_tds",
        "rushing_first_downs": "rushing_first_downs",
        "rushing_2pt_conversions": "rushing_2pt_conversions",
        "rushing_fumbles": "rushing_fumbles",
        "rushing_fumbles_lost": "rushing_fumbles_lost",
        "first_week": "first_week",
        "last_week": "last_week",
        # passing_cpoe deliberately excluded — it's a percentage and would
        # be incorrect if summed across weeks. Computed per stint by
        # agg_passing_advanced from PBP instead.
    },
    ngs_stat_type=None,  # NGS passing exists but isn't wired to the page yet
    pfr_stat_type=None,
    aggregate_stats=[agg_passing_advanced, agg_qb_rushing_advanced],
    ngs_col_map={},
    pfr_col_map={},
    compute_derived=compute_qb_derived,
    stats_to_zscore=STATS_TO_ZSCORE,
    invert_stats=INVERT_STATS,
    zscore_groups=None,  # Single position
    output_columns=OUTPUT_COLUMNS,
    stat_tiers=STAT_TIERS,
    stat_labels=STAT_LABELS,
    stat_methodology=STAT_METHODOLOGY,
)

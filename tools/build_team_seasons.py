#!/usr/bin/env python3
"""
Per (team, season) aggregated stats with z-scores — foundation for
the team rater + league-wide leaderboard + matchup engine.

Output: data/team_seasons.parquet + data/team_stat_metadata.json

Stat coverage:
- Offensive efficiency  (EPA, success, explosives, red zone)
- Defensive efficiency  (EPA allowed, takeaways, pressure, red zone D)
- Scoring               (PPG, points allowed, scoring drive rate)
- Discipline            (penalty rate, turnover differential)
- Situational           (3rd down, 4Q EPA)

Run:
    python tools/build_team_seasons.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
OUTPUT = DATA / "team_seasons.parquet"
META = DATA / "team_stat_metadata.json"

SEASONS = list(range(2016, 2026))

# Stats where lower is better — z-score inverted at the end
INVERTED_STATS = {
    "def_epa_per_play",
    "def_pass_epa_allowed",
    "def_rush_epa_allowed",
    "def_red_zone_td_rate_allowed",
    "def_third_down_conv_allowed",
    "off_giveaway_rate",
    "points_allowed_per_game",
    "penalty_yards_per_game",
}


def main() -> None:
    import nflreadpy as nfl

    print(f"Pulling pbp for {SEASONS[0]}-{SEASONS[-1]}…")
    pbp = nfl.load_pbp(SEASONS).to_pandas()
    pbp = pbp[pbp["season_type"] == "REG"]
    print(f"  {len(pbp):,} regular-season plays loaded")

    # Slim to fields we need (dedupe — pandas chokes on duplicate cols)
    keep = list(dict.fromkeys([
        "season", "week", "game_id", "posteam", "defteam", "play_type",
        "epa", "success", "yards_gained", "complete_pass",
        "interception", "fumble_lost", "fumble",
        "qb_hit", "sack", "pass_attempt", "rush_attempt",
        "down", "ydstogo", "yardline_100", "qtr", "score_differential",
        "first_down", "first_down_pass", "first_down_rush",
        "third_down_converted", "third_down_failed",
        "td_team", "touchdown", "pass_touchdown", "rush_touchdown",
        "no_huddle",
        "penalty", "penalty_yards", "penalty_team",
    ]))
    keep = [c for c in keep if c in pbp.columns]
    p = pbp[keep].copy()

    # ── Offensive stats per (posteam, season) ──
    off = p[p["posteam"].notna()].copy()
    off_grouped = off.groupby(["posteam", "season"])
    off_stats = pd.DataFrame({
        "off_plays":             off_grouped.size(),
        "off_epa_per_play":      off_grouped["epa"].mean(),
        "off_success_rate":      off_grouped["success"].mean(),
        "off_explosive_rate":    off_grouped["yards_gained"].apply(lambda s: (s >= 20).mean()),
    }).reset_index().rename(columns={"posteam": "team"})

    # Pass / rush splits
    pass_off = off[off["pass_attempt"] == 1].groupby(["posteam", "season"])
    rush_off = off[off["rush_attempt"] == 1].groupby(["posteam", "season"])
    off_stats = off_stats.merge(
        pass_off["epa"].mean().reset_index().rename(
            columns={"epa": "off_pass_epa_per_play", "posteam": "team"}),
        on=["team", "season"], how="left",
    )
    off_stats = off_stats.merge(
        rush_off["epa"].mean().reset_index().rename(
            columns={"epa": "off_rush_epa_per_play", "posteam": "team"}),
        on=["team", "season"], how="left",
    )

    # Red zone TD rate (within opponent's 20)
    rz_off = off[(off["yardline_100"] <= 20) & (off["yardline_100"].notna())]
    rz_drives = rz_off.groupby(["posteam", "season", "game_id"]).agg(
        had_rz=("yardline_100", "size"),
        scored_td=("touchdown", "max"),
    ).reset_index()
    rz_summary = rz_drives.groupby(["posteam", "season"]).agg(
        rz_drives=("had_rz", "size"),
        rz_tds=("scored_td", "sum"),
    ).reset_index().rename(columns={"posteam": "team"})
    rz_summary["off_red_zone_td_rate"] = (
        rz_summary["rz_tds"] / rz_summary["rz_drives"]
    )
    off_stats = off_stats.merge(
        rz_summary[["team", "season", "off_red_zone_td_rate"]],
        on=["team", "season"], how="left",
    )

    # Third-down conversion
    third = off[off["down"] == 3]
    third_summary = third.groupby(["posteam", "season"]).agg(
        third_attempts=("third_down_converted", "size"),
        third_converted=("third_down_converted", "sum"),
    ).reset_index().rename(columns={"posteam": "team"})
    third_summary["off_third_down_conv_rate"] = (
        third_summary["third_converted"] / third_summary["third_attempts"]
    )
    off_stats = off_stats.merge(
        third_summary[["team", "season", "off_third_down_conv_rate"]],
        on=["team", "season"], how="left",
    )

    # Giveaways (turnovers committed)
    off_stats = off_stats.merge(
        off.groupby(["posteam", "season"]).agg(
            giveaways=("interception", "sum"),
            fumbles_lost=("fumble_lost", "sum"),
        ).reset_index().rename(columns={"posteam": "team"}),
        on=["team", "season"], how="left",
    )
    off_stats["off_giveaway_rate"] = (
        (off_stats["giveaways"] + off_stats["fumbles_lost"])
        / off_stats["off_plays"]
    )

    # ── Defensive stats per (defteam, season) ──
    def_grouped = off.groupby(["defteam", "season"])  # offensive plays AGAINST them
    def_stats = pd.DataFrame({
        "def_plays":               def_grouped.size(),
        "def_epa_per_play":        def_grouped["epa"].mean(),  # what THEY allow
        "def_success_rate_allowed":def_grouped["success"].mean(),
    }).reset_index().rename(columns={"defteam": "team"})

    def_pass = off[off["pass_attempt"] == 1].groupby(["defteam", "season"])
    def_rush = off[off["rush_attempt"] == 1].groupby(["defteam", "season"])
    def_stats = def_stats.merge(
        def_pass["epa"].mean().reset_index().rename(
            columns={"epa": "def_pass_epa_allowed", "defteam": "team"}),
        on=["team", "season"], how="left",
    )
    def_stats = def_stats.merge(
        def_rush["epa"].mean().reset_index().rename(
            columns={"epa": "def_rush_epa_allowed", "defteam": "team"}),
        on=["team", "season"], how="left",
    )

    # Takeaways (their offense's giveaways = their takeaways)
    take = off.groupby(["defteam", "season"]).agg(
        takeaway_int=("interception", "sum"),
        takeaway_fum=("fumble_lost", "sum"),
    ).reset_index().rename(columns={"defteam": "team"})
    take["def_takeaway_rate"] = (
        (take["takeaway_int"] + take["takeaway_fum"]) / def_stats["def_plays"]
    )
    def_stats = def_stats.merge(
        take[["team", "season", "def_takeaway_rate"]],
        on=["team", "season"], how="left",
    )

    # Pressure rate (qb_hits when defending pass plays)
    pass_def = off[off["pass_attempt"] == 1].groupby(["defteam", "season"]).agg(
        pass_plays_defended=("qb_hit", "size"),
        qb_hits_generated=("qb_hit", "sum"),
        sacks=("sack", "sum"),
    ).reset_index().rename(columns={"defteam": "team"})
    pass_def["def_pressure_rate"] = pass_def["qb_hits_generated"] / pass_def["pass_plays_defended"]
    pass_def["def_sack_rate"] = pass_def["sacks"] / pass_def["pass_plays_defended"]
    def_stats = def_stats.merge(
        pass_def[["team", "season", "def_pressure_rate", "def_sack_rate"]],
        on=["team", "season"], how="left",
    )

    # ── Combine offense + defense ──
    teams = off_stats.merge(def_stats, on=["team", "season"], how="outer")

    # ── Scoring (need points-per-game from schedules) ──
    print("Pulling schedules for scoring data…")
    sched = nfl.load_schedules(SEASONS).to_pandas()
    sched = sched[sched["game_type"] == "REG"]

    home = sched[["season", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team",
                 "home_score": "points_scored",
                 "away_score": "points_allowed"})
    away = sched[["season", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team",
                 "away_score": "points_scored",
                 "home_score": "points_allowed"})
    games = pd.concat([home, away], ignore_index=True)
    scoring = games.groupby(["team", "season"]).agg(
        games=("points_scored", "size"),
        points_scored=("points_scored", "sum"),
        points_allowed=("points_allowed", "sum"),
    ).reset_index()
    scoring["points_per_game"] = scoring["points_scored"] / scoring["games"]
    scoring["points_allowed_per_game"] = scoring["points_allowed"] / scoring["games"]
    scoring["point_differential_per_game"] = (
        scoring["points_per_game"] - scoring["points_allowed_per_game"]
    )
    teams = teams.merge(
        scoring[["team", "season", "games", "points_per_game",
                  "points_allowed_per_game", "point_differential_per_game"]],
        on=["team", "season"], how="left",
    )

    # ── Discipline (penalties per game) ──
    pen = p[(p["penalty"] == 1) & p["penalty_team"].notna()].groupby(
        ["penalty_team", "season"]).agg(
        penalty_count=("penalty", "size"),
        penalty_yards=("penalty_yards", "sum"),
    ).reset_index().rename(columns={"penalty_team": "team"})
    pen["penalty_yards_per_game"] = pen["penalty_yards"] / 17  # approximate
    teams = teams.merge(
        pen[["team", "season", "penalty_yards_per_game"]],
        on=["team", "season"], how="left",
    )

    # ── 4th-quarter EPA / play (clutch metric) ──
    q4 = off[off["qtr"] == 4]
    q4_off = q4.groupby(["posteam", "season"])["epa"].mean().reset_index().rename(
        columns={"posteam": "team", "epa": "fourth_q_off_epa"})
    teams = teams.merge(q4_off, on=["team", "season"], how="left")
    q4_def = q4.groupby(["defteam", "season"])["epa"].mean().reset_index().rename(
        columns={"defteam": "team", "epa": "fourth_q_def_epa"})
    teams = teams.merge(q4_def, on=["team", "season"], how="left")

    # ── Z-score everything (per-stat across all team-seasons) ──
    Z_STATS = [
        "off_epa_per_play", "off_pass_epa_per_play", "off_rush_epa_per_play",
        "off_success_rate", "off_explosive_rate", "off_red_zone_td_rate",
        "off_third_down_conv_rate", "off_giveaway_rate",
        "def_epa_per_play", "def_pass_epa_allowed", "def_rush_epa_allowed",
        "def_success_rate_allowed", "def_takeaway_rate",
        "def_pressure_rate", "def_sack_rate",
        "points_per_game", "points_allowed_per_game",
        "point_differential_per_game", "penalty_yards_per_game",
        "fourth_q_off_epa", "fourth_q_def_epa",
    ]
    for s in Z_STATS:
        if s not in teams.columns:
            continue
        col = teams[s].astype(float)
        mu, sigma = col.mean(), col.std(ddof=0)
        if sigma > 0:
            z = (col - mu) / sigma
            if s in INVERTED_STATS:
                z = -z
            teams[f"{s}_z"] = z
        else:
            teams[f"{s}_z"] = np.nan

    # Tidy columns
    teams = teams.sort_values(["season", "team"]).reset_index(drop=True)

    DATA.mkdir(parents=True, exist_ok=True)
    teams.to_parquet(OUTPUT, index=False)
    print(f"\n✓ wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(teams):,} (team, season) rows × {teams.shape[1]} cols")

    # Stat metadata for the rater UI
    metadata = {
        "stat_labels": {
            "off_epa_per_play_z":          "Offensive EPA/play",
            "off_pass_epa_per_play_z":     "Pass EPA/play",
            "off_rush_epa_per_play_z":     "Rush EPA/play",
            "off_success_rate_z":          "Offensive success rate",
            "off_explosive_rate_z":        "Explosive play rate",
            "off_red_zone_td_rate_z":      "Red zone TD rate",
            "off_third_down_conv_rate_z":  "3rd down conversion",
            "off_giveaway_rate_z":         "Ball security (low giveaways)",
            "def_epa_per_play_z":          "Defensive EPA allowed",
            "def_pass_epa_allowed_z":      "Pass EPA allowed",
            "def_rush_epa_allowed_z":      "Rush EPA allowed",
            "def_success_rate_allowed_z":  "Success rate allowed",
            "def_takeaway_rate_z":         "Takeaway rate",
            "def_pressure_rate_z":         "Pressure rate",
            "def_sack_rate_z":             "Sack rate",
            "points_per_game_z":           "Points/game",
            "points_allowed_per_game_z":   "Points allowed/game",
            "point_differential_per_game_z":"Point differential/game",
            "penalty_yards_per_game_z":    "Discipline (low penalties)",
            "fourth_q_off_epa_z":          "4th-quarter offense (clutch)",
            "fourth_q_def_epa_z":          "4th-quarter defense (clutch)",
        },
        "stat_tiers": {  # 1 = counted, 2 = rate, 3 = modeled
            f"{s}_z": (3 if "epa" in s else 2)
            for s in Z_STATS
        },
        "inverted_stats": list(INVERTED_STATS),
    }
    META.write_text(json.dumps(metadata, indent=2))
    print(f"✓ wrote {META.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

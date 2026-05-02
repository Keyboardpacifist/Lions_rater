"""Player Prop Report — auto-generated multi-section player analysis.

Pick a player + the upcoming week's matchup and the engine returns a
full prop-bet report pulling from every prop engine in the lab. This
is the prop-bet equivalent of the Matchup Report — the lazy-bettor
showcase for retail users.

Sections:
  • headline           — player meta + opponent + matchup context
  • recent_form        — last N game stat lines
  • decomposed         — primary-stat projection with audit rows
  • alt_line_ladder    — default rung ladder ranked by EV at -110
  • td_vector          — anytime / rushing / receiving TD probabilities
  • trend_divergence   — recent role-shift flags
  • longest_play       — distribution + threshold lookups
  • sgp_partners       — top correlated teammates for stack ideas
  • bottom_line        — bet-actionable bullets

Public entry point
------------------
    generate_player_report(player_id, position, season, week,
                            primary_stat=None, opponent=None)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lib_alt_line_ev import rank_ladder
from lib_decomposed_projection import decompose
from lib_longest_play import longest_play_distribution, p_longest_at_least
from lib_td_probability import (
    player_td_rates,
    rz_usage_share,
    td_probability_vector,
)
from lib_trend_divergence import compute_player_window
from lib_weather import primary_stat_for_position


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
PLAYER_STATS = DATA / "nfl_player_stats_weekly.parquet"
SCHEDULES = DATA / "nfl_schedules.parquet"
SGP = DATA / "sgp_correlations.parquet"
DVP = DATA / "dvp.parquet"


@st.cache_data(show_spinner=False)
def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@dataclass
class PlayerPropReport:
    player_id: str
    player_name: str
    position: str
    team: str
    season: int
    week: int
    primary_stat: str
    opponent: str | None
    headline: dict
    recent_form: pd.DataFrame
    decomposition: dict | None
    alt_ladder: pd.DataFrame
    td_vector: dict
    rz_usage: dict
    trend_flags: list[dict]
    longest_play: dict | None
    sgp_partners: list[dict]
    bottom_line_bullets: list[str] = field(default_factory=list)


def _pick_primary_stat(position: str) -> str:
    return primary_stat_for_position(position) or "receiving_yards"


def _baseline_thresholds(baseline: float, n_rungs: int = 5
                          ) -> list[float]:
    """Build a sensible alt-line ladder around a player's baseline.
    Centered on baseline with ±20%/±35%/±50% rungs."""
    rungs = []
    for pct in (-0.35, -0.15, 0.0, 0.15, 0.35):
        rungs.append(round(baseline * (1 + pct) - 0.5, 0) + 0.5)
    return rungs


def _recent_form(player_id: str, season: int, week: int,
                  lookback: int = 5) -> pd.DataFrame:
    """Last N games BEFORE the target (season, week) — never future."""
    df = _load(PLAYER_STATS)
    if df.empty:
        return df
    sub = df[df["player_id"] == player_id].copy()
    # Composite ordering: (season, week) before target
    sub = sub[(sub["season"] < int(season))
              | ((sub["season"] == int(season))
                 & (sub["week"] < int(week)))]
    sub = sub.sort_values(["season", "week"], ascending=[False, False])
    return sub.head(lookback)


def _opponent_for(player_team: str, season: int,
                   week: int) -> str | None:
    sch = _load(SCHEDULES)
    if sch.empty:
        return None
    g = sch[(sch["season"] == int(season))
            & (sch["week"] == int(week))
            & ((sch["home_team"] == player_team)
               | (sch["away_team"] == player_team))]
    if g.empty:
        return None
    row = g.iloc[0]
    return (row["away_team"] if row["home_team"] == player_team
            else row["home_team"])


def _sgp_partners(team: str, season: int, player_name: str | None = None
                   ) -> list[dict]:
    """List top correlated stack candidates for this team-season.
    Filters out the player himself (so we don't recommend stacking
    Chase with Chase)."""
    sgp = _load(SGP)
    if sgp.empty:
        return []
    rows = sgp[(sgp["team"] == team) & (sgp["season"] == int(season))]
    if player_name:
        rows = rows[rows["partner_name"] != player_name]
    if rows.empty:
        return []
    return rows.sort_values("corr_qb_yds_partner_yds",
                            ascending=False).to_dict("records")


def _trend_flags(player_id: str, season: int, week: int
                  ) -> list[dict]:
    rows = compute_player_window(player_id, season, week, lookback=3)
    flags = [r for r in rows if abs(r.delta_z) >= 0.7]
    return [{
        "stat": r.stat,
        "recent_avg": r.recent_avg,
        "season_avg": r.season_avg,
        "delta": r.delta,
        "delta_z": r.delta_z,
    } for r in flags]


def _longest_play_summary(player_id: str, position: str
                            ) -> dict | None:
    if position not in ("WR", "TE"):
        kind = "rush" if position == "RB" else None
    else:
        kind = "reception"
    if not kind:
        return None
    # Threshold = roughly 25 yds for receptions, 15 for rushes
    threshold = 25.0 if kind == "reception" else 15.0
    r = p_longest_at_least(player_id, threshold, kind=kind)
    if r.n_games < 8:
        return None
    return {
        "kind": kind,
        "threshold": threshold,
        "p_at_least": r.p_at_least,
        "median_longest": r.median_longest,
        "p10_longest": r.p10_longest,
        "p90_longest": r.p90_longest,
        "n_games": r.n_games,
    }


def _bottom_line(report: PlayerPropReport) -> list[str]:
    bullets: list[str] = []
    proj = (report.decomposition.get("projection") if
            report.decomposition else None)
    base = (report.decomposition.get("baseline") if
            report.decomposition else None)
    if proj is not None and base is not None:
        delta = proj - base
        bullets.append(
            f"**Projection:** {proj:.1f} {report.primary_stat} "
            f"({delta:+.1f} vs. recent baseline of {base:.1f})"
        )

    # Best EV rung
    if not report.alt_ladder.empty:
        best = report.alt_ladder.iloc[0]
        if best["ev"] >= 0.05:
            bullets.append(
                f"**Best EV rung:** {best['side']} {best['threshold']:.1f} "
                f"@ {best['american_odds']:+d} → "
                f"{best['ev']:+.0%} EV (model {best['p_model']:.0%} "
                f"vs implied {best['p_implied']:.0%})"
            )

    # TD lean
    td = report.td_vector
    if td and td.get("p_any_td"):
        if td["p_any_td"] > 0.45:
            bullets.append(
                f"**TD lean:** {td['p_any_td']:.0%} anytime TD "
                f"(rush {td['p_rush_td']:.0%} / rec {td['p_rec_td']:.0%}) "
                f"— elevated"
            )
        elif td["p_any_td"] > 0.30 and td.get("p_rush_td", 0) > 0.20:
            bullets.append(
                f"**TD lean:** {td['p_any_td']:.0%} anytime TD with "
                f"strong rush-TD-only sub-prop "
                f"({td['p_rush_td']:.0%}) — check rushing-only line"
            )

    # Trend divergence
    if report.trend_flags:
        big = max(report.trend_flags, key=lambda x: abs(x["delta_z"]))
        direction = "expanded" if big["delta"] > 0 else "shrunk"
        bullets.append(
            f"**Trend:** {big['stat']} role has {direction} recently "
            f"({big['recent_avg']:.1f} last 3 vs {big['season_avg']:.1f} "
            f"season, z={big['delta_z']:+.1f}) — book may be anchored "
            f"to season"
        )

    # SGP idea (top partner)
    if report.sgp_partners:
        top = report.sgp_partners[0]
        if (top.get("corr_qb_yds_partner_yds", 0) > 0.6
                and top.get("lift_partner_75_given_qb_300", 0) > 0.10):
            bullets.append(
                f"**SGP idea:** stack with {top['partner_name']} "
                f"(corr {top['corr_qb_yds_partner_yds']:.2f}, "
                f"lift {top['lift_partner_75_given_qb_300']:+.0%})"
            )

    # Longest-play
    if report.longest_play:
        lp = report.longest_play
        bullets.append(
            f"**Longest {lp['kind']}:** P50 {lp['median_longest']:.0f} yds, "
            f"P({lp['threshold']:.0f}+) = {lp['p_at_least']:.0%}"
        )

    if not bullets:
        bullets.append("No flag-level signals. Default to model "
                        "projection vs. book line.")
    return bullets


def generate_player_report(player_id: str, position: str,
                            season: int, week: int,
                            primary_stat: str | None = None,
                            opponent: str | None = None,
                            lookback_games: int = 12,
                            target_temp: float | None = None,
                            target_wind: float | None = None,
                            ) -> PlayerPropReport:
    """Top-level entry point."""
    df = _load(PLAYER_STATS)
    if df.empty:
        raise ValueError("Player stats not available")
    sub = df[df["player_id"] == player_id]
    if sub.empty:
        raise ValueError(f"Unknown player_id: {player_id}")
    most_recent = sub.sort_values(["season", "week"],
                                  ascending=[False, False]).iloc[0]
    name = str(most_recent["player_display_name"])
    team = str(most_recent["team"])
    pos = position or str(most_recent["position"])
    stat = primary_stat or _pick_primary_stat(pos)
    opp = opponent or _opponent_for(team, season, week)

    # Headline
    sch = _load(SCHEDULES)
    headline = {"player": name, "team": team, "position": pos,
                "stat": stat, "opp": opp,
                "season": season, "week": week}
    if not sch.empty:
        g = sch[(sch["season"] == int(season))
                & (sch["week"] == int(week))
                & ((sch["home_team"] == team)
                   | (sch["away_team"] == team))]
        if not g.empty:
            r = g.iloc[0]
            spread_pov = (-float(r["spread_line"])
                           if r["home_team"] == team
                           else float(r["spread_line"]))
            headline.update({
                "spread": spread_pov,
                "total": (float(r["total_line"])
                          if pd.notna(r.get("total_line")) else None),
                "is_home": (r["home_team"] == team),
                "stadium": str(r.get("stadium", "?")),
                "roof": str(r.get("roof", "?")),
                "temp": (float(r["temp"])
                          if pd.notna(r.get("temp")) else None),
                "wind": (float(r["wind"])
                          if pd.notna(r.get("wind")) else None),
            })

    # Recent form
    recent = _recent_form(player_id, season=season, week=week, lookback=5)
    keep_cols = ["season", "week", "team", "opponent_team",
                  "passing_yards", "rushing_yards", "receiving_yards",
                  "completions", "attempts", "carries",
                  "targets", "receptions",
                  "passing_tds", "rushing_tds", "receiving_tds"]
    keep_cols = [c for c in keep_cols if c in recent.columns]
    recent_form_df = recent[keep_cols].reset_index(drop=True)

    # Decomposed projection
    decomp = decompose(
        player_id=player_id, position=pos, team=team,
        stat=stat, opponent=opp,
        season=int(season), week=int(week),
        target_temp=(target_temp if target_temp is not None
                     else headline.get("temp")),
        target_wind=(target_wind if target_wind is not None
                     else headline.get("wind")),
        target_roof=headline.get("roof"),
        lookback_games=lookback_games,
    )
    decomposition = {
        "baseline": decomp.baseline,
        "projection": decomp.projection,
        "n_games_baseline": decomp.n_games_baseline,
        "contributions": [
            {"label": c.label, "delta": c.delta, "note": c.note}
            for c in decomp.contributions
        ],
    }

    # Alt-line ladder around baseline
    rungs = _baseline_thresholds(decomp.baseline)
    ladder_input = []
    for thr in rungs:
        # Default to -110 over and -110 under at each rung (no book
        # available); the lab tab lets users paste real prices.
        ladder_input.append((thr, "over", -110))
    alt = rank_ladder(player_id, stat, ladder_input,
                       lookback_games=lookback_games)

    # TD vector + RZ usage
    v = td_probability_vector(player_id, opp_team=opp,
                               season=int(season),
                               lookback_games=lookback_games)
    u = rz_usage_share(player_id, int(season), team=team)
    td_vector = {
        "p_rush_td": v.p_rush_td_baseline,
        "p_rec_td": v.p_rec_td_baseline,
        "p_any_td": v.p_any_td_baseline,
        "n_games": v.n_games_player,
    }
    rz_usage = {
        "rz_carries_share": u.rz_carries_share,
        "rz_targets_share": u.rz_targets_share,
        "goal_line_carries_share": u.goal_line_carries_share,
        "n_team_rz_plays": u.n_team_rz_plays,
    }

    # Trend
    trend_flags = _trend_flags(player_id, season, week)

    # Longest play
    longest = _longest_play_summary(player_id, pos)

    # SGP partners — exclude the player themselves
    sgp_partners = _sgp_partners(team, season, player_name=name)[:5]

    report = PlayerPropReport(
        player_id=player_id, player_name=name,
        position=pos, team=team, season=int(season),
        week=int(week), primary_stat=stat, opponent=opp,
        headline=headline, recent_form=recent_form_df,
        decomposition=decomposition, alt_ladder=alt,
        td_vector=td_vector, rz_usage=rz_usage,
        trend_flags=trend_flags, longest_play=longest,
        sgp_partners=sgp_partners,
    )
    report.bottom_line_bullets = _bottom_line(report)
    return report

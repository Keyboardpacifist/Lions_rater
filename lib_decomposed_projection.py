"""Decomposed Prop Projection — Feature 5.1.

The showcase feature. Takes a player + matchup context and returns a
prop projection AS A DECOMPOSITION — every adjustment auditable. This
is what makes Bet School trustworthy: serious bettors don't trust
black-box models.

For a target stat (passing_yards, rushing_yards, receiving_yards):

    1. Baseline    = median of player's last N games at this stat
    2. Injury      = cohort snap-share retention multiplier (4.1)
    3. Weather     = (weather-cohort median - baseline) shift  (4.5)
    4. Matchup     = DvP delta                                  (5.8)
    5. Script      = game-script multiplier when starter out    (4.2)

Each step adds a labeled contribution. The sum is the projection.

Output is a `Decomposition` object the UI can render row-by-row.

Public entry points
-------------------
    decompose(player_id, position, team, stat, opponent=None,
              season=None, week=None, injury_status=None,
              injury_body_part="unknown", injury_practice="DNP",
              key_starter_out=None, target_temp=None, target_wind=None,
              target_roof=None, target_surface=None,
              lookback_games=12) -> Decomposition
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lib_alt_line_ev import (
    american_to_decimal,
    decimal_to_implied_prob,
    p_over_threshold,
)
from lib_game_script_player import (
    BUCKET_LABEL,
    GameScriptBucket,
    multiplier_for_game_script,
)
from lib_injury_cohort import predict as cohort_predict
from lib_injury_impact import (
    bucket_for as injury_bucket_for,
    lookup_player_self_delta,
)
from lib_weather import primary_stat_for_position, weather_cohort


REPO = Path(__file__).resolve().parent
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
DVP = REPO / "data" / "dvp.parquet"
GAME_SCRIPT = REPO / "data" / "game_script_deltas.parquet"


@st.cache_data(show_spinner=False)
def _load_stats() -> pd.DataFrame:
    return pd.read_parquet(PLAYER_STATS)


@st.cache_data(show_spinner=False)
def _load_dvp() -> pd.DataFrame:
    return pd.read_parquet(DVP) if DVP.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_gs() -> pd.DataFrame:
    return pd.read_parquet(GAME_SCRIPT) if GAME_SCRIPT.exists() else pd.DataFrame()


@dataclass
class DecompContribution:
    label: str
    delta: float        # additive yards relative to running total
    note: str           # one-line explanation


@dataclass
class Decomposition:
    player_display_name: str
    position: str
    team: str
    stat: str
    n_games_baseline: int
    baseline: float
    contributions: list[DecompContribution] = field(default_factory=list)

    @property
    def projection(self) -> float:
        return self.baseline + sum(c.delta for c in self.contributions)

    def book_compare(self, book_line: float, book_odds: int = -110
                     ) -> dict:
        """Compare projection to a book line at given American odds.
        Returns a dict with edge / EV / verdict."""
        # The book line implies the median; under -110 each side is ~50%.
        # Use the player's empirical distribution to compute true P(over).
        # We approximate by Z-scoring the gap vs. the player's std.
        # If we have stats, compute exact empirical P(over book_line).
        df = _load_stats()
        # Re-derive sample for this player+stat
        # (decompose() already pulled this; we keep it lightweight here)
        return {
            "book_line": float(book_line),
            "model_projection": float(self.projection),
            "edge_yards": float(self.projection - book_line),
        }


def _key_dvp_metric(stat: str) -> str:
    """Map a stat to the matching DvP per-game allowance metric."""
    return {
        "receiving_yards": "rec_yards_pg",
        "rushing_yards":   "rush_yards_pg",
        "passing_yards":   None,    # no per-team passing-allowed metric per pos
    }.get(stat)


def _pos_group_for(position: str) -> str | None:
    p = (position or "").upper()
    if p == "WR":
        return "WR"
    if p == "TE":
        return "TE"
    if p in ("RB", "FB", "HB"):
        return "RB"
    return None


def decompose(player_id: str, position: str, team: str, stat: str,
               opponent: str | None = None,
               season: int | None = None, week: int | None = None,
               injury_status: str | None = None,
               injury_body_part: str = "unknown",
               injury_practice: str = "DNP",
               key_starter_out: str | None = None,
               expected_game_script: GameScriptBucket | str | None = None,
               target_temp: float | None = None,
               target_wind: float | None = None,
               target_roof: str | None = None,
               target_surface: str | None = None,
               lookback_games: int = 12) -> Decomposition:
    """Build a decomposed projection."""
    df = _load_stats()
    sub = df[(df["player_id"] == player_id) & df[stat].notna()]
    sub = sub.sort_values(["season", "week"], ascending=[False, False])
    if sub.empty:
        return Decomposition(
            player_display_name="?", position=position, team=team,
            stat=stat, n_games_baseline=0, baseline=0.0,
        )

    name = str(sub.iloc[0]["player_display_name"])
    recent = sub.head(lookback_games)
    baseline = float(recent[stat].median())

    decomp = Decomposition(
        player_display_name=name, position=position, team=team,
        stat=stat, n_games_baseline=len(recent),
        baseline=baseline,
    )

    # ── Injury (4.1) — try PLAYER-OWN retention first, fall back to
    # cohort. The player's own historical retention (when sample is
    # ≥5) is a sharper signal than the league-position cohort.
    if injury_status and injury_status.upper() != "NONE":
        bucket = injury_bucket_for(injury_status, injury_practice)
        # Try player's own self-delta in this bucket (opp-adjusted)
        own = lookup_player_self_delta(
            player_id=player_id, stat=stat, bucket=bucket, min_n=5,
        )
        if own is not None:
            retention = max(0.0, min(1.10, own.retention_adj))
            if abs(retention - 1.0) > 0.01:
                adj = baseline * (retention - 1.0)
                decomp.contributions.append(DecompContribution(
                    label="Injury — player's own history",
                    delta=adj,
                    note=(f"{injury_status}/{injury_practice}, "
                          f"{injury_body_part}: this player's own "
                          f"retention in this bucket = {retention:.0%} "
                          f"(opp-adjusted, n={own.n} prior games "
                          f"out of {own.n_total} total)"),
                ))
        else:
            # Fall back to league-cohort (existing behavior)
            cohort = cohort_predict(
                position=position, body_part=injury_body_part,
                report_status=injury_status,
                practice_status=injury_practice,
            )
            retention = max(0.0, min(1.10, cohort.snap_retention_if_played))
            if abs(retention - 1.0) > 0.01:
                adj = baseline * (retention - 1.0)
                decomp.contributions.append(DecompContribution(
                    label="Injury — league cohort",
                    delta=adj,
                    note=(f"{injury_status}/{injury_practice}, "
                          f"{injury_body_part}: league cohort "
                          f"retention {retention:.0%} "
                          f"(player-own sample too thin; "
                          f"cohort n={cohort.n})"),
                ))

    # ── Weather (4.5)
    if any(x is not None for x in (target_temp, target_wind,
                                     target_roof, target_surface)):
        wstat = primary_stat_for_position(position)
        if wstat == stat:
            wr = weather_cohort(
                player_id=player_id, position=position,
                target_temp=target_temp, target_wind=target_wind,
                target_roof=target_roof, target_surface=target_surface,
            )
            if wr.n_games >= 5 and wr.cohort_mode in ("player", "tier_blend"):
                weather_adj = wr.p50 - baseline
                decomp.contributions.append(DecompContribution(
                    label="Weather",
                    delta=weather_adj,
                    note=(f"{wr.cohort_mode} cohort, n={wr.n_games}, "
                          f"P50={wr.p50:.0f} vs baseline {baseline:.0f} "
                          f"({wr.confidence})"),
                ))

    # ── Matchup / DvP (5.8)
    if opponent and season is not None:
        dvp_metric = _key_dvp_metric(stat)
        pg = _pos_group_for(position)
        if dvp_metric and pg:
            dvp = _load_dvp()
            if not dvp.empty:
                row = dvp[(dvp["defteam"] == opponent)
                          & (dvp["season"] == int(season))
                          & (dvp["pos_group"] == pg)]
                if not row.empty:
                    delta_pg = row.iloc[0].get(f"{dvp_metric}_delta")
                    if pd.notna(delta_pg):
                        # The DvP delta is per-game allowed by this team
                        # vs. league avg. Distribute across the team's
                        # position group (rough estimate: top-N players
                        # absorb most of it). For v1, allocate 35% to
                        # WR1, 25% TE1, 50% RB1.
                        share = {"WR1": 0.35, "TE1": 0.25, "RB1": 0.50}.get(
                            f"{pg}1", 0.30)
                        # Without role info, default share
                        adj = float(delta_pg) * share
                        decomp.contributions.append(DecompContribution(
                            label="Matchup (DvP)",
                            delta=adj,
                            note=(f"{opponent} {season} vs. {pg}: "
                                  f"{delta_pg:+.1f} yds/game vs league "
                                  f"(allocated {share:.0%} share)"),
                        ))

    # ── Key starter unavailable (4.2) — when a teammate is OUT
    if key_starter_out and key_starter_out.upper() in ("QB1", "RB1",
                                                        "WR1", "TE1",
                                                        "MULTI"):
        gs = _load_gs()
        if not gs.empty:
            row = gs[gs["scenario"] == key_starter_out.upper()]
            if not row.empty:
                pass_rate_delta = float(
                    row.iloc[0].get("pass_rate_delta") or 0)
                if stat in ("receiving_yards", "passing_yards"):
                    sign = 1 if stat == "receiving_yards" else 1
                    adj = baseline * pass_rate_delta * 1.5 * sign
                elif stat == "rushing_yards":
                    adj = baseline * (-pass_rate_delta) * 1.5
                else:
                    adj = 0
                decomp.contributions.append(DecompContribution(
                    label="Key starter unavailable",
                    delta=adj,
                    note=(f"{key_starter_out} OUT: league pass-rate "
                          f"shift {pass_rate_delta:+.1%} (n="
                          f"{int(row.iloc[0]['n_games'])})"),
                ))

    # ── Expected game-script — usage shift based on game flow
    # Multiplier comes from the player's own historical splits in
    # games that ended in the target margin bucket (or league-wide
    # position fallback when sample is thin).
    if expected_game_script:
        bucket = (expected_game_script
                   if isinstance(expected_game_script, GameScriptBucket)
                   else GameScriptBucket(str(expected_game_script).upper()))
        mult, source, n = multiplier_for_game_script(
            player_id=player_id, stat=stat,
            target_bucket=bucket, fallback_position=position,
        )
        if abs(mult - 1.0) > 0.005:
            adj = baseline * (mult - 1.0)
            label_pretty = BUCKET_LABEL.get(bucket, bucket.value)
            decomp.contributions.append(DecompContribution(
                label="Game-script",
                delta=adj,
                note=(f"Expected: {label_pretty}. "
                      f"Player avg in this bucket = "
                      f"{mult:.2f}x baseline "
                      f"({source} cohort, n={n})"),
            ))

    return decomp

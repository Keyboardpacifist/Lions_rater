"""Edge Finder — scans the slate and surfaces mispriced prop bets.

For a given (season, week), runs every prop-relevant player through
the existing engines and returns a ranked DataFrame of findings —
scenarios where books TYPICALLY anchor to wrong baselines (season
averages instead of recent role, prior-injury baseline instead of
healed, etc.).

Two finding types in v1:

  TREND_DIVERGENCE — recent (last 3) usage z-scored vs season prior
                       crosses |z| ≥ 1.0. Books usually anchor to
                       season averages and lag recent role shifts.

  PROJECTION_GAP   — decomposed projection (with opponent / weather /
                       injury / game-script context) deviates ≥15%
                       from the player's own recent baseline. Books
                       price off the baseline; the model factors in
                       the matchup-specific context.

Each finding gets a plain-English blurb, a magnitude, and a 1-5 star
confidence rating based on signal strength × sample size.

Public entry point
------------------
    generate_edge_findings(season, week, position_filter=None,
                             min_z=1.0, min_pct_shift=0.15) -> pd.DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from lib_decomposed_projection import decompose
from lib_trend_divergence import compute_player_window
from lib_weather import primary_stat_for_position


REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
PLAYER_STATS = DATA / "nfl_player_stats_weekly.parquet"
SCHEDULES = DATA / "nfl_schedules.parquet"


@st.cache_data(show_spinner=False)
def _load_stats() -> pd.DataFrame:
    return pd.read_parquet(PLAYER_STATS)


@st.cache_data(show_spinner=False)
def _load_schedules() -> pd.DataFrame:
    return pd.read_parquet(SCHEDULES)


@dataclass
class Finding:
    player_id: str
    player_name: str
    team: str
    position: str
    stat: str
    finding_type: str          # TREND_DIVERGENCE / PROJECTION_GAP
    direction: str             # "OVER" / "UNDER"
    magnitude: float           # raw effect size (yards, target shifts, etc.)
    pct_shift: float           # signed % vs baseline (for ranking)
    confidence: int            # 1-5 stars
    blurb: str                 # plain-English one-liner


# Stats we hunt for divergence on. Usage stats lead the most edges.
TREND_STATS = [
    "targets", "receptions", "carries",
    "passing_yards", "rushing_yards", "receiving_yards",
]

# Minimum BASELINE values to be a "real" prop-market player.
# Below these, the stat's market doesn't really exist (no book lines)
# AND percentage shifts blow up (a 0→4 yard projection is "+infinity"
# percent but means nothing). Keyed on the player's PRIMARY stat.
MIN_BASELINE_BY_POSITION = {
    "QB":  150.0,    # ≥ 150 passing yds/g recent baseline → starter
    "RB":   30.0,    # ≥ 30 rushing yds/g → real RB1/RB2
    "FB":   30.0,
    "WR":   25.0,    # ≥ 25 receiving yds/g → meaningful target share
    "TE":   18.0,    # ≥ 18 receiving yds/g
}

# Same idea for trend divergence — only flag a role expansion that's
# headed into prop-market territory (not a player going from 0.3 to
# 0.7 carries — irrelevant even if z=2)
MIN_RECENT_AVG_FOR_TREND = {
    "passing_yards":   150.0,
    "rushing_yards":    25.0,
    "receiving_yards":  20.0,
    "receptions":        3.0,
    "targets":           4.0,
    "carries":           5.0,
}


def _eligible_players_for_week(season: int, week: int) -> pd.DataFrame:
    """Return active prop-relevant players whose team has a game this
    week. Uses the schedule + most-recent-team logic."""
    df = _load_stats()
    sch = _load_schedules()

    # Find teams playing this week
    g = sch[(sch["season"] == int(season))
            & (sch["week"] == int(week))]
    if g.empty:
        return pd.DataFrame()
    teams_this_week = set(g["home_team"].dropna().astype(str)) | set(
        g["away_team"].dropna().astype(str))

    # Players with at least 4 games this season at QB/RB/FB/WR/TE,
    # whose most-recent team is playing this week
    season_data = df[(df["season"] == int(season))
                     & df["position"].isin(["QB", "RB", "FB", "WR", "TE"])]
    if season_data.empty:
        return pd.DataFrame()

    counts = (season_data.groupby(["player_id", "player_display_name",
                                       "position"])
               .size().reset_index().rename(columns={0: "n_games"}))
    counts = counts[counts["n_games"] >= 4]
    most_recent = (season_data.sort_values(["season", "week"],
                                              ascending=[False, False])
                    .drop_duplicates(["player_id"])[
                        ["player_id", "team"]
                    ])
    out = counts.merge(most_recent, on="player_id", how="left")
    out = out[out["team"].isin(teams_this_week)]
    return out.reset_index(drop=True)


def _opp_for(team: str, season: int, week: int) -> str | None:
    sch = _load_schedules()
    g = sch[(sch["season"] == int(season))
            & (sch["week"] == int(week))
            & ((sch["home_team"] == team)
               | (sch["away_team"] == team))]
    if g.empty:
        return None
    row = g.iloc[0]
    return (row["away_team"] if row["home_team"] == team
            else row["home_team"])


def _weather_for(team: str, season: int, week: int) -> dict:
    sch = _load_schedules()
    g = sch[(sch["season"] == int(season))
            & (sch["week"] == int(week))
            & ((sch["home_team"] == team)
               | (sch["away_team"] == team))]
    if g.empty:
        return {}
    row = g.iloc[0]
    return {
        "temp": (float(row["temp"]) if pd.notna(row.get("temp")) else None),
        "wind": (float(row["wind"]) if pd.notna(row.get("wind")) else None),
        "roof": (str(row["roof"]) if pd.notna(row.get("roof")) else None),
    }


def _trend_findings(season: int, week: int,
                      eligible: pd.DataFrame,
                      min_z: float) -> list[Finding]:
    """Batch-compute trend divergence flags via a single groupby.
    Massively faster than calling compute_player_window per-player."""
    out: list[Finding] = []
    df = _load_stats()
    if df.empty:
        return out
    # Pre-filter to BEFORE the target week
    pre = df[(df["season"] < int(season))
             | ((df["season"] == int(season))
                & (df["week"] < int(week)))].copy()
    pre = pre.merge(
        eligible[["player_id", "player_display_name", "position", "team"]],
        on="player_id", how="inner", suffixes=("", "_e"),
    )
    if pre.empty:
        return out
    # Sort by season, week descending so head() = most recent
    pre = pre.sort_values(["player_id", "season", "week"],
                            ascending=[True, False, False])

    # Hardened trend window — was last-3 vs season-prior, raised to
    # last-5 vs at-least-3-prior and z>=1.5 (was 1.0). The audit found
    # n=3 trend signals have empirical R² < 0.05 against next-game
    # outcomes; the books absolutely DO incorporate recent form
    # (the old blurb's claim that "books anchor to season averages"
    # is largely false). Tighter gating reduces noise pollution.
    RECENT_WINDOW = 5
    MIN_PRIOR = 3
    HARDENED_MIN_Z = max(min_z, 1.5)

    for stat in TREND_STATS:
        if stat not in pre.columns:
            continue
        min_avg = MIN_RECENT_AVG_FOR_TREND.get(stat, 0.0)
        for pid, group in pre.groupby("player_id", sort=False):
            sub = group[group[stat].notna()]
            if len(sub) < (RECENT_WINDOW + MIN_PRIOR):
                continue
            recent = sub.head(RECENT_WINDOW)
            earlier = sub.iloc[RECENT_WINDOW:]
            if len(recent) < RECENT_WINDOW or len(earlier) < MIN_PRIOR:
                continue
            recent_avg = float(recent[stat].astype(float).mean())
            season_avg = float(earlier[stat].astype(float).mean())
            # Union-variance z
            all_vals = pd.concat([recent[stat].astype(float),
                                    earlier[stat].astype(float)])
            std = float(all_vals.std(ddof=0))
            if std < 1e-6:
                continue
            delta = recent_avg - season_avg
            z = delta / std
            if abs(z) < HARDENED_MIN_Z:
                continue
            if recent_avg < min_avg and season_avg < min_avg:
                continue
            direction = "OVER" if delta > 0 else "UNDER"
            stat_label = stat.replace("_", " ")
            verb = "expanded" if delta > 0 else "shrunk"
            # Confidence stars now penalize for thin earlier-prior
            # samples (a 5-vs-3 split is shakier than 5-vs-12).
            base_conf = int(round(abs(z)))
            sample_penalty = 0 if len(earlier) >= 8 else 1
            confidence = min(5, max(1, base_conf - sample_penalty))
            first = group.iloc[0]
            blurb = (
                f"{first['player_display_name']} ({first['position']}, "
                f"{first['team']}): {stat_label} {verb} recently — "
                f"{recent_avg:.1f}/g last {len(recent)} vs "
                f"{season_avg:.1f}/g over prior {len(earlier)} games "
                f"(z={z:+.1f}). Recent-form trend — books typically "
                f"price this in; treat as supporting evidence."
            )
            # Cap pct_shift at ±200% — beyond that, percent is
            # misleading because the season_avg denominator is too
            # small (rookie or depth player just emerging). The
            # ABSOLUTE delta is the more meaningful number for
            # display; we use pct just for relative ranking.
            raw_pct = (delta / season_avg if season_avg > 0
                        else (1.0 if delta > 0 else -1.0))
            pct_shift_capped = max(-1.0, min(2.0, raw_pct))
            out.append(Finding(
                player_id=str(pid),
                player_name=str(first["player_display_name"]),
                team=str(first["team"]),
                position=str(first["position"]),
                stat=stat,
                finding_type="TREND_DIVERGENCE",
                direction=direction,
                magnitude=float(delta),
                pct_shift=float(pct_shift_capped),
                confidence=confidence,
                blurb=blurb,
            ))
    return out


def _batch_baselines(eligible: pd.DataFrame, stat: str,
                       season: int, week: int,
                       lookback: int = 12) -> dict[str, float]:
    """Compute the recent-baseline median for `stat` for ALL eligible
    players in a single pass. Returns {player_id: baseline_median}.

    This is ~50x faster than calling _quick_baseline per player —
    one filter + one groupby vs. N filters."""
    df = _load_stats()
    if df.empty or stat not in df.columns:
        return {}
    pids = set(eligible["player_id"].tolist())
    pre = df[(df["player_id"].isin(pids))
             & df[stat].notna()]
    pre = pre[(pre["season"] < int(season))
              | ((pre["season"] == int(season))
                 & (pre["week"] < int(week)))]
    if pre.empty:
        return {}
    pre = pre.sort_values(["player_id", "season", "week"],
                            ascending=[True, False, False])
    # Take last `lookback` games per player
    pre["_rank"] = pre.groupby("player_id").cumcount()
    pre = pre[pre["_rank"] < lookback]
    return (pre.groupby("player_id")[stat].median()
              .astype(float).to_dict())


def _projection_findings(season: int, week: int,
                            eligible: pd.DataFrame,
                            min_pct_shift: float) -> list[Finding]:
    out: list[Finding] = []
    # Pre-compute baselines per (position-stat) in batch — much faster
    # than calling _quick_baseline per player.
    baselines_by_stat: dict[str, dict[str, float]] = {}
    stats_needed = set()
    for pos in eligible["position"].unique():
        s = primary_stat_for_position(pos)
        if s:
            stats_needed.add(s)
    for s in stats_needed:
        baselines_by_stat[s] = _batch_baselines(eligible, s,
                                                    season, week,
                                                    lookback=12)

    for _, p in eligible.iterrows():
        pos = p["position"]
        stat = primary_stat_for_position(pos)
        if not stat:
            continue

        # Lookup pre-computed baseline (fast)
        min_baseline = MIN_BASELINE_BY_POSITION.get(pos, 0.0)
        baseline = baselines_by_stat.get(stat, {}).get(p["player_id"], 0.0)
        if baseline < min_baseline:
            continue

        # Now do the full decompose (slow path)
        opp = _opp_for(p["team"], season, week)
        wx = _weather_for(p["team"], season, week)
        try:
            d = decompose(
                player_id=p["player_id"], position=pos,
                team=p["team"], stat=stat,
                opponent=opp,
                season=int(season), week=int(week),
                target_temp=wx.get("temp"),
                target_wind=wx.get("wind"),
                target_roof=wx.get("roof"),
                lookback_games=12,
            )
        except Exception:
            continue
        if d.baseline <= 0:
            continue
        pct = (d.projection - d.baseline) / d.baseline
        if abs(pct) < min_pct_shift:
            continue
        # Cap at ±100% — larger means model artifact, not real edge
        if abs(pct) > 1.0:
            continue

        # Identify the largest single contributor for the blurb
        if d.contributions:
            top = max(d.contributions, key=lambda c: abs(c.delta))
            top_label = top.label
            top_delta = top.delta
        else:
            top_label = "—"
            top_delta = 0.0

        direction = "UNDER" if pct < 0 else "OVER"
        stat_label = stat.replace("_", " ")
        confidence = min(5, max(1, int(round(abs(pct) / 0.05))))
        verb = "below" if pct < 0 else "above"
        blurb = (
            f"{p['player_display_name']} ({p['position']}, "
            f"{p['team']}): {stat_label} projects {abs(pct):.0%} "
            f"{verb} recent baseline ({d.baseline:.0f} → "
            f"{d.projection:.0f}). Top driver: {top_label} "
            f"({top_delta:+.0f} yds)."
        )
        out.append(Finding(
            player_id=p["player_id"],
            player_name=p["player_display_name"],
            team=p["team"], position=p["position"],
            stat=stat,
            finding_type="PROJECTION_GAP",
            direction=direction,
            magnitude=float(d.projection - d.baseline),
            pct_shift=float(pct),
            confidence=confidence,
            blurb=blurb,
        ))
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def generate_edge_findings(season: int, week: int,
                              position_filter: list[str] | None = None,
                              min_z: float = 1.0,
                              min_pct_shift: float = 0.15
                              ) -> pd.DataFrame:
    """Top-level entry point. Returns a DataFrame of findings ranked
    by confidence × |magnitude|.

    Cached for 1 hour — same (season, week, filter) input returns
    instantly on second call.
    """
    eligible = _eligible_players_for_week(season, week)
    if eligible.empty:
        return pd.DataFrame()
    if position_filter:
        eligible = eligible[eligible["position"].isin(position_filter)]
    if eligible.empty:
        return pd.DataFrame()

    findings: list[Finding] = []
    findings.extend(_trend_findings(season, week, eligible, min_z))
    findings.extend(_projection_findings(season, week, eligible,
                                            min_pct_shift))

    if not findings:
        return pd.DataFrame()
    df = pd.DataFrame([f.__dict__ for f in findings])
    # Rank: confidence first, then magnitude
    df["_score"] = df["confidence"] * df["pct_shift"].abs()
    df = df.sort_values("_score", ascending=False).drop(
        columns="_score").reset_index(drop=True)
    return df


def latest_week_with_games(season: int) -> int:
    """Return the most-recent week in the schedule for a season that
    has actual games scheduled."""
    sch = _load_schedules()
    sub = sch[(sch["season"] == int(season))
              & sch["spread_line"].notna()]
    if sub.empty:
        return 1
    return int(sub["week"].max())

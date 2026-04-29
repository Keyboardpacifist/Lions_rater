"""Game summary modal — the Monday-morning-QB game recap.

Three layers of differentiated content for each game:

1. Win-probability arc with the biggest WP swings annotated
2. Top 5 critical plays (by |wpa|) with full play context
3. Counterfactual coverage analysis on each critical pass:
   "what would Cover 4 / Cover 2 / Cover 0 have produced for this
   matchup?" — answered by aggregating similar plays across the
   league.

All counterfactual baselines come with confidence flags using the
caution-badge pattern. Small-sample baselines get a hover tooltip
explaining the limitation rather than fabricating certainty.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"


# Coverage types we'll evaluate counterfactually
_COVERAGE_TYPES = [
    ("COVER_0", "Cover 0 (all-out blitz)"),
    ("COVER_1", "Cover 1 (man / 1 high)"),
    ("COVER_2", "Cover 2 (2 deep zone)"),
    ("2_MAN",   "Cover 2 Man"),
    ("COVER_3", "Cover 3 (3 deep)"),
    ("COVER_4", "Cover 4 (quarters)"),
    ("COVER_6", "Cover 6 (split)"),
]
_COVERAGE_LABELS = dict(_COVERAGE_TYPES)


@st.cache_data(show_spinner="Loading game data…")
def _load_season_pbp(season: int) -> pd.DataFrame:
    """Pull season pbp via nflreadpy — cached per season."""
    try:
        import nflreadpy as nfl
        return nfl.load_pbp([season]).to_pandas()
    except Exception:
        return pd.DataFrame()


def get_game_summary(team: str, season: int, week: int,
                       game_type: str = "REG") -> dict | None:
    """Return the summary payload for the specified (team, season, week)."""
    pbp = _load_season_pbp(season)
    if pbp.empty:
        return None
    # Filter to the right game
    game_filter = (
        (pbp["season"] == season) & (pbp["week"] == week)
        & ((pbp["home_team"] == team) | (pbp["away_team"] == team))
    )
    if game_type and game_type != "any":
        game_filter = game_filter & (pbp["season_type"] == game_type)
    plays = pbp[game_filter].copy()
    if plays.empty:
        return None

    is_home = plays["home_team"].iloc[0] == team
    opp = plays["away_team"].iloc[0] if is_home else plays["home_team"].iloc[0]

    # Team's WP per play (from team's perspective, not posteam's)
    plays["team_wp"] = plays["home_wp"] if is_home else plays["away_wp"]
    # WPA from team's perspective: positive when team's WP went up
    if "wpa" in plays.columns:
        # wpa is from posteam perspective — need to flip when team isn't posteam
        plays["team_wpa"] = plays.apply(
            lambda r: r["wpa"] if pd.notna(r.get("wpa"))
                          and r.get("posteam") == team
                       else (-r["wpa"] if pd.notna(r.get("wpa")) else 0),
            axis=1,
        )

    # Game metadata
    final_team_score = (plays["home_score"].iloc[-1] if is_home
                          else plays["away_score"].iloc[-1])
    final_opp_score = (plays["away_score"].iloc[-1] if is_home
                         else plays["home_score"].iloc[-1])
    if pd.isna(final_team_score):
        # Fall back to drives ending — use total_home_score / total_away_score
        # from last play if available
        last = plays.iloc[-1]
        final_team_score = (last.get("total_home_score") if is_home
                              else last.get("total_away_score"))
        final_opp_score = (last.get("total_away_score") if is_home
                             else last.get("total_home_score"))

    # WP arc data — keep one row per real play (skip kickoffs / no-plays
    # to keep the chart less noisy, but include them so the line is continuous)
    arc_plays = plays[plays["game_seconds_remaining"].notna()].copy()
    arc_plays = arc_plays.sort_values("game_seconds_remaining", ascending=False)
    arc = arc_plays[["game_seconds_remaining", "team_wp", "qtr",
                       "desc", "team_wpa"]].copy() if "team_wpa" in arc_plays.columns \
        else arc_plays[["game_seconds_remaining", "team_wp", "qtr", "desc"]].copy()
    arc = arc.dropna(subset=["team_wp"])

    # Critical plays — top 5 by |team_wpa|
    if "team_wpa" in plays.columns:
        critical = (
            plays[plays["team_wpa"].abs() > 0.03]
            .nlargest(5, "team_wpa", keep="all")
            ._append(plays[plays["team_wpa"].abs() > 0.03]
                       .nsmallest(5, "team_wpa", keep="all"))
        )
        critical = (
            critical.drop_duplicates(subset=["play_id"])
            .assign(_abs_wpa=lambda d: d["team_wpa"].abs())
            .nlargest(5, "_abs_wpa")
            .sort_values("game_seconds_remaining", ascending=False)
        )
    else:
        critical = pd.DataFrame()

    return {
        "team": team,
        "opp": opp,
        "is_home": is_home,
        "season": season,
        "week": week,
        "final_team_score": final_team_score,
        "final_opp_score": final_opp_score,
        "n_plays": len(plays),
        "wp_arc": arc,
        "critical_plays": critical,
        "all_plays": plays,
    }


def render_wp_arc(summary: dict) -> None:
    """Plot win probability over the course of the game."""
    arc = summary["wp_arc"]
    if arc.empty:
        return
    team = summary["team"]
    opp = summary["opp"]

    # Convert game_seconds_remaining (3600 → 0) to elapsed minutes (0 → 60)
    arc = arc.copy()
    arc["minutes_elapsed"] = (3600 - arc["game_seconds_remaining"]) / 60

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=arc["minutes_elapsed"],
        y=arc["team_wp"] * 100,
        mode="lines",
        line=dict(color="#1F77B4", width=2.5),
        name=f"{team} WP",
        hovertemplate="<b>Q%{customdata[0]} · %{x:.1f} min in</b><br>"
                      "WP: %{y:.0f}%<br>%{customdata[1]}<extra></extra>",
        customdata=arc[["qtr", "desc"]].values,
    ))
    # 50% reference line
    fig.add_hline(y=50, line_color="#888", line_dash="dash", line_width=1)
    # Quarter markers
    for q_end in (15, 30, 45):
        fig.add_vline(x=q_end, line_color="#ddd", line_dash="dot",
                       line_width=1)
    fig.update_layout(
        title=f"📈 Win probability — {team} vs {opp}",
        xaxis_title="Minutes elapsed",
        yaxis_title=f"{team} win probability (%)",
        yaxis=dict(range=[0, 100]),
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Counterfactual coverage analysis ────────────────────────────
@st.cache_data(show_spinner=False)
def _coverage_baseline_pool() -> pd.DataFrame:
    """All league dropbacks with full participation context — used to
    compute counterfactual coverage baselines."""
    try:
        import polars as pl
        return pl.read_parquet(_DATA / "qb_dropbacks.parquet").to_pandas()
    except Exception:
        return pd.DataFrame()


def _coverage_counterfactual(play_row: pd.Series) -> dict | None:
    """For a critical pass play, compute the expected EPA against
    each major coverage given the matchup (depth + location).

    Returns: {coverage_code: {"epa": float, "n": int, "confidence": str}}
    """
    pool = _coverage_baseline_pool()
    if pool.empty:
        return None
    # Slice the pool to "similar" plays: same pass length + location +
    # ydstogo bucket. Needs to be loose enough to get meaningful samples.
    air_yards = play_row.get("air_yards")
    pass_loc = play_row.get("pass_location")
    if pd.isna(air_yards) or pd.isna(pass_loc):
        return None
    # Depth bucket
    if air_yards >= 20:
        depth_band = (15, 99)
    elif air_yards >= 10:
        depth_band = (8, 22)
    else:
        depth_band = (-5, 12)
    # Down + distance bucket (rough)
    down = play_row.get("down")
    ydstogo = play_row.get("ydstogo")

    similar = pool[
        (pool["air_yards"] >= depth_band[0])
        & (pool["air_yards"] < depth_band[1])
        & (pool["pass_location"] == pass_loc)
        & (pool["pass_attempt"] == 1)
    ].copy()
    # Loosely match down/distance — same down + within 2 yds of distance
    if pd.notna(down) and pd.notna(ydstogo):
        similar = similar[
            (similar["down"] == down)
            & (similar["ydstogo"].between(max(1, ydstogo - 3), ydstogo + 3))
        ]

    out = {}
    for cov_code, _ in _COVERAGE_TYPES:
        cov_subset = similar[similar["defense_coverage_type"] == cov_code]
        if cov_subset.empty:
            continue
        n = len(cov_subset)
        epa = float(cov_subset["epa"].mean())
        if n >= 30:
            conf = "HIGH"
        elif n >= 10:
            conf = "MEDIUM"
        else:
            conf = "LOW"
        out[cov_code] = {"epa": epa, "n": n, "confidence": conf}
    return out


def _confidence_color(conf: str) -> str:
    return {"HIGH": "#34A853", "MEDIUM": "#E67E22", "LOW": "#888"}[conf]


def _render_critical_play(idx: int, play: pd.Series, team: str) -> None:
    """One expandable card per critical play with counterfactual analysis."""
    qtr = int(play.get("qtr", 0)) if pd.notna(play.get("qtr")) else "—"
    qsr = int(play.get("quarter_seconds_remaining", 0)) if pd.notna(play.get("quarter_seconds_remaining")) else 0
    qsr_min = qsr // 60
    qsr_sec = qsr % 60
    wpa = play.get("team_wpa", play.get("wpa", 0))
    wpa_pct = wpa * 100 if pd.notna(wpa) else 0
    desc = str(play.get("desc", ""))[:200]
    sign = "▲" if wpa_pct > 0 else "▼"
    label = (
        f"#{idx} — Q{qtr} {qsr_min}:{qsr_sec:02d} · "
        f"{sign} {abs(wpa_pct):.1f}% WP swing"
    )
    with st.expander(label, expanded=(idx <= 2)):
        st.markdown(f"**Play:** {desc}")
        # Quick context row
        cov = play.get("defense_coverage_type")
        rushers = play.get("number_of_pass_rushers")
        personnel = play.get("offense_personnel")
        formation = play.get("offense_formation")
        ctx = []
        if cov: ctx.append(f"**Coverage:** {_COVERAGE_LABELS.get(cov, cov)}")
        if pd.notna(rushers): ctx.append(f"**Rushers:** {int(rushers)}")
        if formation: ctx.append(f"**Formation:** {formation}")
        if personnel: ctx.append(f"**Personnel:** {personnel}")
        if ctx:
            st.caption(" · ".join(ctx))

        # Counterfactual analysis (only if it was a pass attempt)
        if play.get("pass_attempt") == 1 and pd.notna(play.get("air_yards")):
            cf = _coverage_counterfactual(play)
            if cf:
                st.markdown("**🤔 What if the defense had called something else?**")
                st.caption(
                    "Expected EPA per attempt for similar matchups "
                    "(same pass depth/location + similar down & distance) "
                    "across the league. Confidence flag reflects sample size."
                )
                # Sort coverages by expected EPA (best for defense first = lowest EPA)
                rows = sorted(cf.items(), key=lambda kv: kv[1]["epa"])
                for cov_code, info in rows:
                    is_actual = cov_code == cov
                    badge = (
                        '<span style="font-size:10px;background:rgba(31,119,180,0.15);'
                        'color:#1F77B4;padding:2px 6px;border-radius:4px;'
                        'font-weight:700;margin-left:6px;">ACTUAL</span>'
                        if is_actual else ""
                    )
                    conf_color = _confidence_color(info["confidence"])
                    conf_tip = (
                        f"Sample: {info['n']} comparable plays. "
                        f"≥30 = HIGH · 10–29 = MEDIUM · <10 = LOW. "
                        + ("Strong baseline." if info["confidence"] == "HIGH"
                           else "Treat as directional, not definitive."
                           if info["confidence"] == "MEDIUM"
                           else "Small sample — interpret cautiously.")
                    )
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:6px 10px;background:rgba(0,0,0,0.04);'
                        f'border-radius:6px;margin-bottom:4px;font-size:13px;">'
                        f'<div>{_COVERAGE_LABELS.get(cov_code, cov_code)}{badge}</div>'
                        f'<div>'
                        f'<span style="font-weight:700;">{info["epa"]:+.2f} EPA</span>'
                        f'<span title="{conf_tip}" '
                        f'style="font-size:10px;color:white;background:{conf_color};'
                        f'padding:2px 6px;border-radius:4px;margin-left:8px;'
                        f'cursor:help;font-weight:700;">'
                        f'{info["confidence"]} · n={info["n"]}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                # Auto-narrative: pick the optimal counterfactual
                actual_info = cf.get(cov, {}) if cov in cf else None
                if actual_info and len(cf) >= 2:
                    best_cov, best_info = min(cf.items(), key=lambda kv: kv[1]["epa"])
                    if best_cov != cov and (actual_info["epa"]
                                              - best_info["epa"]) > 0.10:
                        delta = actual_info["epa"] - best_info["epa"]
                        st.info(
                            f"**Monday-morning QB:** "
                            f"{_COVERAGE_LABELS.get(best_cov, best_cov)} would "
                            f"have been the optimal call against this matchup "
                            f"(expected {best_info['epa']:+.2f} EPA vs the "
                            f"{actual_info['epa']:+.2f} they gave up in "
                            f"{_COVERAGE_LABELS.get(cov, cov)}). Difference: "
                            f"{delta:+.2f} EPA."
                        )


def render_game_summary(team: str, season: int, week: int,
                          game_type: str = "REG") -> None:
    """Top-level rendering for the game summary modal."""
    summary = get_game_summary(team, season, week, game_type)
    if not summary:
        st.warning(f"No game data for {team} {season} week {week}.")
        return

    opp = summary["opp"]
    ts = summary["final_team_score"]
    os_ = summary["final_opp_score"]
    venue = "vs" if summary["is_home"] else "at"
    result = "W" if (pd.notna(ts) and pd.notna(os_) and ts > os_) else \
             "L" if (pd.notna(ts) and pd.notna(os_) and ts < os_) else "T"
    st.markdown(
        f"### Game summary — Wk {week}: {team} {venue} {opp} "
        f"({int(ts) if pd.notna(ts) else '—'}–"
        f"{int(os_) if pd.notna(os_) else '—'}, **{result}**)"
    )

    render_wp_arc(summary)

    # Critical plays
    crit = summary["critical_plays"]
    if not crit.empty:
        st.markdown("### 🎯 Critical plays")
        st.caption(
            "Plays sorted by win-probability swing. Counterfactual "
            "coverage analysis on each pass — what other coverage calls "
            "would have produced against this matchup."
        )
        for i, (_, p) in enumerate(crit.iterrows(), start=1):
            _render_critical_play(i, p, team)

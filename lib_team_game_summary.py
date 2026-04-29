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


# ── Offensive concept buckets (the offensive counterfactual) ────
# Given the defense called coverage X, what offensive concept would
# have been the best response? "Concept" here is intentionally
# slightly abstract — not "play call" but a small set of buckets fans
# can reason about.
_OFFENSIVE_CONCEPTS = [
    # (concept_id, label, filter_func, identity_blurb)
    ("run_inside", "Run inside (between the tackles)",
     lambda d: (d["play_type_run"] == 1) & (d["run_location_norm"] == "middle"),
     "Inside zone / power runs through the A-B gaps."),
    ("run_outside_left", "Run outside left",
     lambda d: (d["play_type_run"] == 1) & (d["run_location_norm"] == "left"),
     "Outside zone, sweeps, or end runs to the left side."),
    ("run_outside_right", "Run outside right",
     lambda d: (d["play_type_run"] == 1) & (d["run_location_norm"] == "right"),
     "Outside zone, sweeps, or end runs to the right side."),
    ("pass_short_left", "Short pass left",
     lambda d: (d["play_type_pass"] == 1) & (d["air_yards"] < 8) & (d["pass_location"] == "left"),
     "Quick game / screens / hitches to the left."),
    ("pass_short_middle", "Short pass middle",
     lambda d: (d["play_type_pass"] == 1) & (d["air_yards"] < 8) & (d["pass_location"] == "middle"),
     "Slants, RB checkdowns, drag routes."),
    ("pass_short_right", "Short pass right",
     lambda d: (d["play_type_pass"] == 1) & (d["air_yards"] < 8) & (d["pass_location"] == "right"),
     "Quick game / screens / hitches to the right."),
    ("pass_intermediate_left", "Intermediate pass left",
     lambda d: (d["play_type_pass"] == 1) & (d["air_yards"] >= 8) & (d["air_yards"] < 18) & (d["pass_location"] == "left"),
     "Outs, curls, intermediate digs to the left side."),
    ("pass_intermediate_middle", "Intermediate pass middle",
     lambda d: (d["play_type_pass"] == 1) & (d["air_yards"] >= 8) & (d["air_yards"] < 18) & (d["pass_location"] == "middle"),
     "Digs, in-routes, seam shots into the soft middle.",),
    ("pass_intermediate_right", "Intermediate pass right",
     lambda d: (d["play_type_pass"] == 1) & (d["air_yards"] >= 8) & (d["air_yards"] < 18) & (d["pass_location"] == "right"),
     "Outs, curls, intermediate digs to the right side."),
    ("pass_deep_left", "Deep pass left",
     lambda d: (d["play_type_pass"] == 1) & (d["air_yards"] >= 18) & (d["pass_location"] == "left"),
     "Vertical shots / posts / fades to the left sideline."),
    ("pass_deep_middle", "Deep pass middle",
     lambda d: (d["play_type_pass"] == 1) & (d["air_yards"] >= 18) & (d["pass_location"] == "middle"),
     "Posts and seams attacking the middle of the field deep."),
    ("pass_deep_right", "Deep pass right",
     lambda d: (d["play_type_pass"] == 1) & (d["air_yards"] >= 18) & (d["pass_location"] == "right"),
     "Vertical shots / posts / fades to the right sideline."),
]


@st.cache_data(show_spinner=False)
def _full_play_pool() -> pd.DataFrame:
    """Combined per-play view (runs + dropbacks) tagged with helper
    columns for the offensive counterfactual concept-bucketing."""
    try:
        from lib_splits import _load_rusher_plays
    except Exception:
        return pd.DataFrame()
    drops_path = _DATA / "qb_dropbacks.parquet"
    runs = _load_rusher_plays()
    if runs is None or runs.empty:
        runs = pd.DataFrame()
    else:
        runs = runs.copy()
        runs["play_type_run"] = 1
        runs["play_type_pass"] = 0
        # Normalize team col
        if "team" in runs.columns and "posteam" not in runs.columns:
            runs["posteam"] = runs["team"]
        if "opponent_team" in runs.columns and "defteam" not in runs.columns:
            runs["defteam"] = runs["opponent_team"]
        runs["run_location_norm"] = runs.get("run_location")
    try:
        import polars as pl
        drops = pl.read_parquet(drops_path).to_pandas()
    except Exception:
        drops = pd.DataFrame()
    if not drops.empty:
        drops = drops.copy()
        drops["play_type_run"] = 0
        drops["play_type_pass"] = drops.get("pass_attempt", 0).fillna(0).astype(int)
        drops["run_location_norm"] = None

    # Common columns
    cols = ["game_id", "play_id", "season", "week", "posteam", "defteam",
            "down", "ydstogo", "epa", "play_type_run", "play_type_pass",
            "run_location_norm", "air_yards", "pass_location",
            "defense_coverage_type", "defense_man_zone_type",
            "number_of_pass_rushers"]
    # Some columns may be missing in one source — pad with NaN
    for df in (runs, drops):
        for c in cols:
            if c not in df.columns:
                df[c] = None
    runs_t = runs[cols] if not runs.empty else pd.DataFrame(columns=cols)
    drops_t = drops[cols] if not drops.empty else pd.DataFrame(columns=cols)
    return pd.concat([runs_t, drops_t], ignore_index=True, sort=False)


def _offensive_counterfactual(play_row: pd.Series) -> dict | None:
    """Given the defense called coverage X, what's the expected EPA
    by offensive concept? Returns {concept_id: {"epa","n","confidence","label","blurb"}}.
    """
    pool = _full_play_pool()
    if pool.empty:
        return None
    cov = play_row.get("defense_coverage_type")
    if not cov:
        return None
    down = play_row.get("down")
    ydstogo = play_row.get("ydstogo")
    similar = pool[pool["defense_coverage_type"] == cov].copy()
    if pd.notna(down) and pd.notna(ydstogo):
        similar = similar[
            (similar["down"] == down)
            & (similar["ydstogo"].between(max(1, ydstogo - 3), ydstogo + 3))
        ]
    if similar.empty:
        return None

    out = {}
    for cid, label, filter_fn, blurb in _OFFENSIVE_CONCEPTS:
        try:
            sub = similar[filter_fn(similar)]
        except Exception:
            continue
        sub = sub.dropna(subset=["epa"])
        if sub.empty:
            continue
        n = len(sub)
        epa = float(sub["epa"].mean())
        if n >= 30:
            conf = "HIGH"
        elif n >= 10:
            conf = "MEDIUM"
        else:
            conf = "LOW"
        out[cid] = {
            "label": label, "blurb": blurb,
            "epa": epa, "n": n, "confidence": conf,
        }
    return out


def _classify_actual_offensive_concept(play_row: pd.Series) -> str | None:
    """Given the actual play, identify which concept bucket it falls
    into (used to badge the row that was actually called)."""
    # Run plays
    rl = play_row.get("run_location")
    if pd.notna(rl) and rl in ("left", "middle", "right"):
        if rl == "middle":
            return "run_inside"
        if rl == "left":
            return "run_outside_left"
        if rl == "right":
            return "run_outside_right"
    # Pass plays
    if play_row.get("pass_attempt") == 1:
        ay = play_row.get("air_yards")
        loc = play_row.get("pass_location")
        if pd.notna(ay) and pd.notna(loc) and loc in ("left", "middle", "right"):
            if ay < 8:
                depth = "short"
            elif ay < 18:
                depth = "intermediate"
            else:
                depth = "deep"
            return f"pass_{depth}_{loc}"
    return None


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


# ── Coverage narrative knowledge base ───────────────────────────
# Each coverage gets a base description + depth-specific notes.
# Used to generate 1-2 sentence "what would have helped/hurt and why"
# text for every counterfactual row.
_COVERAGE_TRAITS = {
    "COVER_0": {
        "ident":    "All-out blitz, no safety help over the top.",
        "deep":     "Vulnerable to any contested deep ball — one missed tackle or beaten cornerback becomes a touchdown.",
        "intermediate": "Designed to win on the rush before the route develops; if the QB gets it out clean, the underneath defenders are isolated.",
        "short":    "The quick pass usually beats it — receivers sit behind their man with the safety vacated.",
    },
    "COVER_1": {
        "ident":    "Single-high safety, man across the rest of the field.",
        "deep":     "The lone deep safety has to range to help — favors the offense if the matched WR creates separation.",
        "intermediate": "Sticky if the matchups hold but exposed by rub routes and pick concepts.",
        "short":    "Man coverage on quick routes — a step of separation is all the QB needs.",
    },
    "COVER_2": {
        "ident":    "Two-deep zone with five underneath. Soft middle, hard sideline.",
        "deep":     "Strong against vertical routes split between the safeties; the seam is the soft spot.",
        "intermediate": "Vulnerable to the soft middle hole between linebackers and safeties — the known weakness.",
        "short":    "Underneath defenders flow hard; quick outs and flats can dent it.",
    },
    "2_MAN": {
        "ident":    "Two-deep safeties with man underneath — designed against rub routes and shallow crossers.",
        "deep":     "Mostly equivalent to Cover 2 deep; favors the defense if the safeties stay disciplined.",
        "intermediate": "Strong against in-breakers and crossers; the trade-off is no underneath help on scrambles.",
        "short":    "Tight man coverage on quick routes — separation is the only crack.",
    },
    "COVER_3": {
        "ident":    "Three-deep zone covering the field in thirds, four under.",
        "deep":     "Strong against pure verticals down the sideline. The seam between deep-third and curl/flat is the wrinkle.",
        "intermediate": "Vulnerable to dig routes attacking the middle hole behind the linebackers.",
        "short":    "Underneath defenders read the QB's eyes and flow to the ball — flats can find a window.",
    },
    "COVER_4": {
        "ident":    "Four-deep quarters, pattern-matching underneath. The best deep coverage available.",
        "deep":     "Maxes out deep help — usually the safest call against vertical concepts.",
        "intermediate": "Route combinations that pull the safeties forward (smash, flood) can crack it.",
        "short":    "Outside corners often play soft — quick screens and slants succeed if the OL holds.",
    },
    "COVER_6": {
        "ident":    "Half-field combo — Cover 4 to the boundary, Cover 2 to the field. Designed to be matchup-aware by side.",
        "deep":     "Optimized for splitting deep responsibilities by formation strength.",
        "intermediate": "The boundary side is denser; the field side is more vulnerable to dig/cross combos.",
        "short":    "Underneath structure resembles Cover 2 — flats and quick hitches the soft spot.",
    },
    "COVER_9": {
        "ident":    "A pattern-match variant — usually a Cover 1/3 hybrid based on offensive alignment.",
        "deep":     "Reads the route distribution post-snap; effective when execution is clean.",
        "intermediate": "Exposed when the receivers run combination routes that confuse the matching rules.",
        "short":    "Quick game generally beats it before the match develops.",
    },
    "COMBO": {
        "ident":    "Combination coverage — different rules for different sides of the field.",
        "deep":     "Effective in disguise; depends on which half is responsible for what.",
        "intermediate": "Combo calls are often a defensive coordinator's response to specific route combos they expect.",
        "short":    "Quick passes generally exploit whichever side has zone underneath.",
    },
}


def _depth_band(air_yards: float) -> str:
    if pd.isna(air_yards):
        return "intermediate"
    if air_yards >= 18:
        return "deep"
    if air_yards >= 8:
        return "intermediate"
    return "short"


def _coverage_blurb(cov_code: str, play_row: pd.Series,
                      counterfactual_epa: float,
                      actual_epa: float | None,
                      is_actual: bool) -> str:
    """One or two sentences explaining the coverage's role + how it
    compared to the actual call. Adapts to play depth and to whether
    this coverage would have helped or hurt."""
    traits = _COVERAGE_TRAITS.get(cov_code)
    if not traits:
        return ""
    depth = _depth_band(play_row.get("air_yards", float("nan")))
    depth_note = traits.get(depth, "")
    base = traits["ident"]

    if is_actual:
        # Closing the loop on what actually happened
        return f"_{base} {depth_note}_"

    if actual_epa is None:
        return f"_{base} {depth_note}_"

    diff = counterfactual_epa - actual_epa
    if diff <= -0.20:
        verdict = " **Would have been a notably better call** — saves "\
                  f"~{abs(diff):.2f} EPA on this matchup."
    elif diff <= -0.05:
        verdict = " Would have been a marginally better call here."
    elif diff >= 0.20:
        verdict = " **Would have been worse** — gives up "\
                  f"~{diff:.2f} more EPA than what was actually played."
    elif diff >= 0.05:
        verdict = " Would have been a marginally worse call here."
    else:
        verdict = " Roughly equivalent to what was actually called."

    return f"_{base} {depth_note}_{verdict}"


def _render_offensive_counterfactual(play: pd.Series, cov: str | None) -> None:
    """Offense-view counterfactual: given the defense called X, what
    offensive concepts would have been better than what was actually
    called? Backed by the same league-wide pool with confidence flags.
    """
    cf = _offensive_counterfactual(play)
    if not cf or not cov:
        st.caption("_Not enough comparable data for offensive counterfactual analysis._")
        return
    actual_concept = _classify_actual_offensive_concept(play)
    actual_info = cf.get(actual_concept) if actual_concept else None
    actual_epa = actual_info["epa"] if actual_info else None

    st.markdown(
        f"**🤔 Given the defense called "
        f"{_COVERAGE_LABELS.get(cov, cov)}, what offensive concept "
        f"would have been the best response?**"
    )
    st.caption(
        "Expected **offensive** EPA per play across the league when "
        "facing this same coverage on a comparable down/distance. "
        "**Higher EPA = better for the offense.** The 🟢 **optimal** "
        "concept is highlighted. Confidence flag reflects sample size."
    )

    # Sort by EPA descending — best offensive concept first
    rows = sorted(cf.items(), key=lambda kv: kv[1]["epa"], reverse=True)
    eligible = [(k, v) for k, v in rows
                if k != actual_concept
                and v["confidence"] in ("HIGH", "MEDIUM")]
    optimal_concept = eligible[0][0] if eligible else None

    for cid, info in rows:
        is_actual = cid == actual_concept
        is_optimal = cid == optimal_concept
        badge = (
            '<span style="font-size:10px;background:rgba(31,119,180,0.15);'
            'color:#1F77B4;padding:2px 6px;border-radius:4px;'
            'font-weight:700;margin-left:6px;">ACTUAL</span>'
            if is_actual else
            '<span style="font-size:10px;background:rgba(52,168,83,0.15);'
            'color:#1e7a3a;padding:2px 6px;border-radius:4px;'
            'font-weight:700;margin-left:6px;">🟢 OPTIMAL</span>'
            if is_optimal else ""
        )
        if is_optimal:
            bg, border = "rgba(52,168,83,0.10)", "border-left:4px solid #34A853;"
        elif is_actual:
            bg, border = "rgba(31,119,180,0.08)", "border-left:4px solid #1F77B4;"
        else:
            bg, border = "rgba(0,0,0,0.04)", ""

        conf_color = _confidence_color(info["confidence"])
        conf_tip = (
            f"Sample: {info['n']} comparable plays. "
            f"≥30 = HIGH · 10–29 = MEDIUM · <10 = LOW. "
            + ("Strong baseline." if info["confidence"] == "HIGH"
               else "Treat as directional, not definitive."
               if info["confidence"] == "MEDIUM"
               else "Small sample — interpret cautiously.")
        )
        # Concept blurb + verdict
        verdict = ""
        if not is_actual and actual_epa is not None:
            diff = info["epa"] - actual_epa
            if diff >= 0.20:
                verdict = (f" **Notably higher EV** — gains "
                            f"~{diff:.2f} more EPA than what was called.")
            elif diff >= 0.05:
                verdict = " Marginally higher-EV alternative."
            elif diff <= -0.20:
                verdict = (f" **Notably lower EV** — loses "
                            f"~{abs(diff):.2f} EPA vs what was called.")
            elif diff <= -0.05:
                verdict = " Marginally lower-EV alternative."
            else:
                verdict = " Roughly equivalent to the actual call."
        full_blurb = f"_{info['blurb']}_{verdict}"

        st.markdown(
            f'<div style="padding:8px 10px;background:{bg};{border}'
            f'border-radius:6px;margin-bottom:6px;font-size:13px;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;">'
            f'<div>{info["label"]}{badge}</div>'
            f'<div>'
            f'<span style="font-weight:700;">{info["epa"]:+.2f} EPA</span>'
            f'<span title="{conf_tip}" '
            f'style="font-size:10px;color:white;background:{conf_color};'
            f'padding:2px 6px;border-radius:4px;margin-left:8px;'
            f'cursor:help;font-weight:700;">'
            f'{info["confidence"]} · n={info["n"]}</span>'
            f'</div></div>'
            f'<div style="font-size:12px;opacity:0.85;'
            f'margin-top:6px;line-height:1.4;">{full_blurb}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Auto-narrative
    if actual_info and optimal_concept and optimal_concept in cf:
        best_info = cf[optimal_concept]
        delta = best_info["epa"] - actual_info["epa"]
        if delta > 0.10:
            st.info(
                f"**Monday-morning QB:** "
                f"{best_info['label']} was the highest-EV response to "
                f"{_COVERAGE_LABELS.get(cov, cov)} on this down/distance "
                f"(expected {best_info['epa']:+.2f} EPA vs the "
                f"{actual_info['epa']:+.2f} from the actual call). "
                f"Difference: +{delta:.2f} EPA."
            )


def _render_critical_play(idx: int, play: pd.Series, team: str) -> None:
    """One expandable card per critical play with counterfactual analysis.

    Branches on which side of the ball the team was on:
    - posteam == team (offense): "Given the defense called X, what
      offensive concept would have been the optimal answer?"
    - defteam == team (defense): "What other coverage could we have
      called against this matchup?"
    """
    qtr = int(play.get("qtr", 0)) if pd.notna(play.get("qtr")) else "—"
    qsr = int(play.get("quarter_seconds_remaining", 0)) if pd.notna(play.get("quarter_seconds_remaining")) else 0
    qsr_min = qsr // 60
    qsr_sec = qsr % 60
    wpa = play.get("team_wpa", play.get("wpa", 0))
    wpa_pct = wpa * 100 if pd.notna(wpa) else 0
    desc = str(play.get("desc", ""))[:200]
    sign = "▲" if wpa_pct > 0 else "▼"
    team_was_offense = (play.get("posteam") == team)
    side_label = "🏈 Offense" if team_was_offense else "🛡️ Defense"
    label = (
        f"#{idx} — Q{qtr} {qsr_min}:{qsr_sec:02d} · "
        f"{sign} {abs(wpa_pct):.1f}% WP swing · {side_label}"
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

        # Branch: offense view (offensive counterfactual) vs defense
        # view (defensive coverage counterfactual)
        if team_was_offense:
            _render_offensive_counterfactual(play, cov)
            return

        # DEFENSIVE counterfactual — only meaningful if it was a pass
        if play.get("pass_attempt") == 1 and pd.notna(play.get("air_yards")):
            cf = _coverage_counterfactual(play)
            if cf:
                st.markdown("**🤔 What if the defense had called something else?**")
                st.caption(
                    "Expected **offensive** EPA per attempt for similar "
                    "matchups (same pass depth/location + similar down & "
                    "distance) across the league. **Lower EPA = better for "
                    "the defense** (offense expected to gain less). The "
                    "**🟢 optimal** call against this specific matchup is "
                    "highlighted. Confidence flag reflects sample size."
                )
                # Sort coverages by expected EPA (best for defense first = lowest EPA)
                rows = sorted(cf.items(), key=lambda kv: kv[1]["epa"])
                actual_epa = cf.get(cov, {}).get("epa") if cov in cf else None
                # Identify the optimal call — lowest EPA that's NOT what was
                # called and confidence is at least MEDIUM (don't crown a
                # tiny sample as "optimal").
                _eligible_optimal = [
                    (k, v) for k, v in rows
                    if k != cov and v["confidence"] in ("HIGH", "MEDIUM")
                ]
                optimal_cov = (_eligible_optimal[0][0]
                                if _eligible_optimal else None)

                for cov_code, info in rows:
                    is_actual = cov_code == cov
                    is_optimal = cov_code == optimal_cov
                    badge = (
                        '<span style="font-size:10px;background:rgba(31,119,180,0.15);'
                        'color:#1F77B4;padding:2px 6px;border-radius:4px;'
                        'font-weight:700;margin-left:6px;">ACTUAL</span>'
                        if is_actual else
                        '<span style="font-size:10px;background:rgba(52,168,83,0.15);'
                        'color:#1e7a3a;padding:2px 6px;border-radius:4px;'
                        'font-weight:700;margin-left:6px;">🟢 OPTIMAL</span>'
                        if is_optimal else ""
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
                    blurb = _coverage_blurb(cov_code, play, info["epa"],
                                              actual_epa, is_actual)
                    # Row styling — optimal call gets a green left border +
                    # tint; actual call gets a blue left border.
                    if is_optimal:
                        bg = "rgba(52,168,83,0.10)"
                        border = "border-left:4px solid #34A853;"
                    elif is_actual:
                        bg = "rgba(31,119,180,0.08)"
                        border = "border-left:4px solid #1F77B4;"
                    else:
                        bg = "rgba(0,0,0,0.04)"
                        border = ""
                    st.markdown(
                        f'<div style="padding:8px 10px;background:{bg};'
                        f'{border}border-radius:6px;margin-bottom:6px;font-size:13px;">'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'align-items:center;">'
                        f'<div>{_COVERAGE_LABELS.get(cov_code, cov_code)}{badge}</div>'
                        f'<div>'
                        f'<span style="font-weight:700;">{info["epa"]:+.2f} EPA</span>'
                        f'<span title="{conf_tip}" '
                        f'style="font-size:10px;color:white;background:{conf_color};'
                        f'padding:2px 6px;border-radius:4px;margin-left:8px;'
                        f'cursor:help;font-weight:700;">'
                        f'{info["confidence"]} · n={info["n"]}</span>'
                        f'</div></div>'
                        f'<div style="font-size:12px;opacity:0.85;'
                        f'margin-top:6px;line-height:1.4;">{blurb}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Auto-narrative: pick the optimal counterfactual
                actual_info = cf.get(cov, {}) if cov in cf else None
                if (actual_info and optimal_cov
                        and optimal_cov in cf
                        and len(cf) >= 2):
                    best_info = cf[optimal_cov]
                    delta = actual_info["epa"] - best_info["epa"]
                    if delta > 0.10:
                        st.info(
                            f"**Monday-morning QB:** "
                            f"{_COVERAGE_LABELS.get(optimal_cov, optimal_cov)} "
                            f"would have been the optimal call against this "
                            f"matchup (expected to give up only "
                            f"{best_info['epa']:+.2f} EPA vs the "
                            f"{actual_info['epa']:+.2f} they actually gave "
                            f"up). Difference: −{delta:.2f} EPA — that's how "
                            f"much offense the defense would have prevented."
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

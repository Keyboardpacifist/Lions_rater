"""Stat-level drill-down narratives for the Team page.

When a fan clicks a gap-analysis or trajectory item ("Pass defense
23rd → 3rd"), this generates a 2-3 sentence story citing the actual
players who drove the change — risers, fallers, FA arrivals, departed
veterans — using the existing position parquets.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"


# Stat-label → list of (parquet_filename, position_label) tuples that
# the stat decomposes into. Some stats have multiple position groups.
#
# Brett's call (2026-04-28): exclude OL from the player-level drill-down
# narratives. The OL parquet is gap-attributed EPA, not actual block
# grading — too noisy to attribute team-level rushing changes to specific
# linemen. OL impact gets a unit-level sentence instead (see
# _UNIT_OL_STATS).
_STAT_TO_POSITIONS = {
    # Offense
    "Offensive efficiency":   [("league_qb_all_seasons.parquet", "QB"),
                                  ("league_wr_all_seasons.parquet", "WR"),
                                  ("league_rb_all_seasons.parquet", "RB")],
    "Passing offense":        [("league_qb_all_seasons.parquet", "QB"),
                                  ("league_wr_all_seasons.parquet", "WR"),
                                  ("league_te_all_seasons.parquet", "TE")],
    "Rushing offense":        [("league_rb_all_seasons.parquet", "RB")],
    "Red zone TD rate":       [("league_qb_all_seasons.parquet", "QB"),
                                  ("league_wr_all_seasons.parquet", "WR"),
                                  ("league_te_all_seasons.parquet", "TE")],
    "3rd down conversion":    [("league_qb_all_seasons.parquet", "QB"),
                                  ("league_wr_all_seasons.parquet", "WR")],
    "Ball security":          [("league_qb_all_seasons.parquet", "QB")],
    "Points/game":            [("league_qb_all_seasons.parquet", "QB"),
                                  ("league_wr_all_seasons.parquet", "WR")],
    "4Q offense":             [("league_qb_all_seasons.parquet", "QB")],
    # Defense
    "Defensive efficiency":   [("league_cb_all_seasons.parquet", "CB"),
                                  ("league_s_all_seasons.parquet", "S"),
                                  ("league_lb_all_seasons.parquet", "LB"),
                                  ("league_de_all_seasons.parquet", "EDGE"),
                                  ("league_dt_all_seasons.parquet", "DT")],
    "Pass defense":           [("league_cb_all_seasons.parquet", "CB"),
                                  ("league_s_all_seasons.parquet", "S")],
    "Run defense":            [("league_lb_all_seasons.parquet", "LB"),
                                  ("league_de_all_seasons.parquet", "EDGE"),
                                  ("league_dt_all_seasons.parquet", "DT")],
    "Takeaway rate":          [("league_cb_all_seasons.parquet", "CB"),
                                  ("league_s_all_seasons.parquet", "S"),
                                  ("league_lb_all_seasons.parquet", "LB")],
    "Pressure rate":          [("league_de_all_seasons.parquet", "EDGE"),
                                  ("league_dt_all_seasons.parquet", "DT"),
                                  ("league_lb_all_seasons.parquet", "LB")],
    "Sack rate":              [("league_de_all_seasons.parquet", "EDGE"),
                                  ("league_dt_all_seasons.parquet", "DT")],
    "Points allowed/game":    [("league_cb_all_seasons.parquet", "CB"),
                                  ("league_s_all_seasons.parquet", "S"),
                                  ("league_de_all_seasons.parquet", "EDGE")],
    "4Q defense":             [("league_cb_all_seasons.parquet", "CB"),
                                  ("league_s_all_seasons.parquet", "S")],
    # Special / no-decomp
    "discipline":             [],
    "Discipline":             [],
}

# Map gap-analysis labels (lowercase phrase) to the same position lists.
_GAP_LABEL_NORMALIZE = {
    "offensive efficiency":     "Offensive efficiency",
    "passing offense":          "Passing offense",
    "rushing offense":          "Rushing offense",
    "red zone td rate":         "Red zone TD rate",
    "3rd down conversion":      "3rd down conversion",
    "ball security":            "Ball security",
    "defensive efficiency":     "Defensive efficiency",
    "pass defense":             "Pass defense",
    "run defense":              "Run defense",
    "takeaway production":      "Takeaway rate",
    "pass rush":                "Pressure rate",
    "4th-quarter offense":      "4Q offense",
    "4th-quarter defense":      "4Q defense",
    "discipline":               "Discipline",
}


@st.cache_data(show_spinner=False)
def _load_position_pool(filename: str) -> pd.DataFrame:
    path = _DATA / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pl.read_parquet(path).to_pandas()
    except Exception:
        return pd.DataFrame()
    return df


def _get_team_players_for_position(filename: str, team: str,
                                       season: int) -> pd.DataFrame:
    df = _load_position_pool(filename)
    if df.empty:
        return df
    team_col = "recent_team" if "recent_team" in df.columns else (
        "team" if "team" in df.columns else None)
    season_col = "season_year" if "season_year" in df.columns else (
        "season" if "season" in df.columns else None)
    if team_col is None or season_col is None:
        return pd.DataFrame()
    sub = df[(df[team_col] == team) & (df[season_col] == season)].copy()
    if sub.empty:
        return sub
    z_cols = [c for c in sub.columns if c.endswith("_z")]
    if not z_cols:
        return sub
    sub["_avg_z"] = sub[z_cols].mean(axis=1, skipna=True)
    sub["_n_z_cols"] = len(z_cols)
    for name_col in ("player_display_name", "player_name", "full_name"):
        if name_col in sub.columns:
            sub["_player_name"] = sub[name_col].astype(str)
            break
    pid_col = "player_id" if "player_id" in sub.columns else None
    if pid_col:
        sub["_pid"] = sub[pid_col].astype(str)
    else:
        sub["_pid"] = sub["_player_name"]
    return sub.dropna(subset=["_avg_z", "_player_name"])


def _player_movers(team: str, season: int,
                     parquets: list) -> tuple[list, list, list]:
    """Returns (top_current, biggest_risers, new_arrivals).

    - top_current: top 3 players this year by avg z-score across the
      relevant position groups
    - biggest_risers: players with the biggest score jump from prior
      season (had to be on the team both years)
    - new_arrivals: top players this year who weren't on the team
      last year
    """
    cur_all = []
    prev_all = []
    for filename, pos_label in parquets:
        cur = _get_team_players_for_position(filename, team, season)
        prev = _get_team_players_for_position(filename, team, season - 1)
        if not cur.empty:
            cur = cur.assign(_pos=pos_label)
            cur_all.append(cur)
        if not prev.empty:
            prev = prev.assign(_pos=pos_label)
            prev_all.append(prev)
    if not cur_all:
        return [], [], []
    cur_combined = pd.concat(cur_all, ignore_index=True)
    prev_combined = (pd.concat(prev_all, ignore_index=True)
                     if prev_all else pd.DataFrame())

    top_current = (
        cur_combined.sort_values("_avg_z", ascending=False)
        .head(3)
        .to_dict("records")
    )

    risers = []
    new_arrivals = []
    if not prev_combined.empty:
        prev_pids = set(prev_combined["_pid"].astype(str).tolist())
        joined = cur_combined.merge(
            prev_combined[["_pid", "_avg_z"]].rename(
                columns={"_avg_z": "_prev_avg_z"}),
            on="_pid", how="left",
        )
        joined["_score_delta"] = joined["_avg_z"] - joined["_prev_avg_z"]
        # Risers — were on team both years, biggest improvement
        had_prev = joined.dropna(subset=["_prev_avg_z"])
        if not had_prev.empty:
            risers = (had_prev.sort_values("_score_delta", ascending=False)
                      .head(2).to_dict("records"))
        # New arrivals — weren't on team last year (rookies + FAs)
        new = joined[~joined["_pid"].isin(prev_pids)]
        if not new.empty:
            new_arrivals = (new.sort_values("_avg_z", ascending=False)
                            .head(2).to_dict("records"))
    return top_current, risers, new_arrivals


def _format_score(z: float) -> str:
    sign = "+" if z >= 0 else ""
    return f"{sign}{z:.2f}"


# Stats where the OL is a meaningful contributor — get a unit-level
# observation appended to the narrative instead of player-level callouts.
_OL_INVOLVED_STATS = {
    "Offensive efficiency",
    "Rushing offense",
    "Red zone TD rate",
    "3rd down conversion",
    "Ball security",
    "Points/game",
    "Passing offense",
}


@st.cache_data(show_spinner=False)
def _ol_run_breakdown(team: str, season: int) -> dict | None:
    """Returns the rich run-blocking dataset needed to compute
    confidence-aware per-side narratives:

    {
      "team_by_side": {LEFT: {plays, epa}, CENTER: ..., RIGHT: ...} for current+prior,
      "rb_by_side":    {rb_name: {LEFT: {plays_cur, plays_prev, epa_cur, epa_prev}, ...}},
      "fingerprints":  {rb_name: {side: pct_of_carries_cur}},
    }
    """
    try:
        from lib_splits import _load_rusher_plays, _classify_gap
        rp = _load_rusher_plays()
    except Exception:
        return None
    if rp is None or rp.empty:
        return None
    sub = rp[(rp["team"] == team)
              & (rp["season"].isin([season, season - 1]))].copy()
    if sub.empty:
        return None
    sub["gap_code"] = sub.apply(_classify_gap, axis=1)
    sub = sub.dropna(subset=["gap_code"])
    sub["side"] = sub["gap_code"].apply(
        lambda g: "LEFT" if g.endswith("-L")
                  else "RIGHT" if g.endswith("-R")
                  else "CENTER" if g == "A"
                  else None
    )
    sub = sub.dropna(subset=["side"])
    if sub.empty:
        return None

    # Team-level by side, current + prior
    team_agg = (
        sub.groupby(["season", "side"])
        .agg(plays=("epa", "size"), epa=("epa", "mean"))
        .reset_index()
    )
    team_by_side = {"current": {}, "prior": {}}
    for _, r in team_agg.iterrows():
        bucket = "current" if int(r["season"]) == season else "prior"
        team_by_side[bucket][r["side"]] = {
            "plays": int(r["plays"]),
            "epa": float(r["epa"]),
        }

    # Per-RB by side — keep all primary backs (≥80 total carries either year)
    rb_total = (
        sub.groupby(["rusher_player_name", "season"])
        .size()
        .reset_index(name="n")
    )
    primary_backs = (
        rb_total[rb_total["n"] >= 80]["rusher_player_name"]
        .unique()
        .tolist()
    )
    rb_by_side = {}
    fingerprints = {}
    for rb in primary_backs:
        rb_sub = sub[sub["rusher_player_name"] == rb]
        # Side aggregates
        side_agg = (
            rb_sub.groupby(["season", "side"])
            .agg(plays=("epa", "size"), epa=("epa", "mean"))
            .reset_index()
        )
        record: dict = {}
        for side in ("LEFT", "CENTER", "RIGHT"):
            cur_q = side_agg[(side_agg["season"] == season)
                              & (side_agg["side"] == side)]
            prev_q = side_agg[(side_agg["season"] == season - 1)
                                & (side_agg["side"] == side)]
            record[side] = {
                "plays_cur": int(cur_q["plays"].iloc[0]) if not cur_q.empty else 0,
                "plays_prev": int(prev_q["plays"].iloc[0]) if not prev_q.empty else 0,
                "epa_cur": float(cur_q["epa"].iloc[0]) if not cur_q.empty else None,
                "epa_prev": float(prev_q["epa"].iloc[0]) if not prev_q.empty else None,
            }
        rb_by_side[rb] = record

        # Fingerprint: % of current-season carries per side
        cur_rb = rb_sub[rb_sub["season"] == season]
        total_cur = max(len(cur_rb), 1)
        fingerprints[rb] = {
            "LEFT":   100 * (cur_rb["side"] == "LEFT").sum() / total_cur,
            "CENTER": 100 * (cur_rb["side"] == "CENTER").sum() / total_cur,
            "RIGHT":  100 * (cur_rb["side"] == "RIGHT").sum() / total_cur,
        }

    return {
        "team_by_side": team_by_side,
        "rb_by_side": rb_by_side,
        "fingerprints": fingerprints,
    }


# Confidence thresholds — both seasons must hit _QUAL_PLAYS_PER_SIDE for
# a back's signal on a given side to be considered reliable. 40 ≈ a back
# averaging 2-3 carries/game on that side; below that the per-side EPA
# noise dominates the signal.
_QUAL_PLAYS_PER_SIDE = 40
_AGREE_DELTA_THRESHOLD = 0.04  # |delta| < this = "held"


def _classify_back_direction(epa_cur: float | None,
                                epa_prev: float | None) -> str | None:
    if epa_cur is None or epa_prev is None:
        return None
    delta = epa_cur - epa_prev
    if abs(delta) < _AGREE_DELTA_THRESHOLD:
        return "held"
    return "improved" if delta > 0 else "slipped"


def _confidence_for_side(side: str, breakdown: dict) -> dict:
    """Returns {confidence: 'HIGH'|'CAUTION'|'NA',
                reason: <human-readable why>,
                qualifying_backs: [...], non_qualifying_backs: [...]}.

    Confidence rules:
    - HIGH: ≥2 backs qualify (≥60 carries on that side both years) AND
            they agree on direction (all slipped / all improved / all held)
    - CAUTION: backs disagree, OR only 1 back qualifies, OR all backs
            below sample threshold
    - NA: no current-season carries on that side
    """
    rb_by_side = breakdown.get("rb_by_side", {})
    fingerprints = breakdown.get("fingerprints", {})
    team = breakdown.get("team_by_side", {}).get("current", {}).get(side)
    if not team or team["plays"] == 0:
        return {"confidence": "NA", "reason": "No carries to this side.",
                "qualifying": [], "below_threshold": []}

    qualifying = []
    below = []
    for rb, sides in rb_by_side.items():
        rec = sides.get(side, {})
        if (rec.get("plays_cur", 0) >= _QUAL_PLAYS_PER_SIDE
                and rec.get("plays_prev", 0) >= _QUAL_PLAYS_PER_SIDE):
            qualifying.append({
                "name": rb,
                "epa_cur": rec["epa_cur"],
                "epa_prev": rec["epa_prev"],
                "direction": _classify_back_direction(
                    rec["epa_cur"], rec["epa_prev"]),
                "plays_cur": rec["plays_cur"],
                "plays_prev": rec["plays_prev"],
                "fingerprint_pct": fingerprints.get(rb, {}).get(side, 0),
            })
        elif rec.get("plays_cur", 0) > 0 or rec.get("plays_prev", 0) > 0:
            below.append({
                "name": rb,
                "epa_cur": rec["epa_cur"],
                "epa_prev": rec["epa_prev"],
                "direction": _classify_back_direction(
                    rec["epa_cur"], rec["epa_prev"]),
                "plays_cur": rec["plays_cur"],
                "plays_prev": rec["plays_prev"],
                "fingerprint_pct": fingerprints.get(rb, {}).get(side, 0),
            })

    if len(qualifying) >= 2:
        directions = {q["direction"] for q in qualifying}
        if len(directions) == 1:
            return {
                "confidence": "HIGH",
                "reason": (
                    f"All {len(qualifying)} primary backs with meaningful "
                    f"samples on this side ({_QUAL_PLAYS_PER_SIDE}+ carries "
                    f"each year) agree."
                ),
                "qualifying": qualifying,
                "below_threshold": below,
            }
        # Disagreement among qualifying backs — explain via fingerprints
        slipping = [q for q in qualifying if q["direction"] == "slipped"]
        non_slipping = [q for q in qualifying
                         if q["direction"] != "slipped"]
        slip_names = ", ".join(q["name"] for q in slipping)
        nonslip_names = ", ".join(q["name"] for q in non_slipping)
        # Run-style fingerprint check
        fp_note = ""
        if slipping and non_slipping:
            slip_pct = sum(q["fingerprint_pct"] for q in slipping) / len(slipping)
            nonslip_pct = sum(q["fingerprint_pct"] for q in non_slipping) / len(non_slipping)
            if abs(slip_pct - nonslip_pct) >= 8:
                fp_note = (
                    f" Run-style differs: {slip_names} runs to this side "
                    f"{slip_pct:.0f}% of carries, {nonslip_names} "
                    f"{nonslip_pct:.0f}% — different exposure to the "
                    f"same OL changes."
                )
        return {
            "confidence": "CAUTION",
            "reason": (
                f"Primary backs disagree on this side. {slip_names} "
                f"slipped while {nonslip_names} held or improved.{fp_note}"
            ),
            "qualifying": qualifying,
            "below_threshold": below,
        }

    if len(qualifying) == 1:
        q = qualifying[0]
        return {
            "confidence": "CAUTION",
            "reason": (
                f"Only one primary back ({q['name']}) has a meaningful "
                f"sample ({_QUAL_PLAYS_PER_SIDE}+ carries) on this side "
                f"both years — single-back signal can't be corroborated. "
                f"Other backs ran here too rarely to evaluate."
            ),
            "qualifying": qualifying,
            "below_threshold": below,
        }

    return {
        "confidence": "CAUTION",
        "reason": (
            "No primary back hit the sample threshold "
            f"({_QUAL_PLAYS_PER_SIDE}+ carries) on this side both years. "
            f"Team-level signal is built on small samples."
        ),
        "qualifying": qualifying,
        "below_threshold": below,
    }


def _confidence_badge(conf: dict) -> str:
    """Render a confidence indicator with hover-tooltip explanation.
    Uses the HTML title attribute so hovering shows the reason text.
    """
    if not conf or conf.get("confidence") == "NA":
        return ""
    if conf["confidence"] == "HIGH":
        return (
            f'<span title="{conf["reason"]}" '
            f'style="font-size:11px;background:rgba(52,168,83,0.15);'
            f'color:#1e7a3a;padding:2px 6px;border-radius:6px;'
            f'font-weight:700;cursor:help;">'
            f'🟢 HIGH</span>'
        )
    # CAUTION
    return (
        f'<span title="{conf["reason"]}" '
        f'style="font-size:11px;background:rgba(230,126,34,0.15);'
        f'color:#a8541d;padding:2px 6px;border-radius:6px;'
        f'font-weight:700;cursor:help;">'
        f'⚠️ CAUTION (hover for why)</span>'
    )


def _classify_delta(delta: float) -> tuple[str, str]:
    """(arrow, descriptor) for an EPA delta."""
    if abs(delta) < 0.04:
        return "→", "held"
    if delta < -0.10:
        return "▼▼", "cratered"
    if delta < 0:
        return "▼", "slipped"
    if delta > 0.10:
        return "▲▲", "surged"
    return "▲", "improved"


def _format_side_yoy(side: str, cur: dict, prev: dict | None) -> str:
    cur_epa = cur["epa"]
    sign_cur = "+" if cur_epa >= 0 else ""
    if prev is None or "epa" not in prev:
        return (f"**{side.title()}**: {sign_cur}{cur_epa:.2f} EPA/att "
                f"on {cur['plays']} carries")
    prev_epa = prev["epa"]
    sign_prev = "+" if prev_epa >= 0 else ""
    arrow, descriptor = _classify_delta(cur_epa - prev_epa)
    return (f"**{side.title()}**: {sign_cur}{cur_epa:.2f} EPA/att "
            f"(was {sign_prev}{prev_epa:.2f}) — {arrow} _{descriptor}_")


@st.cache_data(show_spinner=False)
def _rb_consensus_check(team: str, season: int,
                          min_plays_per_side: int = 25) -> dict | None:
    """For each RB on the team with meaningful current+prior workload,
    compute per-side EPA deltas. Returns a dict that lets the narrative
    answer: 'do multiple backs agree on the OL signal, or is this one
    back's regression?'

    When 2+ backs slip on the same side(s), the cause is upstream
    (OL or scheme). When only one back slips, it's likely the back.
    """
    try:
        from lib_splits import _load_rusher_plays, _classify_gap
        rp = _load_rusher_plays()
    except Exception:
        return None
    if rp is None or rp.empty:
        return None
    sub = rp[(rp["team"] == team)
              & (rp["season"].isin([season, season - 1]))].copy()
    if sub.empty:
        return None
    sub["gap_code"] = sub.apply(_classify_gap, axis=1)
    sub = sub.dropna(subset=["gap_code"])
    sub["side"] = sub["gap_code"].apply(
        lambda g: "LEFT" if g.endswith("-L")
                  else "RIGHT" if g.endswith("-R")
                  else "CENTER" if g == "A"
                  else None
    )
    sub = sub.dropna(subset=["side"])
    if sub.empty:
        return None

    # Only consider RBs who have prior-AND-current samples. Drop scrambling
    # QBs (rusher names that match common QB patterns) — heuristic via
    # checking whether the rusher had > 50 carries total.
    rb_carries = (
        sub.groupby(["rusher_player_name", "season"])
        .size()
        .reset_index(name="n")
    )
    rb_seasons = rb_carries.groupby("rusher_player_name")["season"].nunique()
    eligible = rb_seasons[rb_seasons == 2].index.tolist()
    if not eligible:
        return None
    # And require ≥80 carries per season (filters out QBs / specialists)
    qual = (
        sub[sub["rusher_player_name"].isin(eligible)]
        .groupby(["rusher_player_name", "season"])
        .size()
        .reset_index(name="n")
    )
    qual = qual[qual["n"] >= 80]
    eligible = (
        qual.groupby("rusher_player_name")["season"]
        .nunique()
        .pipe(lambda s: s[s == 2].index.tolist())
    )
    if not eligible:
        return None

    out = {}  # {rb_name: {side: {"cur": epa, "prev": epa, "delta": ...}}}
    for rb in eligible:
        rb_sub = sub[sub["rusher_player_name"] == rb]
        agg = (
            rb_sub.groupby(["season", "side"])
            .agg(plays=("epa", "size"), epa=("epa", "mean"),
                 ypc=("yards_gained", "mean"))
            .reset_index()
        )
        rb_record: dict = {}
        for side in ("LEFT", "CENTER", "RIGHT"):
            cur_q = agg[(agg["season"] == season) & (agg["side"] == side)]
            prev_q = agg[(agg["season"] == season - 1) & (agg["side"] == side)]
            if cur_q.empty or prev_q.empty:
                continue
            if int(cur_q["plays"].iloc[0]) < min_plays_per_side or \
               int(prev_q["plays"].iloc[0]) < min_plays_per_side:
                continue
            cur_epa = float(cur_q["epa"].iloc[0])
            prev_epa = float(prev_q["epa"].iloc[0])
            rb_record[side] = {
                "cur_epa": cur_epa,
                "prev_epa": prev_epa,
                "delta": cur_epa - prev_epa,
            }
        if rb_record:
            out[rb] = rb_record
    return out if out else None


def _consensus_signal_text(rb_consensus: dict) -> str:
    """Read the per-RB pattern and write a one-sentence interpretation:
    consensus signal (multiple backs agree → OL/scheme), or divergent
    (likely one back's issue)."""
    if not rb_consensus or len(rb_consensus) < 2:
        return ""

    # For each side, count how many backs slipped (delta < -0.04)
    # and how many improved/held
    side_votes = {"LEFT": [], "CENTER": [], "RIGHT": []}
    for rb_name, sides in rb_consensus.items():
        for side, info in sides.items():
            d = info["delta"]
            if d < -0.04:
                side_votes[side].append(("slipped", rb_name))
            elif d > 0.04:
                side_votes[side].append(("improved", rb_name))
            else:
                side_votes[side].append(("held", rb_name))

    def _direction_consensus(side):
        votes = [v[0] for v in side_votes[side]]
        if not votes:
            return None
        if all(v == "slipped" for v in votes) and len(votes) >= 2:
            return "all-slip"
        if all(v == "improved" for v in votes) and len(votes) >= 2:
            return "all-improve"
        return None

    sides_with_consensus_slip = [
        s for s in ("LEFT", "CENTER", "RIGHT")
        if _direction_consensus(s) == "all-slip"
    ]
    sides_with_consensus_improve = [
        s for s in ("LEFT", "CENTER", "RIGHT")
        if _direction_consensus(s) == "all-improve"
    ]

    n_backs = len(rb_consensus)
    if sides_with_consensus_slip:
        sides_text = ", ".join(s.lower() for s in sides_with_consensus_slip)
        return (
            f"**Unit signal:** all {n_backs} primary backs slipped on the "
            f"{sides_text} side — when multiple backs agree, the cause is "
            f"upstream (OL or scheme), not the runner."
        )
    if sides_with_consensus_improve:
        sides_text = ", ".join(s.lower() for s in sides_with_consensus_improve)
        return (
            f"**Unit signal:** all {n_backs} primary backs ran better to "
            f"the {sides_text} side — that's the part of the line working."
        )

    # Divergent — at least one back slipped on a side where another held
    # or improved. Read it as a back-specific or usage signal, not OL.
    divergent_sides = []
    for side in ("LEFT", "CENTER", "RIGHT"):
        directions = {v[0] for v in side_votes[side]}
        # Need both 'slipped' and either 'improved' or 'held' on the
        # same side to call it divergent.
        if "slipped" in directions and (
            "improved" in directions or "held" in directions
        ):
            divergent_sides.append(side)
    if divergent_sides:
        sides_text = ", ".join(s.lower() for s in divergent_sides)
        slipping_backs = []
        for side in divergent_sides:
            for direction, rb_name in side_votes[side]:
                if direction == "slipped" and rb_name not in slipping_backs:
                    slipping_backs.append(rb_name)
        bk_text = " and ".join(slipping_backs)
        return (
            f"**Unit signal:** backs disagree on the {sides_text} side — "
            f"{bk_text} slipped while the other primary back held or "
            f"improved. When backs diverge, the cause is more likely "
            f"back-specific or usage-pattern (different gap/personnel "
            f"deployment) than pure OL deterioration."
        )
    return ""


@st.cache_data(show_spinner=False)
def _ol_unit_observation(team: str, season: int) -> str:
    """Unit-wide story for the offensive line — pass protection,
    run-blocking, and discipline as a unit. Ranks computed across
    all 32 NFL OL units in the same season."""
    ol_path = _DATA / "league_ol_all_seasons.parquet"
    if not ol_path.exists():
        return ""
    try:
        ol = pl.read_parquet(ol_path).to_pandas()
    except Exception:
        return ""

    team_col = "recent_team" if "recent_team" in ol.columns else "team"
    season_col = "season_year" if "season_year" in ol.columns else "season"

    # Team-season aggregate. team_sack_rate / team_pressure_rate are
    # shared across linemen so first() works. Run-blocking is mean of
    # pos_run_epa across the unit. Penalty rate also unit-mean.
    def _agg(season_arg):
        df_s = ol[ol[season_col] == season_arg]
        if df_s.empty:
            return None
        agg = (
            df_s.groupby(team_col)
            .agg(
                team_sack_rate=("team_sack_rate", "first"),
                team_pressure_rate=("team_pressure_rate", "first"),
                avg_run_epa=("pos_run_epa", "mean"),
                avg_run_success=("pos_run_success", "mean"),
                avg_penalty_rate=("penalty_rate", "mean"),
            )
            .reset_index()
            .rename(columns={team_col: "team"})
        )
        return agg

    cur_all = _agg(season)
    prev_all = _agg(season - 1)
    if cur_all is None:
        return ""

    cur_row_q = cur_all[cur_all["team"] == team]
    if cur_row_q.empty:
        return ""
    cur_row = cur_row_q.iloc[0]

    prev_row = None
    if prev_all is not None:
        prev_row_q = prev_all[prev_all["team"] == team]
        if not prev_row_q.empty:
            prev_row = prev_row_q.iloc[0]

    def _rank(df, col, value, ascending):
        s = df[col].dropna()
        if s.empty:
            return None, 0
        ranked = s.sort_values(ascending=ascending).reset_index(drop=True)
        # Position of the value in the sort
        pos = (ranked.values == value).nonzero()[0]
        if len(pos) == 0:
            return None, len(ranked)
        return int(pos[0]) + 1, len(ranked)

    sack_rank, total = _rank(cur_all, "team_sack_rate",
                                cur_row["team_sack_rate"], ascending=True)
    press_rank, _ = _rank(cur_all, "team_pressure_rate",
                            cur_row["team_pressure_rate"], ascending=True)
    run_rank, _ = _rank(cur_all, "avg_run_epa",
                          cur_row["avg_run_epa"], ascending=False)
    pen_rank, _ = _rank(cur_all, "avg_penalty_rate",
                          cur_row["avg_penalty_rate"], ascending=True)

    def _ord(n):
        if n is None: return "—"
        suf = "th"
        if n % 100 not in (11, 12, 13):
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    def _yoy(cur_v, prev_v, fmt="{:.1%}"):
        if prev_row is None or pd.isna(prev_v):
            return ""
        d = float(cur_v) - float(prev_v)
        if abs(d) < 0.001:
            return ""
        sign = "▲" if d > 0 else "▼"
        return f" ({sign} from {fmt.format(prev_v)})"

    pieces = []

    # Pass protection
    pp_yoy_sack = _yoy(
        cur_row["team_sack_rate"],
        prev_row["team_sack_rate"] if prev_row is not None else None,
    )
    pp_yoy_press = _yoy(
        cur_row["team_pressure_rate"],
        prev_row["team_pressure_rate"] if prev_row is not None else None,
    )
    pieces.append(
        f"**Pass protection:** {cur_row['team_sack_rate']*100:.1f}% sack rate"
        f" allowed ({_ord(sack_rank)} of {total}){pp_yoy_sack}, "
        f"{cur_row['team_pressure_rate']*100:.1f}% pressure rate "
        f"({_ord(press_rank)}){pp_yoy_press}."
    )

    # Run blocking — team-level per-side YoY + RB consensus check.
    # When multiple backs show the same side-by-side YoY pattern, the
    # cause is upstream (OL or scheme). When only one back shows it,
    # it's likely the back. This is the methodology Brett identified —
    # using RB consensus as the unit-level OL signal.
    run_breakdown = _ol_run_breakdown(team, season)
    if run_breakdown and run_breakdown.get("team_by_side", {}).get("current"):
        side_lines = []
        for side in ("LEFT", "CENTER", "RIGHT"):
            cur = run_breakdown["team_by_side"]["current"].get(side)
            prev = run_breakdown["team_by_side"].get("prior", {}).get(side)
            if cur is None:
                continue
            base_text = _format_side_yoy(side, cur, prev)
            conf = _confidence_for_side(side, run_breakdown)
            badge = _confidence_badge(conf)
            side_lines.append(f"{base_text} {badge}")
        if side_lines:
            pieces.append("**Run blocking by side:**  \n" + "  \n".join(side_lines))

            # Per-back fingerprint summary — shows the primary backs and
            # their YoY direction per side. Concrete data behind the
            # confidence verdicts above. Each back name links to their
            # RB page where the full gap-distribution chart lives.
            rb_lines = []
            for rb_name, sides in run_breakdown.get("rb_by_side", {}).items():
                parts_rb = []
                for side in ("LEFT", "CENTER", "RIGHT"):
                    rec = sides.get(side, {})
                    epa_cur = rec.get("epa_cur")
                    epa_prev = rec.get("epa_prev")
                    plays_cur = rec.get("plays_cur", 0)
                    if epa_cur is None or epa_prev is None or plays_cur < 20:
                        continue
                    arrow, _ = _classify_delta(epa_cur - epa_prev)
                    parts_rb.append(f"{side[0]} {arrow}")
                if parts_rb:
                    rb_lines.append(f"_{rb_name}_: " + " · ".join(parts_rb))
            if rb_lines:
                # Track the RB names so the page can render st.page_link
                # buttons below the markdown block.
                pieces.append(
                    "**Per-back YoY** (L = left, C = center, R = right):  \n"
                    + "  \n".join(rb_lines)
                )
                primary_rb_names = list(run_breakdown.get("rb_by_side", {}).keys())
                if primary_rb_names:
                    cta = (
                        "_See full gap-distribution charts: "
                        + " · ".join(f"**{n}**" for n in primary_rb_names)
                        + " — click any name on the RB rater page._"
                    )
                    pieces.append(cta)
        else:
            pieces.append(
                f"**Run blocking:** {cur_row['avg_run_epa']:+.3f} EPA per "
                f"gap-attributed run, {cur_row['avg_run_success']*100:.1f}% "
                f"success rate ({_ord(run_rank)} of {total})."
            )
    else:
        # Fallback when rusher-plays data isn't available locally
        run_yoy = _yoy(
            cur_row["avg_run_epa"],
            prev_row["avg_run_epa"] if prev_row is not None else None,
            fmt="{:+.3f}",
        )
        pieces.append(
            f"**Run blocking:** {cur_row['avg_run_epa']:+.3f} EPA per "
            f"gap-attributed run, {cur_row['avg_run_success']*100:.1f}% success "
            f"rate ({_ord(run_rank)} of {total}){run_yoy}."
        )

    # Discipline
    pen_yoy = _yoy(
        cur_row["avg_penalty_rate"],
        prev_row["avg_penalty_rate"] if prev_row is not None else None,
        fmt="{:.2f}/g",
    )
    pieces.append(
        f"**Discipline:** {cur_row['avg_penalty_rate']:.2f} flags per game "
        f"per starter ({_ord(pen_rank)} fewest of {total}){pen_yoy}."
    )

    return "**OL unit:**  \n" + "  \n".join(pieces)


def get_drilldown_narrative(team: str, season: int,
                                stat_label: str,
                                direction: str = "neutral") -> str:
    """Generate a 2-3 sentence drill-down explanation for a given stat.

    `direction` — 'improvement' | 'gap' | 'slipped' — tweaks framing.
    Returns plain text suitable for st.markdown rendering.
    """
    normalized = _GAP_LABEL_NORMALIZE.get(stat_label.lower(), stat_label)
    parquets = _STAT_TO_POSITIONS.get(normalized, [])
    if not parquets:
        if normalized.lower() == "discipline":
            return (
                "**Discipline doesn't decompose cleanly to player-level data** "
                "in our system — penalty data sits at the team level. The "
                "rank shift here reflects the unit's overall flag count, not "
                "any one player."
            )
        return (
            f"This stat doesn't have a clean per-player breakdown in our "
            f"system yet — the rank shift reflects team-level aggregate "
            f"performance."
        )

    top, risers, arrivals = _player_movers(team, season, parquets)
    if not top:
        return (
            "Not enough player-level data for this team-season to write "
            "a detailed breakdown."
        )

    parts = []

    # Top contributors this year
    if top:
        names = []
        for p in top[:3]:
            names.append(
                f"**{p['_player_name']}** ({p['_pos']}, "
                f"{_format_score(p['_avg_z'])})"
            )
        parts.append("Top contributors: " + ", ".join(names) + ".")

    # Risers
    if risers:
        riser_phrases = []
        for r in risers:
            delta = r.get("_score_delta")
            if delta is None or pd.isna(delta) or delta < 0.2:
                continue
            riser_phrases.append(
                f"**{r['_player_name']}** ({r['_pos']}) jumped "
                f"{_format_score(r['_prev_avg_z'])} → "
                f"{_format_score(r['_avg_z'])} year-over-year"
            )
        if riser_phrases:
            parts.append("Biggest internal rise: " + " · ".join(riser_phrases) + ".")

    # New arrivals
    if arrivals:
        arr_names = []
        for a in arrivals:
            arr_names.append(
                f"**{a['_player_name']}** ({a['_pos']}, "
                f"{_format_score(a['_avg_z'])})"
            )
        if arr_names:
            parts.append("New this year: " + ", ".join(arr_names) + ".")

    if direction == "improvement":
        parts.insert(0, "📈 **What drove the rise:**")
    elif direction in ("gap", "slipped"):
        parts.insert(0, "🔍 **Where the issue lives:**")

    # Append OL unit-level note for OL-involved stats
    if normalized in _OL_INVOLVED_STATS:
        ol_note = _ol_unit_observation(team, season)
        if ol_note:
            parts.append(ol_note)

    return "\n\n".join(parts)

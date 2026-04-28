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
_STAT_TO_POSITIONS = {
    # Offense
    "Offensive efficiency":   [("league_qb_all_seasons.parquet", "QB"),
                                  ("league_wr_all_seasons.parquet", "WR"),
                                  ("league_rb_all_seasons.parquet", "RB"),
                                  ("league_ol_all_seasons.parquet", "OL")],
    "Passing offense":        [("league_qb_all_seasons.parquet", "QB"),
                                  ("league_wr_all_seasons.parquet", "WR"),
                                  ("league_te_all_seasons.parquet", "TE")],
    "Rushing offense":        [("league_rb_all_seasons.parquet", "RB"),
                                  ("league_ol_all_seasons.parquet", "OL")],
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
    return "\n\n".join(parts)

"""Top-of-page summary strip for the NFL Team page.

Five tiles in one row: Record · SOS · Offense rating · Defense
rating · Team rating. The 'rating' tiles are EPA-based composite
z-scores from team_seasons.parquet — not actual DVOA. We label
them honestly; an opponent-adjusted DVOA-style metric is queued
for a later pipeline pass.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def _season_records(season: int) -> dict:
    """Return {team: {wins, losses, ties, gp, win_pct}} for the
    season's REG-season games. Skips games with missing scores
    (future weeks)."""
    try:
        from lib_team_game_log import load_schedule
    except ImportError:
        return {}
    sched = load_schedule(season)
    if sched.empty:
        return {}
    if "game_type" in sched.columns:
        reg = sched[sched["game_type"] == "REG"].copy()
    else:
        reg = sched.copy()
    out: dict = {}
    teams = pd.concat([reg["home_team"], reg["away_team"]]).dropna().unique()
    for team in teams:
        home = reg[reg["home_team"] == team].dropna(
            subset=["home_score", "away_score"]
        )
        away = reg[reg["away_team"] == team].dropna(
            subset=["home_score", "away_score"]
        )
        wins = int((home["home_score"] > home["away_score"]).sum()
                   + (away["away_score"] > away["home_score"]).sum())
        losses = int((home["home_score"] < home["away_score"]).sum()
                     + (away["away_score"] < away["home_score"]).sum())
        ties = int((home["home_score"] == home["away_score"]).sum()
                   + (away["away_score"] == away["home_score"]).sum())
        gp = wins + losses + ties
        out[team] = {
            "wins": wins, "losses": losses, "ties": ties, "gp": gp,
            "win_pct": (wins + 0.5 * ties) / gp if gp else 0.0,
        }
    return out


@st.cache_data(show_spinner=False)
def _league_sos(season: int) -> dict:
    """Return {team: sos_value} where sos_value is the average
    opponents' regular-season win%. Cached because we compute
    every team's SOS to rank against."""
    try:
        from lib_team_game_log import load_schedule
    except ImportError:
        return {}
    sched = load_schedule(season)
    if sched.empty:
        return {}
    records = _season_records(season)
    if not records:
        return {}
    if "game_type" in sched.columns:
        reg = sched[sched["game_type"] == "REG"].copy()
    else:
        reg = sched.copy()

    out: dict = {}
    for t in records:
        team_games = reg[(reg["home_team"] == t) | (reg["away_team"] == t)]
        opps_winpct = []
        for _, r in team_games.iterrows():
            opp = (r["away_team"] if r["home_team"] == t
                   else r["home_team"])
            if opp in records and records[opp]["gp"] > 0:
                opps_winpct.append(records[opp]["win_pct"])
        if opps_winpct:
            out[t] = sum(opps_winpct) / len(opps_winpct)
    return out


def _tile(label: str, value: str, sub: str = "") -> str:
    sub_html = (f'<div style="font-size:0.72rem;color:#888;'
                f'margin-top:3px;">{sub}</div>') if sub else ""
    return (
        '<div style="flex:1;background:#f8f9fb;border-radius:10px;'
        'padding:12px 14px;text-align:center;'
        'box-shadow:0 1px 2px rgba(0,0,0,0.04);">'
        f'<div style="font-size:0.7rem;color:#888;text-transform:uppercase;'
        f'letter-spacing:0.6px;font-weight:600;">{label}</div>'
        f'<div style="font-size:1.4rem;font-weight:800;color:#1a1a2e;'
        f'margin-top:4px;line-height:1.1;">{value}</div>'
        f'{sub_html}'
        '</div>'
    )


def _fmt_z(z) -> str:
    if z is None or pd.isna(z):
        return "—"
    sign = "+" if z >= 0 else ""
    return f"{sign}{z:.2f}"


def _winpct_str(p: float) -> str:
    s = f"{p:.3f}"
    return s.lstrip("0") if s.startswith("0.") else s


def render_team_header_strip(team: str, season: int,
                                team_row: pd.Series) -> None:
    """Render the 5-tile summary strip. team_row is a row from
    team_seasons.parquet for (team, season)."""
    season = int(season)

    # ── Record + win% ──
    records = _season_records(season)
    rec = records.get(team)
    if rec and rec["gp"] > 0:
        record_str = f"{rec['wins']}-{rec['losses']}"
        if rec["ties"]:
            record_str += f"-{rec['ties']}"
        winpct_sub = _winpct_str(rec["win_pct"]) + " win%"
    else:
        record_str = "—"
        winpct_sub = ""

    # ── SOS — avg opponents' win%, rank among season's teams ──
    sos_map = _league_sos(season)
    sos_val = sos_map.get(team)
    if sos_val is not None:
        sos_str = _winpct_str(sos_val)
        sorted_teams = sorted(sos_map.items(), key=lambda kv: -kv[1])
        rank = next((i + 1 for i, (t, _) in enumerate(sorted_teams)
                       if t == team), None)
        sos_sub = (f"#{rank} of {len(sos_map)} hardest"
                   if rank is not None else "")
    else:
        sos_str = "—"; sos_sub = ""

    # ── EPA-based composite ratings ──
    off_z = team_row.get("off_epa_per_play_z")
    def_z = team_row.get("def_epa_per_play_z")
    team_z = None
    if pd.notna(off_z) and pd.notna(def_z):
        team_z = (float(off_z) + float(def_z)) / 2.0

    tiles_html = (
        '<div style="display:flex;gap:10px;margin:0 0 6px 0;">'
        f'{_tile("RECORD", record_str, winpct_sub)}'
        f'{_tile("STRENGTH OF SCHEDULE", sos_str, sos_sub)}'
        f'{_tile("OFFENSE", _fmt_z(off_z), "EPA composite z")}'
        f'{_tile("DEFENSE", _fmt_z(def_z), "EPA composite z")}'
        f'{_tile("TEAM", _fmt_z(team_z), "off + def avg")}'
        '</div>'
        '<div style="font-size:0.72rem;color:#888;margin:0 0 14px 0;'
        'letter-spacing:0.3px;font-style:italic;">'
        'Offense / Defense / Team are EPA-based composite z-scores '
        '(0 = league avg, +1 ≈ top-25%). Not actual DVOA — opponent-'
        'adjusted version coming later.'
        '</div>'
    )
    st.markdown(tiles_html, unsafe_allow_html=True)

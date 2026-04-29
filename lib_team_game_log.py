"""Per-team game log — schedule with results, score, opponent context.

Pulls from nflreadpy schedules + joins our team_pass_def_quality
parquet so opponent strength is visible alongside each game.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"


@st.cache_data(show_spinner=False)
def load_schedule(season: int) -> pd.DataFrame:
    """Pull the season's schedule via nflreadpy. Cached because the
    pull is moderately expensive."""
    try:
        import nflreadpy as nfl
        return nfl.load_schedules([season]).to_pandas()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_pass_def_quality() -> pd.DataFrame:
    path = _DATA / "team_pass_def_quality.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def get_team_game_log(team: str, season: int) -> pd.DataFrame:
    """Return one row per game for the given (team, season). Columns:

      week, opponent, home_away, team_score, opp_score, result,
      margin, opp_pass_d_rank, ats_result, division_game,
      our_qb, opp_qb, our_coach, opp_coach
    """
    sched = load_schedule(season)
    if sched.empty:
        return pd.DataFrame()
    games = sched[
        ((sched["home_team"] == team) | (sched["away_team"] == team))
    ].copy()
    if games.empty:
        return pd.DataFrame()

    # Lions-perspective fields
    is_home = games["home_team"] == team
    games["opponent"] = games["away_team"].where(is_home, games["home_team"])
    games["home_away"] = is_home.map({True: "vs", False: "at"})
    games["team_score"] = games["home_score"].where(is_home, games["away_score"])
    games["opp_score"] = games["away_score"].where(is_home, games["home_score"])
    games["margin"] = games["team_score"] - games["opp_score"]
    games["result"] = games["margin"].apply(
        lambda m: "W" if pd.notna(m) and m > 0
                   else "L" if pd.notna(m) and m < 0
                   else "T" if pd.notna(m) and m == 0
                   else "—"
    )

    # ATS — vegas spread is from HOME perspective (positive = home favored
    # by that many; the home team covers if margin > spread_line).
    def _ats(row):
        line = row.get("spread_line")
        if pd.isna(line) or pd.isna(row.get("margin")):
            return "—"
        # Convert spread to "team_we_care_about" perspective:
        # if Lions are home, our spread = home line; else our spread = -home line
        our_spread = line if row["home_team"] == team else -line
        # We "covered" if our margin is greater than the spread we were given
        # (e.g. we were +3 favored: cover if win by 4+; we were -3 underdog:
        # cover if lose by ≤2 or win)
        if row["margin"] > our_spread:
            return "cover"
        if row["margin"] < our_spread:
            return "ats_loss"
        return "push"
    games["ats_result"] = games.apply(_ats, axis=1)

    # QB + coach
    games["our_qb"] = games["home_qb_name"].where(is_home, games["away_qb_name"])
    games["opp_qb"] = games["away_qb_name"].where(is_home, games["home_qb_name"])
    games["our_coach"] = games["home_coach"].where(is_home, games["away_coach"])
    games["opp_coach"] = games["away_coach"].where(is_home, games["home_coach"])

    # Opp pass D quality from our existing rank table
    pdq = load_pass_def_quality()
    if not pdq.empty:
        games = games.merge(
            pdq[["team", "season", "epa_rank", "teams_in_season",
                  "quality_quartile"]].rename(columns={
                "team": "opponent", "epa_rank": "opp_pass_d_rank",
                "teams_in_season": "opp_pass_d_total",
                "quality_quartile": "opp_pass_d_quartile",
            }),
            on=["opponent", "season"], how="left",
        )

    # Custom game-type ordering — REG first, then playoff rounds in
    # chronological order (WC → DIV → CON → SB)
    _gt_order = {"REG": 0, "WC": 1, "DIV": 2, "CON": 3, "SB": 4}
    games["_gt_order"] = games["game_type"].map(_gt_order).fillna(99)
    games = games.sort_values(["_gt_order", "week"]).drop(
        columns=["_gt_order"]
    ).reset_index(drop=True)

    # Trim to display columns
    keep = [
        "week", "game_type", "gameday", "home_away", "opponent",
        "team_score", "opp_score", "result", "margin",
        "opp_pass_d_rank", "opp_pass_d_total", "opp_pass_d_quartile",
        "spread_line", "ats_result", "div_game",
        "our_qb", "opp_qb", "our_coach", "opp_coach",
        "stadium",
    ]
    keep = [c for c in keep if c in games.columns]
    return games[keep]


def render_game_log(team: str, season: int) -> None:
    """Render the game log as a Streamlit dataframe with formatting."""
    df = get_team_game_log(team, season)
    if df.empty:
        st.info("No game log data available for this team-season yet.")
        return

    display = pd.DataFrame()
    display["Wk"] = df["week"].astype("Int64")
    display["Type"] = df.get("game_type", pd.Series(dtype=str))
    display["Date"] = df.get("gameday", pd.Series(dtype=str))
    display["Opp"] = df["home_away"] + " " + df["opponent"]
    display["Score"] = (
        df["team_score"].fillna(0).astype(int).astype(str)
        + "–" + df["opp_score"].fillna(0).astype(int).astype(str)
    )
    display["Result"] = df["result"]
    display["Margin"] = df["margin"].apply(
        lambda m: f"{int(m):+d}" if pd.notna(m) else "—"
    )
    if "opp_pass_d_rank" in df.columns:
        display["Opp pass D"] = df.apply(
            lambda r: (f"{int(r['opp_pass_d_rank'])} of "
                        f"{int(r['opp_pass_d_total'])}"
                        if pd.notna(r.get("opp_pass_d_rank")) else "—"),
            axis=1,
        )
    display["ATS"] = df.get("ats_result", pd.Series(["—"] * len(df)))
    display["Spread"] = df.get("spread_line", pd.Series(dtype=float)).apply(
        lambda v: f"{v:+g}" if pd.notna(v) else "—"
    )
    display["Div?"] = df.get("div_game", pd.Series(dtype=int)).apply(
        lambda v: "✓" if v == 1 else ""
    )
    display["Opp QB"] = df.get("opp_qb", pd.Series(dtype=str))
    display["Opp HC"] = df.get("opp_coach", pd.Series(dtype=str))

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Quick highlights — best win, worst loss
    wins = df[df["result"] == "W"]
    losses = df[df["result"] == "L"]
    if not wins.empty:
        bw = wins.loc[wins["margin"].idxmax()]
        st.caption(
            f"🏆 **Best win:** Wk {int(bw['week'])} "
            f"{bw['home_away']} {bw['opponent']} ({int(bw['team_score'])}–"
            f"{int(bw['opp_score'])}, +{int(bw['margin'])})"
        )
    if not losses.empty:
        wl = losses.loc[losses["margin"].idxmin()]
        st.caption(
            f"📉 **Worst loss:** Wk {int(wl['week'])} "
            f"{wl['home_away']} {wl['opponent']} ({int(wl['team_score'])}–"
            f"{int(wl['opp_score'])}, {int(wl['margin'])})"
        )
    # Record summary
    rec = df["result"].value_counts()
    w, l, t = int(rec.get("W", 0)), int(rec.get("L", 0)), int(rec.get("T", 0))
    rec_str = f"{w}–{l}" + (f"–{t}" if t else "")
    ats = df["ats_result"].value_counts()
    cover, lose_ats = int(ats.get("cover", 0)), int(ats.get("ats_loss", 0))
    ats_str = f"{cover}–{lose_ats}" if (cover + lose_ats) > 0 else "n/a"
    st.caption(f"📊 **Record:** {rec_str} · **ATS:** {ats_str}")

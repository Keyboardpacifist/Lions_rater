"""QB panel splits — pressure, situational, competition, throw map.

All built from the per-dropback feed (`data/qb_dropbacks.parquet`).
Used by `pages/QB.py` to surface what the season-aggregate parquet
can't: contextual performance.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_qb_dropbacks() -> pd.DataFrame:
    return pd.read_parquet(_DATA / "qb_dropbacks.parquet")


@st.cache_data
def load_team_pass_def_quality() -> pd.DataFrame:
    return pd.read_parquet(_DATA / "team_pass_def_quality.parquet")


@st.cache_data
def _league_pressure_avg(season: int | None) -> dict:
    db = load_qb_dropbacks()
    if season is not None:
        db = db[db["season"] == season]
    pressured = db[db["is_pressured"]]
    clean = db[~db["is_pressured"]]
    return {
        "clean_epa": float(clean["epa"].mean()) if len(clean) else 0.0,
        "pressured_epa": float(pressured["epa"].mean()) if len(pressured) else 0.0,
        "pressure_rate": float(db["is_pressured"].mean()) if len(db) else 0.0,
    }


def _filter_player(player_id: str, season: int | None) -> pd.DataFrame:
    db = load_qb_dropbacks()
    plays = db[db["passer_player_id"] == player_id]
    if season is not None:
        plays = plays[plays["season"] == season]
    return plays


def render_pressure_split(player_id: str, player_name: str, *,
                            season: int | None = None,
                            theme: dict | None = None) -> None:
    """Bucket 7 — Under pressure.

    Two-bar EPA comparison (clean vs. pressured) with league-avg
    benchmarks, plus a detail row of completion/sack rates.
    """
    plays = _filter_player(player_id, season)
    if plays.empty or len(plays) < 30:
        st.caption(f"_Not enough dropbacks ({len(plays)}) to compute pressure splits._")
        return

    pressured = plays[plays["is_pressured"]]
    clean = plays[~plays["is_pressured"]]

    qb_clean = float(clean["epa"].mean()) if len(clean) else 0.0
    qb_pressed = float(pressured["epa"].mean()) if len(pressured) else 0.0
    league = _league_pressure_avg(season)

    primary = (theme or {}).get("primary", "#1F77B4")
    pressed_color = (theme or {}).get("secondary", "#D62728")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=player_name,
        x=["Clean pocket", "Pressured"],
        y=[qb_clean, qb_pressed],
        marker_color=[primary, pressed_color],
        text=[f"{qb_clean:+.2f}", f"{qb_pressed:+.2f}"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:+.3f} EPA/dropback<br>"
                      "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=["Clean pocket", "Pressured"],
        y=[league["clean_epa"], league["pressured_epa"]],
        mode="markers",
        marker=dict(symbol="line-ew", size=60, color="#999",
                    line=dict(width=3, color="#999")),
        name="League avg",
        hovertemplate="League avg: %{y:+.3f} EPA<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="EPA per dropback",
        height=360,
        margin=dict(l=40, r=20, t=30, b=40),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                    bgcolor="rgba(255,255,255,0.7)"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_hline(y=0, line_color="#ccc", line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pressure rate",
               f"{plays['is_pressured'].mean()*100:.1f}%",
               help="% of his dropbacks where he got hit. League avg: "
                    f"{league['pressure_rate']*100:.1f}%")
    c2.metric("Clean comp %",
               f"{clean['complete_pass'].mean()*100:.1f}%" if len(clean) else "—")
    c3.metric("Pressured comp %",
               f"{pressured['complete_pass'].mean()*100:.1f}%" if len(pressured) else "—")
    sack_rate = float(pressured["sack"].mean()) if len(pressured) else 0.0
    c4.metric("Sack-when-pressured",
               f"{sack_rate*100:.1f}%",
               help="When he got hit, did he get sacked? Lower = better")

    # Narrative line: how does his clean→pressured drop compare to league?
    qb_drop = qb_clean - qb_pressed
    lg_drop = league["clean_epa"] - league["pressured_epa"]
    delta = qb_drop - lg_drop
    if delta < -0.05:
        st.success(
            f"**Holds up under pressure.** EPA drops only "
            f"{qb_drop:.2f}/dropback when hit — league avg drop is "
            f"{lg_drop:.2f}, so he loses {abs(delta):.2f} less than typical."
        )
    elif delta > 0.05:
        st.warning(
            f"**Struggles under pressure.** EPA drops "
            f"{qb_drop:.2f}/dropback when hit — league avg drop is only "
            f"{lg_drop:.2f}, so he loses {delta:.2f} more than typical."
        )
    else:
        st.info(
            f"**Average under pressure.** EPA drop of {qb_drop:.2f}/dropback "
            f"is roughly league-typical ({lg_drop:.2f})."
        )

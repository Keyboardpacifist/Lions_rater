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


def render_competition_split(player_id: str, player_name: str, *,
                                season: int | None = None,
                                theme: dict | None = None) -> None:
    """Bucket 8 — Elite vs. weak competition.

    Splits the QB's plays by which quartile of pass D they faced
    (1=elite, 4=bad), shows aggregate EPA & completion % per quartile,
    and renders a per-game scatter (opp D quality vs. QB EPA) that
    reveals whether he rises or fades against good defenses.
    """
    plays = _filter_player(player_id, season)
    if plays.empty or len(plays) < 50:
        st.caption(f"_Not enough dropbacks ({len(plays)}) to compute "
                    f"competition splits._")
        return

    def_quality = load_team_pass_def_quality()
    # Merge: opponent's pass D quality for each play
    plays = plays.merge(
        def_quality[["team", "season", "pass_epa_allowed_per_play",
                     "epa_rank", "quality_quartile"]],
        left_on=["defteam", "season"],
        right_on=["team", "season"],
        how="left",
    )
    plays = plays.dropna(subset=["quality_quartile"])
    if plays.empty:
        st.caption("_No matched defenses for these plays._")
        return

    # Aggregate by quartile
    by_q = (
        plays.groupby("quality_quartile")
        .agg(
            n_plays=("epa", "size"),
            epa=("epa", "mean"),
            comp_pct=("complete_pass", "mean"),
            int_rate=("interception", "mean"),
            sack_rate=("sack", "mean"),
        )
        .reset_index()
    )
    quartile_labels = {1: "Elite (top 25%)", 2: "Above avg", 3: "Below avg",
                        4: "Bad (bottom 25%)"}
    by_q["label"] = by_q["quality_quartile"].map(quartile_labels)

    # ── Bar chart: EPA per dropback by quartile ──
    primary = (theme or {}).get("primary", "#1F77B4")
    secondary = (theme or {}).get("secondary", "#D62728")
    bar_colors = [secondary, "#aa6e7e", "#7e9eaa", primary]  # 1→elite (red), 4→bad (primary)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_q["label"],
        y=by_q["epa"],
        marker_color=[bar_colors[int(q) - 1] for q in by_q["quality_quartile"]],
        text=[f"{e:+.2f}" for e in by_q["epa"]],
        textposition="outside",
        customdata=by_q[["n_plays", "comp_pct"]].values,
        hovertemplate="<b>%{x}</b><br>EPA/play: %{y:+.3f}<br>"
                      "Plays: %{customdata[0]}<br>"
                      "Comp %: %{customdata[1]:.1%}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0, line_color="#ccc", line_width=1)
    fig.update_layout(
        yaxis_title="EPA per dropback",
        xaxis_title="Opposing pass defense quality",
        height=340,
        margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Per-game scatter: opp D quality on x, QB EPA on y ──
    per_game = (
        plays.groupby(["game_id", "defteam", "season",
                        "pass_epa_allowed_per_play"])
        .agg(qb_epa=("epa", "mean"), n_plays=("epa", "size"))
        .reset_index()
    )
    per_game = per_game[per_game["n_plays"] >= 10]

    if len(per_game) >= 5:
        # Trendline coefficients
        import numpy as np
        x = per_game["pass_epa_allowed_per_play"].astype(float)
        y = per_game["qb_epa"].astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        # If slope > 0: as opp D gets worse (higher EPA allowed), QB EPA
        # rises → he beats up bad defenses. If slope < 0: he rises against
        # good D (the rare elite-clutch profile).
        x_range = np.linspace(x.min(), x.max(), 20)
        y_fit = slope * x_range + intercept

        scat = go.Figure()
        scat.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(size=8, color=primary, opacity=0.7,
                        line=dict(width=1, color="white")),
            text=per_game["defteam"] + " " + per_game["season"].astype(str),
            hovertemplate="<b>vs. %{text}</b><br>"
                          "Opp pass D allowed: %{x:+.3f} EPA/play<br>"
                          "QB EPA: %{y:+.3f}<extra></extra>",
            name="Game",
        ))
        scat.add_trace(go.Scatter(
            x=x_range, y=y_fit,
            mode="lines",
            line=dict(color="#888", width=2, dash="dot"),
            name="Trendline",
            hoverinfo="skip",
        ))
        scat.add_hline(y=0, line_color="#ccc", line_width=1)
        scat.add_vline(x=0, line_color="#ccc", line_width=1)
        scat.update_layout(
            xaxis_title="Opponent pass EPA allowed/play  (← elite D    bad D →)",
            yaxis_title="QB EPA in this game",
            height=380,
            margin=dict(l=40, r=20, t=30, b=50),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(scat, use_container_width=True)

        # Narrative line about slope
        if slope > 1.5:
            st.warning(
                f"**Stat-padder profile.** Slope of {slope:.2f} — "
                f"clearly feasts on bad defenses, struggles vs. elite. "
                f"The kind of QB whose box-score average outruns his "
                f"playoff readiness."
            )
        elif slope > 0.5:
            st.info(
                f"**Plays to opponent quality.** Slope of {slope:.2f} — "
                f"performs better vs. weaker defenses, as most QBs do, "
                f"but not extreme."
            )
        elif slope > -0.5:
            st.success(
                f"**Steady against any defense.** Slope of {slope:.2f} "
                f"(near zero) — plays at his level regardless of opponent. "
                f"The mark of a real dog."
            )
        else:
            st.success(
                f"**Rises against elite D.** Slope of {slope:.2f} — "
                f"counterintuitively *better* vs. top defenses. Rare "
                f"clutch profile."
            )

    # ── Detail row ──
    by_q_display = by_q.copy()
    by_q_display["EPA/play"] = by_q_display["epa"].apply(lambda v: f"{v:+.3f}")
    by_q_display["Comp %"] = by_q_display["comp_pct"].apply(lambda v: f"{v*100:.1f}%")
    by_q_display["INT rate"] = by_q_display["int_rate"].apply(lambda v: f"{v*100:.2f}%")
    by_q_display["Sack rate"] = by_q_display["sack_rate"].apply(lambda v: f"{v*100:.1f}%")
    by_q_display = by_q_display.rename(columns={"label": "Defense", "n_plays": "Plays"})
    st.dataframe(
        by_q_display[["Defense", "Plays", "EPA/play", "Comp %", "INT rate", "Sack rate"]],
        use_container_width=True, hide_index=True,
    )

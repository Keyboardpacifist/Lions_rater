"""Internal gambling-product playground.

This page is the testbed for the gambling-product engines we're building
toward launch. It is intentionally NOT polished for end-users — it's a
validation harness to confirm each engine produces sensible numbers on
historical and live data before we wire them into a public Bet School
flow.

Tabs (built incrementally):
  • Injury Cohort — Feature 1: Pr(plays Sunday) + usage retention
  • Scheme Deltas — coming soon
  • DvP — coming soon
  • Coaching Tendencies — coming soon
  • SGP Correlations — coming soon
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lib_injury_cohort import (
    body_part_normalize,
    find_comparable_cases,
    load_archive,
    load_cohort_rates,
    practice_status_code,
    predict,
    report_status_code,
)
from lib_scheme_deltas import (
    DEFENSE_METRICS,
    METRIC_LABELS,
    OFFENSE_METRICS,
    load_scheme_deltas,
    rank_teams,
    season_year_over_year,
)
from lib_shared import inject_css


st.set_page_config(
    page_title="Gambling Lab",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()


st.markdown("# 🎲 Gambling Lab")
st.caption("Internal playground for the gambling-product engines. "
           "Numbers shown here are empirical from historical data — no "
           "live odds, no real bets. This is a validation harness.")


tab_injury, tab_scheme, tab_dvp, tab_coach, tab_sgp = st.tabs([
    "🩹 Injury Cohort",
    "🧪 Scheme Deltas",
    "🛡️ DvP",
    "📋 Coaching Tendencies",
    "🔗 SGP Correlations",
])


# ════════════════════════════════════════════════════════════════
# Tab 1 — Injury Cohort
# ════════════════════════════════════════════════════════════════

with tab_injury:
    st.markdown("### Pr(plays Sunday) + usage retention")
    st.caption(
        "Given a player's Friday injury report, find historical comparable "
        "cases and report the empirical fraction who actually took a snap "
        "on Sunday — and the average snap share when they did. Cohort key: "
        "(position, body-part bucket, game-day designation, Friday practice "
        "status). Tightening: `tight → loose → fallback → marginal`."
    )

    arch = load_archive()
    rates = load_cohort_rates()

    if arch.empty or rates.empty:
        st.error(
            "Missing data. Run `python tools/pull_injuries.py` and "
            "`python tools/build_injury_cohort_rates.py` first."
        )
    else:
        col_l, col_r = st.columns([1, 1.2])

        # ── Left: cohort builder ──
        with col_l:
            st.markdown("#### Build a cohort")

            mode = st.radio(
                "Mode",
                ["Manual (pick the four keys)",
                 "From historical row (recent injury)"],
                horizontal=False,
                key="inj_mode",
            )

            if mode == "Manual (pick the four keys)":
                positions = sorted(arch["position"].dropna().astype(str)
                                   .str.upper().unique())
                body_parts = sorted(b for b in arch["_body_part_bucket"]
                                    .dropna().unique() if b)
                position = st.selectbox("Position", positions,
                                        index=positions.index("WR")
                                        if "WR" in positions else 0)
                body_part = st.selectbox("Body part", body_parts,
                                         index=body_parts.index("hamstring")
                                         if "hamstring" in body_parts else 0)
                report_status = st.selectbox(
                    "Report status (game-day designation)",
                    ["NONE", "PROBABLE", "QUESTIONABLE", "DOUBTFUL", "OUT"],
                    index=2,
                )
                practice_status = st.selectbox(
                    "Friday practice status",
                    ["NONE", "FULL", "LIMITED", "DNP"],
                    index=2,
                )

            else:
                seasons = sorted(arch["season"].dropna().astype(int).unique(),
                                 reverse=True)
                season_pick = st.selectbox("Season", seasons,
                                           index=0)
                weeks_avail = sorted(arch[arch["season"] == season_pick]
                                     ["week"].dropna().astype(int).unique())
                week_pick = st.selectbox("Week", weeks_avail,
                                         index=len(weeks_avail) - 1
                                         if weeks_avail else 0)
                slate = arch[(arch["season"] == season_pick)
                             & (arch["week"] == week_pick)
                             & (arch["report_status"].notna())]
                if slate.empty:
                    st.info("No injury report rows for that week.")
                    st.stop()
                slate = slate.assign(_label=lambda d: (
                    d["full_name"].fillna("?")
                    + "  ·  " + d["team"].astype(str)
                    + "  ·  " + d["position"].astype(str)
                    + "  ·  " + d["report_primary_injury"].fillna("?")
                    + "  ·  " + d["report_status"].fillna("None")
                    + " / " + d["practice_status"].fillna("None")
                ))
                pick = st.selectbox("Player",
                                    slate["_label"].tolist(),
                                    key="inj_player_pick")
                row = slate[slate["_label"] == pick].iloc[0]
                position = str(row["position"]).upper()
                body_part = body_part_normalize(row["report_primary_injury"])
                report_status = report_status_code(row["report_status"])
                practice_status = practice_status_code(row["practice_status"])

                st.caption(
                    f"**Resolved cohort key** — pos: `{position}` · "
                    f"body: `{body_part}` · report: `{report_status}` · "
                    f"practice: `{practice_status}`"
                )

            run_btn = st.button("Run cohort", type="primary", use_container_width=True)

        # ── Right: result ──
        with col_r:
            if run_btn or st.session_state.get("_inj_last_key"):
                key = (position, body_part, report_status, practice_status)
                st.session_state["_inj_last_key"] = key

                result = predict(
                    position=position, body_part=body_part,
                    report_status=report_status,
                    practice_status=practice_status,
                )
                cases = find_comparable_cases(
                    position=position, body_part=body_part,
                    report_status=report_status,
                    practice_status=practice_status,
                )

                # Headline metrics
                st.markdown("#### Result")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pr(plays Sunday)", f"{result.p_played:.1%}")
                m2.metric("Snap share if active", f"{result.snap_share_if_played:.0%}"
                          if result.snap_share_if_played else "—")
                m3.metric("Cohort size", f"{result.n:,}")
                m4.metric("Cohort tier", result.cohort_level.upper())

                # Interpretation
                level_blurb = {
                    "tight":    "Strong empirical signal — the cohort matches "
                                "all four keys with ≥30 historical cases.",
                    "loose":    "Aggregated across practice statuses for this "
                                "position + body + game-day designation.",
                    "fallback": "Aggregated across all designations for this "
                                "position + body part.",
                    "marginal": "No body-part-specific cohort; using the "
                                "league-wide marginal for this Friday status.",
                }.get(result.cohort_level, "")
                st.caption(level_blurb)

                # Comparable cases — show the most recent
                if not cases.empty:
                    st.markdown(f"##### {len(cases):,} comparable historical cases "
                                f"— most recent 25")
                    show = cases.sort_values(
                        ["season", "week"], ascending=[False, False]
                    ).head(25)
                    show = show[[
                        "season", "week", "team", "full_name",
                        "report_primary_injury", "report_status",
                        "practice_status",
                    ]].reset_index(drop=True)
                    st.dataframe(show, use_container_width=True, hide_index=True)
                else:
                    st.info("No exact-match historical rows. The marginal "
                            "fallback was used.")
            else:
                st.info("Pick a cohort on the left and click **Run cohort**.")

        st.markdown("---")
        st.markdown("#### Cohort table inspector")
        st.caption(
            "Browse the empirical cohort table directly. Use this to spot "
            "the high-EV cells where market pricing tends to mis-weigh."
        )
        min_n = st.slider("Min cohort size", 5, 200, 30, step=5)
        sort_col = st.radio("Sort by", ["n_cases", "play_rate"],
                            horizontal=True, index=0)
        ascending = st.checkbox("Ascending", value=False)
        top_n = st.slider("Show top N", 25, 500, 100, step=25)
        view = (rates[rates["n_cases"] >= min_n]
                .sort_values(sort_col, ascending=ascending)
                .head(top_n)
                .reset_index(drop=True))
        st.dataframe(view, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# Tab 2 — Scheme Deltas (placeholder)
# ════════════════════════════════════════════════════════════════

with tab_scheme:
    st.markdown("### Scheme Deltas")
    st.caption(
        "Per-team-season offensive and defensive scheme metrics, with a "
        "league-relative delta column. Use this to spot scheme outliers "
        "(blitz-heavy DCs, deep-shot offenses, no-huddle teams) and to "
        "track year-over-year scheme change when a coordinator turns over."
    )

    sd = load_scheme_deltas()
    if sd.empty:
        st.error(
            "Missing data. Run `python tools/build_scheme_deltas.py` first."
        )
    else:
        sub_off, sub_yoy, sub_team = st.tabs(
            ["League leaderboard", "Year-over-year", "Team season profile"]
        )

        # ── Sub-tab A: leaderboard ──
        with sub_off:
            c1, c2, c3, c4 = st.columns([1, 1, 1.6, 1])
            seasons_avail = sorted(sd["season"].unique(), reverse=True)
            season_pick = c1.selectbox("Season", seasons_avail, index=0,
                                       key="sd_season")
            side_pick = c2.selectbox("Side", ["offense", "defense"],
                                     index=0, key="sd_side")
            metrics_avail = (OFFENSE_METRICS if side_pick == "offense"
                             else DEFENSE_METRICS)
            metric_label = c3.selectbox(
                "Metric",
                [METRIC_LABELS.get(m, m) for m in metrics_avail],
                key="sd_metric_lbl",
            )
            metric_pick = next(m for m in metrics_avail
                               if METRIC_LABELS.get(m, m) == metric_label)
            top_n = c4.number_input("Top N", 5, 32, 10, key="sd_top_n")
            ascending = st.checkbox("Ascending (bottom of league)",
                                    value=False, key="sd_asc")
            ranked = rank_teams(int(season_pick), side_pick, metric_pick,
                                top_n=int(top_n), ascending=ascending)
            if ranked.empty:
                st.info("No data for this slice.")
            else:
                cols = ["team", "season", metric_pick,
                        f"{metric_pick}_delta"]
                view = ranked[cols].copy()
                # Format rate-style columns as %
                if "rate" in metric_pick or metric_pick.endswith("_pct"):
                    view[metric_pick] = view[metric_pick].apply(
                        lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                    view[f"{metric_pick}_delta"] = (
                        view[f"{metric_pick}_delta"].apply(
                            lambda x: (f"{x:+.1%}" if pd.notna(x) else "—")))
                else:
                    view[metric_pick] = view[metric_pick].round(3)
                    view[f"{metric_pick}_delta"] = view[
                        f"{metric_pick}_delta"].round(3)
                st.dataframe(view, use_container_width=True, hide_index=True)

        # ── Sub-tab B: YoY for one team ──
        with sub_yoy:
            c1, c2, c3 = st.columns([1, 1, 1.6])
            teams_avail = sorted(sd["team"].dropna().unique())
            team_pick = c1.selectbox("Team", teams_avail,
                                     index=teams_avail.index("DET")
                                     if "DET" in teams_avail else 0,
                                     key="sd_yoy_team")
            side_pick2 = c2.selectbox("Side", ["offense", "defense"],
                                       index=0, key="sd_yoy_side")
            metrics_avail2 = (OFFENSE_METRICS if side_pick2 == "offense"
                              else DEFENSE_METRICS)
            metric_lbl2 = c3.selectbox(
                "Metric",
                [METRIC_LABELS.get(m, m) for m in metrics_avail2],
                key="sd_yoy_metric_lbl",
            )
            metric_pick2 = next(m for m in metrics_avail2
                                if METRIC_LABELS.get(m, m) == metric_lbl2)
            yoy = season_year_over_year(team_pick, side_pick2, metric_pick2)
            if yoy.empty:
                st.info("No YoY data.")
            else:
                view = yoy.copy()
                if "rate" in metric_pick2 or metric_pick2.endswith("_pct"):
                    view[metric_pick2] = view[metric_pick2].apply(
                        lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                    view[f"{metric_pick2}_delta"] = view[
                        f"{metric_pick2}_delta"].apply(
                            lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
                else:
                    view[metric_pick2] = view[metric_pick2].round(3)
                    view[f"{metric_pick2}_delta"] = view[
                        f"{metric_pick2}_delta"].round(3)
                st.dataframe(view, use_container_width=True, hide_index=True)

                # Mini-chart
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yoy["season"], y=yoy[metric_pick2],
                    mode="lines+markers", name=METRIC_LABELS.get(
                        metric_pick2, metric_pick2),
                    line=dict(width=3),
                ))
                fig.update_layout(
                    height=300, margin=dict(l=20, r=20, t=20, b=20),
                    yaxis_title=METRIC_LABELS.get(metric_pick2, metric_pick2),
                    xaxis_title="Season",
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Sub-tab C: full team profile ──
        with sub_team:
            c1, c2, c3 = st.columns([1, 1, 1])
            teams_avail = sorted(sd["team"].dropna().unique())
            team_pick3 = c1.selectbox("Team", teams_avail,
                                      index=teams_avail.index("DET")
                                      if "DET" in teams_avail else 0,
                                      key="sd_team_pick")
            seasons_avail3 = sorted(sd["season"].unique(), reverse=True)
            season_pick3 = c2.selectbox("Season", seasons_avail3, index=0,
                                        key="sd_team_season")
            side_pick3 = c3.selectbox("Side", ["offense", "defense"],
                                       index=0, key="sd_team_side")
            row = sd[(sd["team"] == team_pick3)
                     & (sd["season"] == int(season_pick3))
                     & (sd["side"] == side_pick3)]
            if row.empty:
                st.info("No row for that selection.")
            else:
                row = row.iloc[0]
                metrics_use = (OFFENSE_METRICS if side_pick3 == "offense"
                               else DEFENSE_METRICS)
                profile_rows = []
                for m in metrics_use:
                    if m not in row.index:
                        continue
                    raw = row[m]
                    delta = row.get(f"{m}_delta")
                    is_rate = ("rate" in m) or m.endswith("_pct")
                    profile_rows.append({
                        "metric": METRIC_LABELS.get(m, m),
                        "value": (f"{raw:.1%}" if (is_rate and pd.notna(raw))
                                   else (round(raw, 3) if pd.notna(raw) else "—")),
                        "vs league": (f"{delta:+.1%}" if (is_rate and pd.notna(delta))
                                       else (round(delta, 3) if pd.notna(delta) else "—")),
                    })
                st.dataframe(pd.DataFrame(profile_rows),
                             use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# Tab 3 — DvP (placeholder)
# ════════════════════════════════════════════════════════════════

with tab_dvp:
    st.markdown("### Defense vs. Position (DvP)")
    st.caption(
        "Per-defense per-game allowances by position group, with a "
        "league-relative delta. Use to grade matchup softness for "
        "fantasy/prop bets — `+25 yards/game vs. WR` means this "
        "defense gives up 25 more receiving yards per game to WRs "
        "than league average."
    )

    DVP_PATH = Path(__file__).resolve().parent.parent / "data" / "dvp.parquet"
    if not DVP_PATH.exists():
        st.error(
            "Missing data. Run `python tools/pull_rosters.py` and "
            "`python tools/build_dvp.py` first."
        )
    else:
        @st.cache_data(show_spinner=False)
        def _load_dvp() -> pd.DataFrame:
            return pd.read_parquet(DVP_PATH)
        dvp = _load_dvp()

        DVP_METRICS = {
            "rec_yards_pg":      "Rec yards allowed / game",
            "rec_targets_pg":    "Targets allowed / game",
            "rec_completions_pg":"Receptions allowed / game",
            "rec_tds_pg":        "Rec TDs allowed / game",
            "rec_air_yards_pg":  "Air yards allowed / game",
            "rec_yac_pg":        "YAC allowed / game",
            "rush_yards_pg":     "Rush yards allowed / game",
            "rush_attempts_pg":  "Rush attempts allowed / game",
            "rush_tds_pg":       "Rush TDs allowed / game",
        }

        c1, c2, c3, c4 = st.columns([1, 1, 1.6, 1])
        seasons_avail = sorted(dvp["season"].unique(), reverse=True)
        season_pick = c1.selectbox("Season", seasons_avail, index=0,
                                   key="dvp_season")
        pos_avail = sorted(dvp["pos_group"].dropna().unique())
        pos_pick = c2.selectbox("Position group", pos_avail,
                                 index=pos_avail.index("WR")
                                 if "WR" in pos_avail else 0,
                                 key="dvp_pos")
        # Filter metrics to receiving for WR/TE, both for RB
        if pos_pick == "RB":
            metric_keys = list(DVP_METRICS.keys())
        else:
            metric_keys = [k for k in DVP_METRICS.keys() if k.startswith("rec_")]
        metric_lbl = c3.selectbox(
            "Metric", [DVP_METRICS[k] for k in metric_keys],
            key="dvp_metric_lbl",
        )
        metric_pick = next(k for k in metric_keys
                           if DVP_METRICS[k] == metric_lbl)
        ascending = c4.checkbox("Ascending (toughest D)", value=False,
                                 key="dvp_asc")

        sub = (dvp[(dvp["season"] == int(season_pick))
                   & (dvp["pos_group"] == pos_pick)]
               .sort_values(metric_pick, ascending=ascending)
               .reset_index(drop=True))
        if sub.empty:
            st.info("No data for this slice.")
        else:
            view = sub[["defteam", "games", metric_pick,
                        f"{metric_pick}_delta"]].copy()
            view[metric_pick] = view[metric_pick].round(2)
            view[f"{metric_pick}_delta"] = view[
                f"{metric_pick}_delta"].apply(
                    lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
            view.columns = ["Defense", "Games", DVP_METRICS[metric_pick],
                            "vs league"]
            st.dataframe(view, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Single-team profile")
        c1, c2 = st.columns([1, 1])
        teams_avail = sorted(dvp["defteam"].dropna().unique())
        team_pick = c1.selectbox("Defense", teams_avail,
                                  index=teams_avail.index("DET")
                                  if "DET" in teams_avail else 0,
                                  key="dvp_team")
        season_pick2 = c2.selectbox("Season", seasons_avail, index=0,
                                     key="dvp_team_season")
        team_rows = dvp[(dvp["defteam"] == team_pick)
                        & (dvp["season"] == int(season_pick2))]
        if team_rows.empty:
            st.info("No row for that selection.")
        else:
            profile_rows = []
            for _, r in team_rows.iterrows():
                pg = r["pos_group"]
                metric_keys2 = (list(DVP_METRICS.keys()) if pg == "RB"
                                else [k for k in DVP_METRICS
                                      if k.startswith("rec_")])
                for m in metric_keys2:
                    if m not in r.index or pd.isna(r[m]):
                        continue
                    profile_rows.append({
                        "Position": pg,
                        "Metric": DVP_METRICS[m],
                        "Per game": round(r[m], 2),
                        "vs league": (f"{r[f'{m}_delta']:+.2f}"
                                       if pd.notna(r[f"{m}_delta"]) else "—"),
                    })
            st.dataframe(pd.DataFrame(profile_rows),
                         use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# Tab 4 — Coaching Tendencies (placeholder)
# ════════════════════════════════════════════════════════════════

with tab_coach:
    st.markdown("### Coaching / Play-Caller Tendencies")
    st.caption(
        "Game-script behavior and aggression metrics — the decisions "
        "that actually move lines. Sharp markets price 4th-down aggression "
        "and 2-point conversion rates poorly; this is where in-game live "
        "betting EV lives."
    )

    COACH_PATH = (Path(__file__).resolve().parent.parent
                  / "data" / "coaching_tendencies.parquet")
    if not COACH_PATH.exists():
        st.error(
            "Missing data. Run `python tools/build_coaching_tendencies.py` first."
        )
    else:
        @st.cache_data(show_spinner=False)
        def _load_coach() -> pd.DataFrame:
            return pd.read_parquet(COACH_PATH)
        coach = _load_coach()

        COACH_METRICS = {
            "fourth_short_go_rate":  "4th-and-short (≤2) go rate",
            "fourth_long_go_rate":   "4th-and-long (3+) go rate",
            "two_pt_attempt_rate":   "2-point conversion rate after TD",
            "pass_rate_leading_7p":  "Pass rate when leading by 7+ (Q1-3)",
            "pass_rate_trailing_7p": "Pass rate when trailing by 7+ (Q1-3)",
            "pass_rate_q4_trailing": "Pass rate in Q4 when trailing",
            "run_rate_q4_leading":   "Run rate in Q4 when leading 7+",
            "two_min_drill_plays_pg":"2-min drill plays / game",
            "rz_run_rate":           "Red-zone run rate",
        }

        c1, c2, c3 = st.columns([1, 1.6, 1])
        seasons_avail = sorted(coach["season"].unique(), reverse=True)
        season_pick = c1.selectbox("Season", seasons_avail, index=0,
                                    key="coach_season")
        metric_lbl = c2.selectbox(
            "Metric",
            list(COACH_METRICS.values()),
            index=0,
            key="coach_metric",
        )
        metric_pick = next(k for k, v in COACH_METRICS.items()
                           if v == metric_lbl)
        ascending = c3.checkbox("Ascending", value=False, key="coach_asc")

        sub = (coach[coach["season"] == int(season_pick)]
               .sort_values(metric_pick, ascending=ascending)
               .reset_index(drop=True))
        if sub.empty:
            st.info("No data for this slice.")
        else:
            view = sub[["team", metric_pick, f"{metric_pick}_delta"]].copy()
            is_rate = "rate" in metric_pick
            if is_rate:
                view[metric_pick] = view[metric_pick].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                view[f"{metric_pick}_delta"] = view[
                    f"{metric_pick}_delta"].apply(
                        lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
            else:
                view[metric_pick] = view[metric_pick].round(2)
                view[f"{metric_pick}_delta"] = view[
                    f"{metric_pick}_delta"].apply(
                        lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
            view.columns = ["Team", COACH_METRICS[metric_pick], "vs league"]
            st.dataframe(view, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Single team — full coaching profile")
        c1, c2 = st.columns([1, 1])
        teams_avail = sorted(coach["team"].dropna().unique())
        team_pick = c1.selectbox("Team", teams_avail,
                                  index=teams_avail.index("DET")
                                  if "DET" in teams_avail else 0,
                                  key="coach_team")
        season_pick2 = c2.selectbox("Season", seasons_avail, index=0,
                                     key="coach_team_season")
        team_row = coach[(coach["team"] == team_pick)
                         & (coach["season"] == int(season_pick2))]
        if team_row.empty:
            st.info("No row for that selection.")
        else:
            r = team_row.iloc[0]
            profile_rows = []
            for m, label in COACH_METRICS.items():
                val = r[m]
                delta = r[f"{m}_delta"]
                is_rate = "rate" in m
                profile_rows.append({
                    "Metric": label,
                    "Value": (f"{val:.1%}" if (is_rate and pd.notna(val))
                               else (round(val, 2) if pd.notna(val) else "—")),
                    "vs league": (f"{delta:+.1%}" if (is_rate and pd.notna(delta))
                                   else (round(delta, 2) if pd.notna(delta) else "—")),
                })
            st.dataframe(pd.DataFrame(profile_rows),
                         use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# Tab 5 — SGP Correlations (placeholder)
# ════════════════════════════════════════════════════════════════

with tab_sgp:
    st.markdown("### Same-Game-Parlay Correlations")
    st.caption(
        "Per-team-season correlation between QB passing yards and each "
        "primary partner's stats. Sportsbook SGP engines often assume "
        "independence; positive **lift** means the parlay is "
        "under-priced — that's where EV lives."
    )
    st.markdown(
        "**Lift** = `P(partner ≥ 75 yds | QB ≥ 250 yds)` − "
        "`P(partner ≥ 75 yds)`. Positive lift = the partner hits 75+ more "
        "often than baseline when the QB has a big passing day."
    )

    SGP_PATH = (Path(__file__).resolve().parent.parent
                / "data" / "sgp_correlations.parquet")
    if not SGP_PATH.exists():
        st.error(
            "Missing data. Run `python tools/pull_player_stats.py` and "
            "`python tools/build_sgp_correlations.py` first."
        )
    else:
        @st.cache_data(show_spinner=False)
        def _load_sgp() -> pd.DataFrame:
            return pd.read_parquet(SGP_PATH)
        sgp = _load_sgp()

        sub_a, sub_b = st.tabs(["League leaderboard", "Team profile"])

        with sub_a:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            seasons_avail = sorted(sgp["season"].unique(), reverse=True)
            season_pick = c1.selectbox("Season", seasons_avail, index=0,
                                        key="sgp_season")
            roles_avail = ["WR1", "WR2", "WR3", "TE1", "RB1"]
            role_pick = c2.selectbox("Partner role", roles_avail,
                                      index=0, key="sgp_role")
            min_n = c3.number_input("Min games together", 4, 17, 8,
                                     key="sgp_min_n")
            sort_by = c4.selectbox(
                "Sort by",
                ["lift_partner_75_given_qb_300",
                 "corr_qb_yds_partner_yds"],
                key="sgp_sort",
            )
            sub = (sgp[(sgp["season"] == int(season_pick))
                       & (sgp["partner_role"] == role_pick)
                       & (sgp["n_games_both"] >= int(min_n))]
                   .sort_values(sort_by, ascending=False)
                   .reset_index(drop=True))
            if sub.empty:
                st.info("No data for this slice.")
            else:
                view = sub[["team", "qb_name", "partner_name",
                            "n_games_both",
                            "corr_qb_yds_partner_yds",
                            "qb_yds_p_300", "partner_yds_p_75",
                            "partner_yds_p_75_given_qb_300",
                            "lift_partner_75_given_qb_300"]].copy()
                for col in ("corr_qb_yds_partner_yds",):
                    view[col] = view[col].round(2)
                for col in ("qb_yds_p_300", "partner_yds_p_75",
                            "partner_yds_p_75_given_qb_300",
                            "lift_partner_75_given_qb_300"):
                    view[col] = view[col].apply(
                        lambda x: f"{x:.0%}" if pd.notna(x) else "—")
                view.columns = ["Team", "QB", "Partner", "Games",
                                "Corr (yds)",
                                "QB 300+ rate", "Partner 75+ rate",
                                "Partner 75+ | QB 300+",
                                "Lift"]
                st.dataframe(view, use_container_width=True, hide_index=True)

        with sub_b:
            c1, c2 = st.columns([1, 1])
            teams_avail = sorted(sgp["team"].dropna().unique())
            team_pick = c1.selectbox("Team", teams_avail,
                                      index=teams_avail.index("DET")
                                      if "DET" in teams_avail else 0,
                                      key="sgp_team")
            seasons_avail2 = sorted(sgp["season"].unique(), reverse=True)
            season_pick2 = c2.selectbox("Season", seasons_avail2, index=0,
                                         key="sgp_team_season")
            team_rows = sgp[(sgp["team"] == team_pick)
                            & (sgp["season"] == int(season_pick2))]
            if team_rows.empty:
                st.info("No row for that selection.")
            else:
                view = team_rows[[
                    "partner_role", "qb_name", "partner_name",
                    "n_games_both",
                    "corr_qb_yds_partner_yds",
                    "qb_yds_p_300", "partner_yds_p_75",
                    "partner_yds_p_75_given_qb_300",
                    "lift_partner_75_given_qb_300",
                ]].copy()
                view["corr_qb_yds_partner_yds"] = view[
                    "corr_qb_yds_partner_yds"].round(2)
                for col in ("qb_yds_p_300", "partner_yds_p_75",
                            "partner_yds_p_75_given_qb_300",
                            "lift_partner_75_given_qb_300"):
                    view[col] = view[col].apply(
                        lambda x: f"{x:.0%}" if pd.notna(x) else "—")
                view.columns = ["Role", "QB", "Partner", "Games",
                                "Corr (yds)",
                                "QB 300+ rate", "Partner 75+ rate",
                                "Partner 75+ | QB 300+",
                                "Lift"]
                st.dataframe(view, use_container_width=True, hide_index=True)

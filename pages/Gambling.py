"""Internal gambling-product playground.

Validation harness for the engines we're building toward launch. Not
polished for end-users — every tab is a data inspector that lets us
confirm each engine produces sensible numbers before we wire it into
the public Bet School flow.

Game-bet engines (4.x):
  • 4.1 Injury Cohort — Pr(plays Sunday) + usage retention
  • 4.2 Game-Script Simulator — team scoring/scheme delta when starter out
  • 4.3 Books vs Model — historical line miss by injury cohort
  • 4.4 Smart Alerts — fusion of all four into one bet-actionable update
  • 4.5 Weather Window — empirical P10/P50/P90 by weather conditions
  • 4.6 Scheme Deltas + Coaching Tendencies (foundation tables)

Prop-bet engines (5.x):
  • 5.2 SGP Correlations
  • 5.8 DvP
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
from lib_smart_alerts import fuse_alert
from lib_weather import (
    all_player_options,
    confidence_for_n,
    primary_stat_for_position,
    weather_cohort,
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


(tab_alerts, tab_injury, tab_gscript, tab_books, tab_weather,
 tab_scheme, tab_dvp, tab_coach, tab_sgp) = st.tabs([
    "⭐ Smart Alerts (4.4)",
    "🩹 Injury Cohort (4.1)",
    "🎯 Game-Script (4.2)",
    "📊 Books vs Model (4.3)",
    "🌧️ Weather Window (4.5)",
    "🧪 Scheme Deltas (4.6)",
    "🛡️ DvP (5.8)",
    "📋 Coaching Tendencies",
    "🔗 SGP Correlations (5.2)",
])


# ════════════════════════════════════════════════════════════════
# Tab — Smart Alerts (Feature 4.4) — showcase tab
# ════════════════════════════════════════════════════════════════

with tab_alerts:
    st.markdown("### Smart Alerts — fusion engine")
    st.caption(
        "Drop in a single news event (player + status + body part) and "
        "the engine fuses every other feature: cohort play probability "
        "(4.1), team scoring shift (4.2), book over/under-reaction "
        "history (4.3), and weather context (4.5). This is what a "
        "Bet School push notification will read like in production."
    )

    arch = load_archive()
    if arch.empty:
        st.error("Missing data — pull injuries first.")
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        seasons = sorted(arch["season"].dropna().astype(int).unique(),
                         reverse=True)
        season_pick = c1.selectbox("Season", seasons, index=0,
                                    key="al_season")
        weeks = sorted(arch[arch["season"] == season_pick]
                       ["week"].dropna().astype(int).unique())
        week_pick = c2.selectbox("Week", weeks,
                                  index=len(weeks) - 1 if weeks else 0,
                                  key="al_week")

        slate = arch[(arch["season"] == season_pick)
                     & (arch["week"] == week_pick)
                     & (arch["report_status"].notna())].copy()
        if slate.empty:
            st.info("No injury report rows for that week.")
        else:
            slate["_label"] = (
                slate["full_name"].fillna("?")
                + "  ·  " + slate["team"].astype(str)
                + "  ·  " + slate["position"].astype(str)
                + "  ·  " + slate["report_primary_injury"].fillna("?")
                + "  ·  " + slate["report_status"].fillna("None")
                + " / " + slate["practice_status"].fillna("None")
            )
            pick = c3.selectbox("Player", slate["_label"].tolist(),
                                 key="al_pick")
            row = slate[slate["_label"] == pick].iloc[0]

            if st.button("Generate alert", type="primary",
                         use_container_width=True, key="al_run"):
                bundle = fuse_alert(
                    player_name=row["full_name"],
                    team=str(row["team"]),
                    position=str(row["position"]),
                    status=report_status_code(row["report_status"]),
                    body_part=body_part_normalize(
                        row["report_primary_injury"]),
                    practice_status=practice_status_code(
                        row["practice_status"]),
                    season=int(row["season"]),
                    week=int(row["week"]),
                )
                st.markdown(f"### {bundle.headline}")
                st.markdown("---")
                for b in bundle.bullet_points:
                    st.markdown(f"- {b}")
                st.markdown("---")
                st.markdown("**Cohort (4.1):** " + bundle.cohort_line)
                st.markdown("**Game-script (4.2):** "
                            + bundle.game_script_line)
                st.markdown("**Book behavior (4.3):** "
                            + bundle.book_behavior_line)
                if bundle.weather_line:
                    st.markdown("**Weather (4.5):** "
                                + bundle.weather_line)


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
# Tab — Game-Script Simulator (Feature 4.2)
# ════════════════════════════════════════════════════════════════

with tab_gscript:
    st.markdown("### Game-Script Simulator")
    st.caption(
        "League-wide game-script shifts when a key starter is out. "
        "Built from 2013+ pbp + snap counts — for every team-game, we "
        "classify which key starters (QB1/RB1/WR1/TE1) were active vs. "
        "missing, then compute scheme + scoring deltas vs. the 'all "
        "starters playing' baseline."
    )

    GS_PATH = (Path(__file__).resolve().parent.parent
               / "data" / "game_script_deltas.parquet")
    if not GS_PATH.exists():
        st.error("Missing data. Run `python tools/build_game_script_deltas.py` first.")
    else:
        @st.cache_data(show_spinner=False)
        def _load_gs() -> pd.DataFrame:
            return pd.read_parquet(GS_PATH)
        gs = _load_gs()

        view = gs.copy()
        view = view[view["scenario"].isin(["NONE", "QB1", "RB1", "WR1",
                                            "TE1", "MULTI"])]
        # Reorder so NONE (baseline) is first
        order = {"NONE": 0, "QB1": 1, "RB1": 2, "WR1": 3,
                 "TE1": 4, "MULTI": 5}
        view = view.sort_values("scenario", key=lambda s: s.map(order))

        # Friendly column names + percent formatting
        display = view.copy()
        display["scenario"] = display["scenario"].replace({
            "NONE": "All starters playing (baseline)",
            "QB1": "QB1 OUT",
            "RB1": "RB1 OUT",
            "WR1": "WR1 OUT",
            "TE1": "TE1 OUT",
            "MULTI": "Multiple starters OUT",
        })
        for col in ("pass_rate", "early_down_pass_rate",
                    "shotgun_rate", "no_huddle_rate"):
            display[col] = display[col].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—")
            display[f"{col}_delta"] = display[f"{col}_delta"].apply(
                lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
        for col in ("plays_per_game", "points_per_game"):
            display[col] = display[col].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "—")
            display[f"{col}_delta"] = display[f"{col}_delta"].apply(
                lambda x: f"{x:+.1f}" if pd.notna(x) else "—")

        display = display[[
            "scenario", "n_games", "points_per_game",
            "points_per_game_delta", "plays_per_game",
            "plays_per_game_delta", "pass_rate", "pass_rate_delta",
            "early_down_pass_rate_delta", "shotgun_rate_delta",
            "no_huddle_rate_delta",
        ]]
        display.columns = ["Scenario", "n", "Pts/G", "Δ Pts",
                           "Plays/G", "Δ Plays", "Pass rate",
                           "Δ Pass rate", "Δ Early pass",
                           "Δ Shotgun", "Δ No-huddle"]
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.info(
            "**Read it like this.** When a team's QB1 is out, league-wide "
            "they score ~4.1 fewer points per game. Losing a WR1 / TE1 / "
            "RB1 is nearly zero scoring impact (committee absorbs). "
            "RB1 out → pass rate +2.2% (committee leans pass-heavier). "
            "Multi-starter outages scale closer to QB1-out impact."
        )


# ════════════════════════════════════════════════════════════════
# Tab — Books vs Model (Feature 4.3)
# ════════════════════════════════════════════════════════════════

with tab_books:
    st.markdown("### Books vs Model — Behavioral Baseline")
    st.caption(
        "For each historical (role_lost, status, body_part) cohort: how "
        "did the affected team perform vs. the closing spread? **Mean "
        "line miss > 0** = team beat the close = books OVER-reacted "
        "(contrarian: bet ON the affected team). **Mean line miss < 0** "
        "= team underperformed = books UNDER-reacted (fade the team)."
    )

    BV_PATH = (Path(__file__).resolve().parent.parent
               / "data" / "books_vs_model.parquet")
    if not BV_PATH.exists():
        st.error("Missing data. Run `python tools/build_books_vs_model.py` first.")
    else:
        @st.cache_data(show_spinner=False)
        def _load_bv() -> pd.DataFrame:
            return pd.read_parquet(BV_PATH)
        bv = _load_bv()

        c1, c2, c3 = st.columns([1, 1, 1])
        roles = sorted(bv["position_lost"].dropna().unique())
        role_pick = c1.selectbox("Role lost", roles,
                                  index=roles.index("QB")
                                  if "QB" in roles else 0,
                                  key="bv_role")
        statuses = sorted(bv[bv["position_lost"] == role_pick]
                           ["status"].dropna().unique())
        status_pick = c2.selectbox("Status", statuses,
                                    index=statuses.index("OUT")
                                    if "OUT" in statuses else 0,
                                    key="bv_status")
        min_n = c3.number_input("Min n", 5, 200, 15, key="bv_min_n")

        sub = (bv[(bv["position_lost"] == role_pick)
                  & (bv["status"] == status_pick)
                  & (bv["n_games"] >= int(min_n))]
               .sort_values("mean_line_miss")
               .reset_index(drop=True))
        if sub.empty:
            st.info("No cohorts at that filter.")
        else:
            display = sub[["body_part", "n_games", "mean_line_miss",
                           "cover_rate", "mean_total_miss",
                           "median_actual_margin", "median_spread"]].copy()
            display["mean_line_miss"] = display["mean_line_miss"].apply(
                lambda x: f"{x:+.1f}" if pd.notna(x) else "—")
            display["cover_rate"] = display["cover_rate"].apply(
                lambda x: f"{x:.0%}" if pd.notna(x) else "—")
            display["mean_total_miss"] = display["mean_total_miss"].apply(
                lambda x: f"{x:+.1f}" if pd.notna(x) else "—")
            display.columns = ["Body part", "n", "Mean line miss",
                               "Cover rate", "Total miss",
                               "Median margin", "Median spread"]
            st.dataframe(display, use_container_width=True, hide_index=True)

            st.markdown("**Interpretation key**")
            st.markdown(
                "- `mean line miss > +1.5` → team beat the close → books **OVER**-reacted → **bet ON** affected team"
            )
            st.markdown(
                "- `mean line miss < -1.5` → team underperformed → books **UNDER**-reacted → **fade** affected team"
            )


# ════════════════════════════════════════════════════════════════
# Tab — Weather Production Window (Feature 4.5)
# ════════════════════════════════════════════════════════════════

with tab_weather:
    st.markdown("### Weather Production Window")
    st.caption(
        "For any player, find his historical games matching target "
        "weather conditions and return the empirical P10 / P50 / P90 "
        "of his primary production stat (passing/rushing/receiving "
        "yards). Confidence labels: HIGH (15+ games) / MEDIUM (6–14) / "
        "LOW (<5, falls back to player baseline)."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        position_pick = st.selectbox(
            "Position", ["QB", "WR", "RB", "TE"], index=0,
            key="w_pos",
        )
        opts = all_player_options(position=position_pick, min_games=20)
        if opts.empty:
            st.info("No players found at this position.")
            st.stop()
        opts["_label"] = (opts["player_display_name"]
                          + " (" + opts["n_games"].astype(str)
                          + " games)")
        player_label = st.selectbox(
            "Player", opts["_label"].tolist(),
            key="w_player",
        )
        chosen = opts[opts["_label"] == player_label].iloc[0]

        st.markdown("**Target weather**")
        target_temp = st.slider("Temperature (°F)", -5, 100, 50,
                                key="w_temp")
        target_wind = st.slider("Wind (mph)", 0, 35, 5, key="w_wind")
        target_roof = st.selectbox(
            "Roof", ["any", "outdoors", "dome", "closed", "open"],
            index=0, key="w_roof",
        )
        target_surface = st.selectbox(
            "Surface", ["any", "grass", "turf", "fieldturf",
                        "a_turf", "sportturf"],
            index=0, key="w_surf",
        )
        run_w = st.button("Run weather cohort", type="primary",
                          use_container_width=True, key="w_run")

    with c2:
        if run_w:
            r = weather_cohort(
                player_id=chosen["player_id"],
                position=position_pick,
                target_temp=target_temp,
                target_wind=target_wind,
                target_roof=None if target_roof == "any" else target_roof,
                target_surface=None if target_surface == "any"
                                else target_surface,
            )

            st.markdown(f"#### {chosen['player_display_name']} — {r.stat}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P10", f"{r.p10:.0f}")
            m2.metric("P50 (median)", f"{r.p50:.0f}")
            m3.metric("P90", f"{r.p90:.0f}")
            m4.metric("Mean", f"{r.mean:.0f}")

            m1, m2 = st.columns(2)
            m1.metric("Cohort size", f"{r.n_games} games")
            m2.metric("Confidence", r.confidence)

            mode_blurb = {
                "player": "Cohort drawn from this player's own historical "
                          "games matching the target conditions.",
                "tier_blend": "Player's specific weather cohort was thin "
                              "(<5 games). Falling back to all of his "
                              "games as the baseline.",
                "league": "No matching games at all. Showing league-wide "
                          "distribution at this position.",
            }.get(r.cohort_mode, "")
            st.caption(mode_blurb)
        else:
            st.info("Pick a player and target weather, then click **Run**.")


# ════════════════════════════════════════════════════════════════
# Tab — Scheme Deltas (4.6 foundation)
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

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
from lib_alt_line_ev import (
    american_to_decimal,
    decimal_to_implied_prob,
    p_over_threshold,
    player_distribution,
    rank_ladder,
)
from lib_decomposed_projection import decompose
from lib_longest_play import (
    p_longest_at_least,
    longest_play_distribution,
    player_options as longest_player_options,
)
from lib_sgp_pricing import Leg, sgp_price
from lib_smart_parlay import detect_anti_correlated, score_parlay
from lib_td_probability import (
    player_td_rates,
    rz_usage_share,
    td_probability_vector,
)
from lib_trend_divergence import (
    USAGE_STATS,
    league_divergence_today,
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


section_game, section_props = st.tabs([
    "🏟️  GAME BETS  —  spread / total / moneyline",
    "👤  PLAYER PROPS  —  yards / TDs / receptions / parlays",
])

with section_game:
    (tab_alerts, tab_injury, tab_gscript, tab_books, tab_weather,
     tab_scheme, tab_coach) = st.tabs([
        "⭐ Smart Alerts (4.4)",
        "🩹 Injury Cohort (4.1)",
        "🎯 Game-Script (4.2)",
        "📊 Books vs Model (4.3)",
        "🌧️ Weather Window (4.5)",
        "🧪 Scheme Deltas (4.6)",
        "📋 Coaching Tendencies",
    ])

with section_props:
    (tab_decomp, tab_sgp, tab_alt, tab_parlay, tab_td, tab_trend,
     tab_long, tab_dvp) = st.tabs([
        "🔬 Decomposed Projection (5.1)",
        "🔗 SGP Correlations (5.2)",
        "🎲 Alt-Line EV (5.3)",
        "🎰 Smart Parlay (5.4)",
        "🎯 Anytime / First TD (5.5)",
        "📈 Trend Divergence (5.6)",
        "💥 Longest-Play Edge (5.7)",
        "🛡️ DvP (5.8)",
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


# ════════════════════════════════════════════════════════════════
# Tab — Decomposed Projection (Feature 5.1)
# ════════════════════════════════════════════════════════════════

with tab_decomp:
    st.markdown("### Decomposed Prop Projection")
    st.caption(
        "**The transparency feature.** Pick a player and a stat, "
        "configure the matchup context, and the engine returns a "
        "row-by-row decomposition: baseline + each adjustment "
        "(injury / weather / matchup / game-script) shown as its own "
        "labeled contribution. Every yard is auditable."
    )

    @st.cache_data(show_spinner=False)
    def _decomp_player_options(position: str) -> pd.DataFrame:
        df = pd.read_parquet(
            Path(__file__).resolve().parent.parent
            / "data" / "nfl_player_stats_weekly.parquet"
        )
        recent = df[df["season"] >= 2023]
        if position:
            recent = recent[recent["position"] == position]
        opts = (recent.groupby(["player_id", "player_display_name",
                                 "position", "team"])
                .size().reset_index().rename(columns={0: "n_games"}))
        opts = opts[opts["n_games"] >= 6]
        return opts.sort_values("n_games", ascending=False).reset_index(drop=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        pos_pick = st.selectbox("Position", ["QB", "WR", "TE", "RB"],
                                 key="dec_pos")
        stat_choices = {
            "QB": ["passing_yards"],
            "WR": ["receiving_yards"],
            "TE": ["receiving_yards"],
            "RB": ["rushing_yards", "receiving_yards"],
        }
        stat_pick = st.selectbox("Stat", stat_choices[pos_pick],
                                  key="dec_stat")
        opts = _decomp_player_options(pos_pick)
        opts["_label"] = (opts["player_display_name"]
                          + " · " + opts["team"]
                          + f" ({opts['n_games']} games)")
        player_label = st.selectbox("Player",
                                     opts["_label"].tolist(),
                                     key="dec_player")
        if player_label:
            chosen = opts[opts["_label"] == player_label].iloc[0]

            st.markdown("**Matchup context**")
            opp_pick = st.text_input("Opponent (e.g. HOU)",
                                       key="dec_opp", value="")
            season_pick = st.number_input("Season", 2016, 2025, 2024,
                                            key="dec_season")
            week_pick = st.number_input("Week", 1, 22, 10,
                                          key="dec_week")

            st.markdown("**Injury status (optional)**")
            inj_status = st.selectbox(
                "Status", ["NONE", "PROBABLE", "QUESTIONABLE",
                          "DOUBTFUL", "OUT"],
                key="dec_status",
            )
            inj_body = st.text_input("Body part",
                                      key="dec_body", value="unknown")
            inj_practice = st.selectbox(
                "Practice", ["FULL", "LIMITED", "DNP"],
                key="dec_practice",
            )

            st.markdown("**Weather (optional)**")
            use_weather = st.checkbox("Apply weather adjustment",
                                       key="dec_useweather")
            if use_weather:
                w_temp = st.slider("Temp (°F)", -5, 100, 50,
                                    key="dec_temp")
                w_wind = st.slider("Wind (mph)", 0, 35, 5,
                                    key="dec_wind")
            else:
                w_temp = w_wind = None

            st.markdown("**Game-script (optional)**")
            starter_out = st.selectbox(
                "Key starter out",
                ["none", "QB1", "RB1", "WR1", "TE1", "MULTI"],
                key="dec_starter",
            )

            run_d = st.button("Run decomposition", type="primary",
                              use_container_width=True, key="dec_run")

    with c2:
        if c1 and "dec_run" in st.session_state and \
                st.session_state.get("dec_run"):
            d = decompose(
                player_id=chosen["player_id"],
                position=pos_pick,
                team=str(chosen["team"]),
                stat=stat_pick,
                opponent=(opp_pick.upper() if opp_pick else None),
                season=int(season_pick),
                week=int(week_pick),
                injury_status=(inj_status if inj_status != "NONE" else None),
                injury_body_part=inj_body,
                injury_practice=inj_practice,
                key_starter_out=(starter_out if starter_out != "none"
                                  else None),
                target_temp=w_temp, target_wind=w_wind,
            )
            st.markdown(f"#### {d.player_display_name} — {d.stat}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Baseline (median)", f"{d.baseline:.1f}")
            m2.metric("Adjustments",
                       f"{sum(c.delta for c in d.contributions):+.1f}")
            m3.metric("Projection", f"{d.projection:.1f}")

            if d.contributions:
                rows = [{"Adjustment": c.label,
                         "Δ yards": f"{c.delta:+.1f}",
                         "Note": c.note} for c in d.contributions]
                st.dataframe(pd.DataFrame(rows),
                              use_container_width=True, hide_index=True)
            else:
                st.info("No adjustments applied — projection equals baseline.")

            st.markdown("**Compare to a book line**")
            book_line = st.number_input(
                "Book line", 0.0, 500.0, float(round(d.baseline)),
                step=0.5, key="dec_book_line",
            )
            edge_yards = d.projection - book_line
            verdict = ("📈 LEAN OVER" if edge_yards > 3
                        else "📉 LEAN UNDER" if edge_yards < -3
                        else "≈ pass — within 3 yds of model")
            st.metric("Model − book line",
                       f"{edge_yards:+.1f} yds",
                       delta=verdict)
        else:
            st.info("Pick a player on the left and click "
                     "**Run decomposition**.")


# ════════════════════════════════════════════════════════════════
# Tab — Alt-Line EV Finder (Feature 5.3)
# ════════════════════════════════════════════════════════════════

with tab_alt:
    st.markdown("### Alt-Line EV Finder")
    st.caption(
        "Build an alt-line ladder for any player+stat, paste in the "
        "American odds at each rung, and the engine returns each rung "
        "ranked by EV. Most bettors leave 5-15% of EV on the table by "
        "reflexively betting main lines — this finds the rung where the "
        "book is most wrong."
    )

    @st.cache_data(show_spinner=False)
    def _alt_player_options(position: str) -> pd.DataFrame:
        df = pd.read_parquet(
            Path(__file__).resolve().parent.parent
            / "data" / "nfl_player_stats_weekly.parquet"
        )
        recent = df[df["season"] >= 2022]
        if position:
            recent = recent[recent["position"] == position]
        return (recent.groupby(["player_id", "player_display_name",
                                 "position", "team"])
                .size().reset_index().rename(columns={0: "n_games"})
                .sort_values("n_games", ascending=False)
                .reset_index(drop=True))

    c1, c2 = st.columns([1, 2])
    with c1:
        a_pos = st.selectbox("Position", ["QB", "WR", "TE", "RB"],
                              key="alt_pos")
        a_stat_options = {
            "QB": ["passing_yards", "passing_tds", "completions"],
            "WR": ["receiving_yards", "receptions", "targets"],
            "TE": ["receiving_yards", "receptions", "targets"],
            "RB": ["rushing_yards", "carries", "receiving_yards",
                   "receptions"],
        }
        a_stat = st.selectbox("Stat", a_stat_options[a_pos],
                              key="alt_stat")
        a_opts = _alt_player_options(a_pos)
        a_opts["_label"] = (a_opts["player_display_name"]
                            + " · " + a_opts["team"]
                            + f" ({a_opts['n_games']}g)")
        a_player = st.selectbox("Player", a_opts["_label"].tolist(),
                                 key="alt_player")
        a_lookback = st.slider("Lookback (games)", 5, 50, 20,
                                key="alt_lookback")
        st.markdown(
            "**Ladder (one rung per row)** — `threshold,side,odds`. "
            "Side = `over` or `under`."
        )
        default_ladder = "65.5,over,-150\n75.5,over,-110\n85.5,over,+120\n95.5,over,+180\n105.5,over,+260\n75.5,under,-110"
        ladder_text = st.text_area(
            "Ladder", value=default_ladder, height=180, key="alt_ladder",
        )
        run_a = st.button("Score ladder", type="primary",
                           use_container_width=True, key="alt_run")

    with c2:
        if run_a:
            chosen = a_opts[a_opts["_label"] == a_player].iloc[0]
            ladder_rungs = []
            for line in ladder_text.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3:
                    continue
                try:
                    threshold = float(parts[0])
                    side = parts[1]
                    odds = int(parts[2])
                    ladder_rungs.append((threshold, side, odds))
                except ValueError:
                    continue
            if not ladder_rungs:
                st.warning("Couldn't parse the ladder.")
            else:
                df = rank_ladder(chosen["player_id"], a_stat,
                                  ladder_rungs,
                                  lookback_games=a_lookback)
                if df.empty:
                    st.info("No valid rungs after parsing.")
                else:
                    view = df.copy()
                    view["p_model"] = view["p_model"].apply(
                        lambda x: f"{x:.1%}")
                    view["p_implied"] = view["p_implied"].apply(
                        lambda x: f"{x:.1%}")
                    view["edge"] = view["edge"].apply(
                        lambda x: f"{x:+.1%}")
                    view["ev"] = view["ev"].apply(lambda x: f"{x:+.1%}")
                    view["decimal_odds"] = view["decimal_odds"].round(2)
                    view = view[["threshold", "side", "american_odds",
                                  "decimal_odds", "p_model", "p_implied",
                                  "edge", "ev", "n_games"]]
                    view.columns = ["Line", "Side", "Odds", "Decimal",
                                     "Model P", "Implied P", "Edge",
                                     "EV", "n"]
                    st.dataframe(view, use_container_width=True,
                                  hide_index=True)


# ════════════════════════════════════════════════════════════════
# Tab — Smart Parlay Builder (Feature 5.4)
# ════════════════════════════════════════════════════════════════

with tab_parlay:
    st.markdown("### Smart Parlay Builder")
    st.caption(
        "Build a multi-leg parlay; the engine computes the joint "
        "probability empirically from games where ALL players played, "
        "compares to the independence assumption (book's default), "
        "and surfaces the EV gap at the book's quoted parlay odds."
    )

    @st.cache_data(show_spinner=False)
    def _parlay_player_options() -> pd.DataFrame:
        df = pd.read_parquet(
            Path(__file__).resolve().parent.parent
            / "data" / "nfl_player_stats_weekly.parquet"
        )
        recent = df[df["season"] >= 2022]
        opts = (recent.groupby(["player_id", "player_display_name",
                                 "position", "team"])
                .size().reset_index().rename(columns={0: "n_games"}))
        opts = opts[opts["n_games"] >= 8]
        opts["_label"] = (opts["player_display_name"]
                          + " · " + opts["team"]
                          + " · " + opts["position"]
                          + f" ({opts['n_games']}g)")
        return opts.sort_values("n_games", ascending=False).reset_index(drop=True)

    p_opts = _parlay_player_options()
    n_legs = st.number_input("Number of legs", 2, 5, 2, key="par_n")
    legs: list[Leg] = []
    cols = st.columns(int(n_legs))
    for i in range(int(n_legs)):
        with cols[i]:
            st.markdown(f"**Leg {i+1}**")
            player = st.selectbox(
                "Player", p_opts["_label"].tolist(),
                key=f"par_player_{i}", index=i if i < len(p_opts) else 0,
            )
            stat = st.selectbox(
                "Stat",
                ["passing_yards", "rushing_yards", "receiving_yards",
                 "receptions", "targets", "carries"],
                key=f"par_stat_{i}",
            )
            threshold = st.number_input(
                "Threshold", 0.0, 500.0, 75.0, step=0.5,
                key=f"par_thr_{i}",
            )
            side = st.selectbox(
                "Side", ["over", "under"], key=f"par_side_{i}",
            )
            row = p_opts[p_opts["_label"] == player].iloc[0]
            legs.append(Leg(
                player_id=row["player_id"],
                player_display_name=row["player_display_name"],
                stat=stat, threshold=float(threshold), side=side,
            ))

    book_odds = st.number_input(
        "Book parlay odds (American, e.g., +600)",
        -1000, 5000, 600, step=10, key="par_odds",
    )
    p_lookback = st.slider("Lookback (joint games)", 5, 50, 25,
                            key="par_lookback")
    if st.button("Score parlay", type="primary",
                  use_container_width=True, key="par_run"):
        try:
            r = score_parlay(legs, book_odds=int(book_odds),
                              lookback_games=p_lookback)
        except Exception as e:
            st.error(f"Pricing failed: {e}")
        else:
            st.markdown(f"#### {r.n_legs}-leg parlay")
            for label, p in r.leg_marginals:
                p_str = f"{p:.1%}" if (p == p) else "—"
                st.markdown(f"- **{label}** → P_model = {p_str}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P (independent)",
                      f"{r.p_independent:.1%}" if r.p_independent == r.p_independent else "—")
            m2.metric("P (correlated)",
                      f"{r.p_correlated:.1%}" if r.p_correlated == r.p_correlated else "—")
            m3.metric("Lift",
                      f"{r.correlation_lift:+.1%}" if r.correlation_lift == r.correlation_lift else "—")
            m4.metric("Joint games", f"{r.n_games_joint}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Fair odds (corr)",
                      f"{r.fair_american_correlated:+d}" if r.fair_american_correlated else "—")
            m2.metric("Book odds",
                      f"{r.book_american:+d}" if r.book_american else "—")
            if r.ev_vs_book is not None:
                m3.metric("EV vs book", f"{r.ev_vs_book:+.1%}")
            st.success(f"**Verdict:** {r.verdict}")

            anti = detect_anti_correlated(legs, lookback_games=p_lookback)
            if anti:
                st.warning(
                    f"⚠ {len(anti)} anti-correlated leg pair(s) detected: "
                    + ", ".join([f"legs {i+1}↔{j+1} (lift {l:+.0%})"
                                  for i, j, l in anti])
                )


# ════════════════════════════════════════════════════════════════
# Tab — Anytime / First TD (Feature 5.5)
# ════════════════════════════════════════════════════════════════

with tab_td:
    st.markdown("### Anytime / First TD Probability Vector")
    st.caption(
        "Per-player TD probability decomposed into rushing-only, "
        "receiving-only, and anytime. Pair with red-zone usage share "
        "to find players whose TD market is mispriced (most often, "
        "the rushing-TD-only line for a pass-catching back or the "
        "receiving-TD-only line for a goal-line back)."
    )

    @st.cache_data(show_spinner=False)
    def _td_player_options(position: str) -> pd.DataFrame:
        df = pd.read_parquet(
            Path(__file__).resolve().parent.parent
            / "data" / "nfl_player_stats_weekly.parquet"
        )
        recent = df[df["season"] >= 2023]
        if position:
            recent = recent[recent["position"] == position]
        return (recent.groupby(["player_id", "player_display_name",
                                 "position", "team"])
                .size().reset_index().rename(columns={0: "n_games"})
                .pipe(lambda d: d[d["n_games"] >= 6])
                .sort_values("n_games", ascending=False)
                .reset_index(drop=True))

    c1, c2 = st.columns([1, 2])
    with c1:
        td_pos = st.selectbox("Position", ["RB", "WR", "TE", "QB"],
                                key="td_pos")
        td_opts = _td_player_options(td_pos)
        td_opts["_label"] = (td_opts["player_display_name"]
                              + " · " + td_opts["team"]
                              + f" ({td_opts['n_games']}g)")
        td_player = st.selectbox("Player",
                                   td_opts["_label"].tolist(),
                                   key="td_player")
        td_lookback = st.slider("Lookback", 5, 40, 20, key="td_lookback")
        td_season = st.number_input("Season (for RZ usage)",
                                       2016, 2025, 2024,
                                       key="td_season")
        if st.button("Run", type="primary", use_container_width=True,
                       key="td_run"):
            row = td_opts[td_opts["_label"] == td_player].iloc[0]
            v = td_probability_vector(
                row["player_id"], season=int(td_season),
                lookback_games=td_lookback,
            )
            u = rz_usage_share(row["player_id"], int(td_season),
                                team=str(row["team"]))
            with c2:
                st.markdown(f"#### {row['player_display_name']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("P(rushing TD)",
                          f"{v.p_rush_td_baseline:.0%}")
                m2.metric("P(receiving TD)",
                          f"{v.p_rec_td_baseline:.0%}")
                m3.metric("P(anytime TD)",
                          f"{v.p_any_td_baseline:.0%}")
                st.markdown("**Per-game expected count:**")
                st.markdown(
                    f"- Rush TDs/g: {v.p_rush_td_baseline:.2f}  "
                    f"·  Rec TDs/g: {v.p_rec_td_baseline:.2f}  "
                    f"·  Any TDs/g: {v.p_any_td_baseline:.2f}"
                )
                st.markdown("**Red-zone usage share (this season):**")
                m1, m2, m3 = st.columns(3)
                m1.metric("RZ carries share", f"{u.rz_carries_share:.0%}")
                m2.metric("RZ targets share", f"{u.rz_targets_share:.0%}")
                m3.metric("Goal-line carries share",
                          f"{u.goal_line_carries_share:.0%}")
                st.caption(
                    f"Sample: {v.n_games_player} player games · "
                    f"team RZ plays this season: {u.n_team_rz_plays}"
                )


# ════════════════════════════════════════════════════════════════
# Tab — Trend Divergence (Feature 5.6)
# ════════════════════════════════════════════════════════════════

with tab_trend:
    st.markdown("### Snap-Share / Target-Share Trend Divergence")
    st.caption(
        "Flags players whose recent (last-3-week) usage has decoupled "
        "from their season baseline. Books often anchor to season "
        "averages and miss recent role expansions or contractions."
    )

    c1, c2, c3, c4 = st.columns(4)
    seasons_avail = list(range(2025, 2015, -1))
    season_pick = c1.selectbox("Season", seasons_avail, key="td_div_season")
    week_pick = c2.number_input("Through week", 4, 22, 18,
                                  key="td_div_week")
    pos_pick = c3.selectbox("Position", ["WR", "RB", "TE", "QB", "all"],
                              index=0, key="td_div_pos")
    min_z = c4.slider("Min |z|", 0.5, 3.0, 1.0, step=0.1,
                       key="td_div_minz")

    if st.button("Find divergences", type="primary",
                   use_container_width=True, key="td_div_run"):
        pos = None if pos_pick == "all" else pos_pick
        with st.spinner(f"Computing divergences for {pos_pick} W{week_pick} {season_pick}..."):
            df = league_divergence_today(
                int(season_pick), int(week_pick), position=pos,
                min_z=float(min_z),
            )
        if df.empty:
            st.info("No divergence flags at this filter.")
        else:
            df = df.sort_values("delta_z", ascending=False)
            view = df[["player_display_name", "team", "position",
                        "stat", "recent_avg", "season_avg",
                        "delta", "delta_z", "n_recent",
                        "n_season"]].copy()
            view["recent_avg"] = view["recent_avg"].round(2)
            view["season_avg"] = view["season_avg"].round(2)
            view["delta"] = view["delta"].apply(lambda x: f"{x:+.2f}")
            view["delta_z"] = view["delta_z"].apply(lambda x: f"{x:+.2f}")
            view.columns = ["Player", "Team", "Pos", "Stat",
                             "Recent avg", "Season avg", "Δ",
                             "z", "Recent n", "Season n"]
            st.dataframe(view, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# Tab — Longest-Play Edge (Feature 5.7)
# ════════════════════════════════════════════════════════════════

with tab_long:
    st.markdown("### Longest-Play Edge Finder")
    st.caption(
        "Empirical distribution of a player's per-game longest single "
        "play. Books model 'longest reception' / 'longest rush' on "
        "smooth distributions, but reality is bimodal — most plays are "
        "short, then a heavy tail. Players with elite explosive rates "
        "have structurally undervalued longest-play markets."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        kind = st.radio("Kind", ["reception", "rush"], horizontal=True,
                          key="lp_kind")
        opts = longest_player_options(kind=kind, min_games=20)
        opts["_label"] = (opts["player_display_name"]
                          + f" ({opts['n_games']}g)")
        player_label = st.selectbox("Player", opts["_label"].tolist(),
                                      key="lp_player")
        threshold = st.number_input("Target threshold (yards)",
                                       0.0, 200.0, 25.0, step=2.5,
                                       key="lp_thr")
        if st.button("Score", type="primary", use_container_width=True,
                       key="lp_run"):
            row = opts[opts["_label"] == player_label].iloc[0]
            r = p_longest_at_least(row["player_id"], threshold,
                                     kind=kind)
            with c2:
                st.markdown(
                    f"#### {row['player_display_name']} — "
                    f"longest {kind}"
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"P(≥ {threshold:.0f} yds)",
                           f"{r.p_at_least:.1%}")
                m2.metric("Median longest",
                           f"{r.median_longest:.0f}")
                m3.metric("P10 longest",
                           f"{r.p10_longest:.0f}")
                m4.metric("P90 longest",
                           f"{r.p90_longest:.0f}")
                st.caption(
                    f"Sample: {r.n_games} player games"
                )
                # Show distribution
                dist = longest_play_distribution(row["player_id"],
                                                   kind=kind)
                if not dist.empty:
                    import plotly.express as px
                    fig = px.histogram(
                        dist, x="longest_play",
                        nbins=20,
                        title=(f"Distribution of "
                                f"longest-{kind}-per-game"),
                    )
                    fig.add_vline(x=threshold, line_dash="dash",
                                    line_color="red")
                    fig.update_layout(height=320,
                                       margin=dict(l=10, r=10,
                                                   t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)

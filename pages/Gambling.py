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
    st.caption("Coming next. Will quantify how an OC/DC's scheme shifts "
               "league-average tendencies — pace, neutral pass rate, play-action "
               "rate, target distribution, run-game splits — vs. the prior year.")
    st.info("Not built yet. See roadmap in CLAUDE.md.")


# ════════════════════════════════════════════════════════════════
# Tab 3 — DvP (placeholder)
# ════════════════════════════════════════════════════════════════

with tab_dvp:
    st.markdown("### Defense vs. Position (DvP)")
    st.caption("Coming next. Per-defense allowance tables for fantasy "
               "points / yards / receptions / TDs by position group, "
               "with rolling-window views (last 4 / season / vs same-tier).")
    st.info("Not built yet. See roadmap in CLAUDE.md.")


# ════════════════════════════════════════════════════════════════
# Tab 4 — Coaching Tendencies (placeholder)
# ════════════════════════════════════════════════════════════════

with tab_coach:
    st.markdown("### Coaching / Play-Caller Tendencies")
    st.caption("Coming next. Per-OC tendencies (red zone pass rate, 4th-down "
               "aggression, neutral pace, RB committee splits) and per-HC "
               "game-script biases.")
    st.info("Not built yet. See roadmap in CLAUDE.md.")


# ════════════════════════════════════════════════════════════════
# Tab 5 — SGP Correlations (placeholder)
# ════════════════════════════════════════════════════════════════

with tab_sgp:
    st.markdown("### Same-Game-Parlay Correlations")
    st.caption("Coming next. Player-pair correlation matrices for SGP "
               "construction — e.g., when QB throws for 300+, what's the "
               "conditional probability his WR1 hits 75 yards?")
    st.info("Not built yet. See roadmap in CLAUDE.md.")

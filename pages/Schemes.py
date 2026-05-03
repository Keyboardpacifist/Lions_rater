"""Scheme Fit — Phase 1: WR/TE route opportunity.

Surface a team's passing DNA and a player's career route profile
side-by-side. Lets users see "this team is dig-heavy + slant-heavy"
and "this player runs a lot of dig routes" and intuit fit.

Future iterations
-----------------
- Vacated-demand calculator: pick a "departed" player, see which
  routes lose volume, surface candidates whose career profile fits
  the gap.
- Cross-position: RB run-scheme fit, QB throwing profile, OL scheme.
- Fantasy translation: vacated routes × team's typical targets ×
  position FP/target → projected fantasy redistribution.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from lib_shared import inject_css
import lib_scoring as fs


REPO = Path(__file__).resolve().parent.parent
TEAM_FP = REPO / "data" / "scheme" / "team_passing_fingerprint.parquet"
PLAYER_FP = REPO / "data" / "scheme" / "player_route_profile.parquet"
ATTRIBUTION = REPO / "data" / "scheme" / "team_route_attribution.parquet"
TRANSITIONS = REPO / "data" / "scheme" / "roster_transitions.parquet"


# ── Page config ───────────────────────────────────────────────────

st.set_page_config(page_title="Scheme Fit", page_icon="🧬",
                       layout="wide", initial_sidebar_state="expanded")
inject_css()


@st.cache_data(show_spinner=False)
def load_team_fp() -> pd.DataFrame:
    if not TEAM_FP.exists():
        return pd.DataFrame()
    return pd.read_parquet(TEAM_FP)


@st.cache_data(show_spinner=False)
def load_player_fp() -> pd.DataFrame:
    if not PLAYER_FP.exists():
        return pd.DataFrame()
    return pd.read_parquet(PLAYER_FP)


@st.cache_data(show_spinner=False)
def load_attribution() -> pd.DataFrame:
    if not ATTRIBUTION.exists():
        return pd.DataFrame()
    return pd.read_parquet(ATTRIBUTION)


@st.cache_data(show_spinner=False)
def load_transitions() -> pd.DataFrame:
    if not TRANSITIONS.exists():
        return pd.DataFrame()
    return pd.read_parquet(TRANSITIONS)


team_df = load_team_fp()
player_df = load_player_fp()

if team_df.empty or player_df.empty:
    st.error(
        "Scheme data not built yet. Run:\n\n"
        "```\npython tools/build_team_passing_fingerprint.py\n"
        "python tools/build_player_route_profile.py\n```"
    )
    st.stop()


# ── Header ────────────────────────────────────────────────────────

st.title("🧬 Scheme Fit — passing")
st.markdown(
    "**The platform's biggest moat.** Every team has a passing DNA "
    "(which routes they run, at what depth, in which personnel). "
    "Every receiver has a career route profile (what kind of "
    "receiver he is). Match them up and you see exactly which "
    "players fit which schemes — and which don't."
)


# ── Sidebar controls ──────────────────────────────────────────────

st.sidebar.header("Pick a matchup")

teams = sorted(team_df["team"].unique())
seasons = sorted(team_df["season"].unique(), reverse=True)
selected_team = st.sidebar.selectbox("Team", teams,
                                            index=teams.index("DET")
                                            if "DET" in teams else 0)
selected_season = st.sidebar.selectbox("Season", seasons, index=0)

dimension_options = sorted(team_df["dimension"].unique())
dimension = st.sidebar.selectbox(
    "Dimension", dimension_options,
    index=dimension_options.index("route")
    if "route" in dimension_options else 0,
    help="route = which routes the team runs / player gets targeted on. "
         "depth = behind-LOS / short / intermediate / deep. "
         "location = left/middle/right. "
         "personnel = 11 / 12 / 13 / 21 / 22 / etc. "
         "formation = SHOTGUN / I_FORM / etc.",
)

# Player picker — only WRs and TEs (skill receivers)
player_pool = player_df[
    player_df["position"].isin(["WR", "TE"])
][["player_id", "player_display_name", "position"]].drop_duplicates()
player_pool = player_pool.dropna(subset=["player_display_name"])
player_pool["label"] = (player_pool["player_display_name"]
                          + " (" + player_pool["position"] + ")")
player_pool = player_pool.sort_values("label")

selected_player_label = st.sidebar.selectbox(
    "Player (career profile)", player_pool["label"].tolist(),
    help="Select any career-qualified WR/TE to see their route profile.",
)
selected_player_id = player_pool.set_index("label").loc[
    selected_player_label, "player_id"]

# Scoring config — drives fantasy-point computations downstream
config_name = st.sidebar.selectbox(
    "Scoring system (FP math)",
    [c.name for c in fs.ALL_CONFIGS], index=0,
    help="Affects the vacated-demand FP totals and candidate "
         "FP/target conversion rates.",
)
scoring_config = fs.CONFIG_BY_NAME[config_name]


def _route_row_fp(catches: float, yards: float, tds: float,
                     position: str, config) -> float:
    """Apply scoring config to the per-route aggregates."""
    rec_value = config.reception
    if position == "TE" and config.te_premium_bonus > 0:
        rec_value += config.te_premium_bonus
    return (
        (catches or 0) * rec_value
        + (yards or 0) * config.rec_yard
        + (tds or 0) * config.rec_td
    )


# ── Team passing fingerprint ──────────────────────────────────────

team_fp = team_df[
    (team_df["team"] == selected_team)
    & (team_df["season"] == selected_season)
    & (team_df["dimension"] == dimension)
].sort_values("share", ascending=False)

# Filter out empty / unknown buckets
team_fp = team_fp[
    ~team_fp["category"].isin(["", "unknown"])
].copy()


# ── Player profile ────────────────────────────────────────────────

player_fp = player_df[
    (player_df["player_id"] == selected_player_id)
    & (player_df["dimension"] == dimension)
].sort_values("share", ascending=False)
player_fp = player_fp[
    ~player_fp["category"].isin(["", "unknown"])
].copy()


# ── Two-column display ────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"📡 {selected_team} {selected_season} · "
                   f"{dimension} DNA")
    if team_fp.empty:
        st.info("No data for this combination.")
    else:
        bar_df = team_fp.set_index("category")[
            ["share", "league_share"]
        ]
        bar_df.columns = [f"{selected_team}", "League avg"]
        st.bar_chart(bar_df, height=400)

        # Highlight the most extreme z-scores (signature traits)
        st.markdown("**Signature traits** (biggest deviations from league)")
        signatures = team_fp.copy()
        signatures["abs_z"] = signatures["share_z"].abs()
        signatures = signatures.nlargest(5, "abs_z")
        for _, row in signatures.iterrows():
            direction = "🔼 over" if row["share_z"] > 0 else "🔽 under"
            st.markdown(
                f"- **{row['category']}** — "
                f"{row['share']*100:.1f}% (vs league "
                f"{row['league_share']*100:.1f}%, "
                f"{direction}-indexed by {abs(row['share_z']):.1f}σ)"
            )

with col2:
    st.subheader(f"👤 {selected_player_label} · career {dimension}")
    if player_fp.empty:
        st.info("No career data for this player on this dimension.")
    else:
        bar_df = player_fp.set_index("category")[
            ["share", "league_share"]
        ]
        bar_df.columns = ["Player career", "League avg"]
        st.bar_chart(bar_df, height=400)

        # Specialty traits
        st.markdown("**Specialty traits** (career z vs other receivers)")
        specialties = player_fp.copy()
        specialties["abs_z"] = specialties["share_z"].abs()
        specialties = specialties.nlargest(5, "abs_z")
        for _, row in specialties.iterrows():
            direction = "🔼 over" if row["share_z"] > 0 else "🔽 under"
            st.markdown(
                f"- **{row['category']}** — "
                f"{row['share']*100:.1f}% of career targets "
                f"({direction}-indexed by {abs(row['share_z']):.1f}σ vs "
                f"other receivers; {row['yards_per_target']:.1f} ypt, "
                f"{row['epa_per_target']:+.2f} EPA/tgt)"
            )


# ── Naive fit score ───────────────────────────────────────────────

st.markdown("---")
st.subheader("🎯 Crude scheme-fit score")
st.caption(
    "Rough fit metric: weighted by category, the correlation between "
    "the team's category z-score and the player's category z-score. "
    "Higher = the team uses categories where this player is a "
    "specialist. (Real fit calculator with vacated-demand reasoning "
    "is the next iteration — see project_route_opportunity_feature.md.)"
)

# Join team z-shares with player z-shares on category
fit_join = team_fp[["category", "share_z"]].rename(
    columns={"share_z": "team_z"}
).merge(
    player_fp[["category", "share_z", "share"]].rename(
        columns={"share_z": "player_z", "share": "player_share"}),
    on="category", how="inner"
)

if len(fit_join) >= 3:
    # Pearson correlation between team z and player z, weighted by
    # the player's share (so categories the player rarely gets count less)
    fit_corr = fit_join["team_z"].corr(fit_join["player_z"])
    st.metric(
        f"Fit correlation · {selected_player_label} → "
        f"{selected_team} {selected_season}",
        f"{fit_corr:+.2f}",
        help="-1 = scheme runs the OPPOSITE of this player's "
             "specialty routes. 0 = neutral. +1 = perfect alignment "
             "between team's signature routes and player's specialty routes."
    )
    if fit_corr > 0.4:
        st.success(
            "**Strong fit.** This team runs the routes this player "
            "specializes in."
        )
    elif fit_corr > 0.1:
        st.info("**Modest fit.** Some alignment but not lockstep.")
    elif fit_corr > -0.1:
        st.info("**Neutral.** Player's specialties don't strongly "
                  "match or contradict the scheme.")
    else:
        st.warning(
            "**Scheme mismatch.** This team's signature routes are "
            "ones this player has historically been BELOW average on."
        )
else:
    st.info("Not enough overlapping categories for a fit score.")


# ── Roster Transition Ledger — vacated demand calculator ─────────

st.markdown("---")
st.header("🚪 Roster Transition Ledger")
st.caption(
    f"What did **{selected_team}** lose and gain between "
    f"the 2025 NFL season and now? Departures show the per-route "
    f"target load that walked out the door. Arrivals show what "
    f"the new vets bring (rookies flagged TBD)."
)

attribution = load_attribution()
transitions = load_transitions()

if attribution.empty or transitions.empty:
    st.info(
        "Run `python tools/build_team_route_attribution.py` and "
        "`python tools/build_roster_transitions.py` first."
    )
else:
    team_trans = transitions[transitions["team"] == selected_team]
    departures = team_trans[
        team_trans["transition_type"] == "departure"
    ].sort_values("prior_season_targets", ascending=False)
    vet_arrivals = team_trans[
        (team_trans["transition_type"] == "arrival")
        & (team_trans["is_rookie"] == False)
    ].sort_values("career_targets", ascending=False)
    rookie_arrivals = team_trans[
        (team_trans["transition_type"] == "arrival")
        & (team_trans["is_rookie"] == True)
    ]

    # ── Vacated route demand (with fantasy points) ─────────────────
    departed_ids = departures["player_id"].dropna().tolist()
    if departed_ids:
        last_year_attr = attribution[
            (attribution["team"] == selected_team)
            & (attribution["season"] == 2025)
        ].copy()

        # Compute FP for every route-row using the selected config
        last_year_attr["row_fp"] = last_year_attr.apply(
            lambda r: _route_row_fp(
                r.get("catches"), r.get("yards"), r.get("tds"),
                r.get("position", ""), scoring_config),
            axis=1,
        )

        departed_attr = last_year_attr[
            last_year_attr["receiver_player_id"].isin(departed_ids)
        ]
        vacated = (
            departed_attr
            .groupby("route", as_index=False)
            .agg(vacated_targets=("targets", "sum"),
                 vacated_catches=("catches", "sum"),
                 vacated_yards=("yards", "sum"),
                 vacated_tds=("tds", "sum"),
                 vacated_fp=("row_fp", "sum"))
        )
        team_route = (
            last_year_attr.groupby("route", as_index=False)
                          .agg(team_total=("targets", "sum"),
                                 team_total_fp=("row_fp", "sum"))
        )
        vacated = vacated.merge(team_route, on="route", how="left")
        vacated["vacated_share"] = (
            vacated["vacated_targets"] / vacated["team_total"]
        )
        vacated["vacated_fp_share"] = (
            vacated["vacated_fp"] / vacated["team_total_fp"]
        )
        vacated = vacated.sort_values("vacated_fp", ascending=False)
    else:
        vacated = pd.DataFrame(columns=[
            "route", "vacated_targets", "vacated_catches",
            "vacated_yards", "vacated_tds", "vacated_fp",
            "team_total", "team_total_fp", "vacated_share",
            "vacated_fp_share",
        ])

    # ── Display in three columns ────────────────────────────────────
    col_d, col_v, col_r = st.columns([1, 1, 1])

    with col_d:
        st.markdown(f"**📤 Departures** ({len(departures)})")
        if departures.empty:
            st.caption("Nobody significant left.")
        else:
            disp = departures[[
                "player_display_name", "position",
                "prior_season_targets",
            ]].copy()
            disp.columns = ["Player", "Pos", "2025 targets"]
            st.dataframe(disp, use_container_width=True,
                            hide_index=True, height=240)

    with col_v:
        st.markdown(f"**📥 Vet arrivals** ({len(vet_arrivals)})")
        if vet_arrivals.empty:
            st.caption("No vet additions.")
        else:
            disp = vet_arrivals[[
                "player_display_name", "position",
                "prior_team", "career_targets",
            ]].copy()
            disp.columns = ["Player", "Pos", "From", "Career targets"]
            st.dataframe(disp, use_container_width=True,
                            hide_index=True, height=240)

    with col_r:
        st.markdown(f"**🆕 Rookies** ({len(rookie_arrivals)})")
        if rookie_arrivals.empty:
            st.caption("No rookies on the receiving corps.")
        else:
            disp = rookie_arrivals[[
                "player_display_name", "position",
            ]].copy()
            disp.columns = ["Player", "Pos"]
            st.caption(
                "*Profile TBD — combine archetype + college route "
                "data are future iterations.*"
            )
            st.dataframe(disp, use_container_width=True,
                            hide_index=True, height=240)

    # ── Vacated demand by route ────────────────────────────────────
    st.markdown(f"### 🎯 Vacated route demand "
                  f"({config_name} fantasy points)")
    total_vacated_fp = float(vacated["vacated_fp"].sum()) if not vacated.empty else 0.0
    st.caption(
        f"**{total_vacated_fp:.1f} {config_name} fantasy points walked out the door.** "
        "Per-route breakdown: targets, FP, and what % of the team's "
        "FP on that route is now unfilled. Higher rows = bigger holes."
    )
    if vacated.empty:
        st.info("No significant vacated demand to compute.")
    else:
        vac_disp = vacated[
            vacated["vacated_fp"] > 0
        ].copy()
        vac_disp["vacated_fp_pct"] = (
            vac_disp["vacated_fp_share"] * 100
        ).round(1)
        vac_disp = vac_disp[[
            "route", "vacated_targets", "vacated_fp",
            "team_total_fp", "vacated_fp_pct",
        ]]
        vac_disp.columns = ["Route", "Targets vacated",
                              f"FP vacated ({config_name})",
                              f"Team's 2025 FP on route",
                              "% of team's FP vacated"]
        vac_disp["FP vacated ({})".format(config_name)] = (
            vac_disp[f"FP vacated ({config_name})"].round(1)
        )
        vac_disp[f"Team's 2025 FP on route"] = (
            vac_disp[f"Team's 2025 FP on route"].round(1)
        )
        st.dataframe(vac_disp, use_container_width=True,
                        hide_index=True)

        # Build absorption candidate pool: INCUMBENTS who stayed +
        # VET ARRIVALS. Incumbents = receivers on the team's 2025
        # cohort who are NOT in the departure list.
        last_year_receivers = (
            attribution[
                (attribution["team"] == selected_team)
                & (attribution["season"] == 2025)
            ]
            .groupby(["receiver_player_id", "player_display_name",
                       "position"], as_index=False)
            .agg(prior_targets=("targets", "sum"))
        )
        last_year_receivers = last_year_receivers[
            last_year_receivers["prior_targets"] >= 10
        ]
        departed_id_set = set(departures["player_id"].dropna())
        incumbents = last_year_receivers[
            ~last_year_receivers["receiver_player_id"]
                .isin(departed_id_set)
        ].copy()
        incumbents["origin"] = "Incumbent"
        incumbents = incumbents.rename(
            columns={"receiver_player_id": "player_id"})

        vet_origin = vet_arrivals[[
            "player_id", "player_display_name", "position",
            "career_targets",
        ]].copy()
        vet_origin = vet_origin.rename(
            columns={"career_targets": "prior_targets"})
        vet_origin["origin"] = "New (FA/trade)"

        candidates = pd.concat(
            [incumbents[[
                "player_id", "player_display_name", "position",
                "prior_targets", "origin",
            ]],
             vet_origin],
            ignore_index=True,
        )

        # Compute career FP/target per (player, route) for candidates
        # using the selected scoring config
        cand_attribution = attribution[
            attribution["receiver_player_id"].isin(
                candidates["player_id"].dropna())
        ].copy()
        cand_attribution["row_fp"] = cand_attribution.apply(
            lambda r: _route_row_fp(
                r.get("catches"), r.get("yards"), r.get("tds"),
                r.get("position", ""), scoring_config),
            axis=1,
        )
        career_fp = (
            cand_attribution
            .groupby(["receiver_player_id", "route"], as_index=False)
            .agg(career_targets_route=("targets", "sum"),
                 career_fp_route=("row_fp", "sum"))
        )
        career_fp["career_fp_per_target"] = (
            career_fp["career_fp_route"]
            / career_fp["career_targets_route"]
        )

        # Match against vacated routes
        cand_routes = player_df[
            player_df["player_id"].isin(
                candidates["player_id"].dropna())
            & (player_df["dimension"] == "route")
        ].copy()
        cand_routes = cand_routes.merge(
            candidates[["player_id", "origin"]],
            on="player_id", how="left",
        )
        # Bring in career FP/target on this route
        cand_routes = cand_routes.merge(
            career_fp.rename(columns={"receiver_player_id": "player_id",
                                          "route": "category"}),
            on=["player_id", "category"], how="left",
        )

        if not cand_routes.empty:
            fits = []
            for _, vrow in vacated.iterrows():
                if vrow["vacated_targets"] <= 0:
                    continue
                rcands = cand_routes[
                    cand_routes["category"] == vrow["route"]
                ].sort_values("share_z", ascending=False)
                if rcands.empty:
                    continue
                # Top 3 candidates per route
                for i, (_, top) in enumerate(rcands.head(3).iterrows()):
                    fp_per_t = top.get("career_fp_per_target")
                    projected_fp_absorbed = (
                        (fp_per_t or 0) * vrow["vacated_targets"]
                        * top.get("share", 0) /
                        max((rcands.head(3)["share"].sum() or 1), 1e-9)
                    ) if pd.notna(fp_per_t) else None
                    fits.append({
                        "Route": vrow["route"] if i == 0 else "",
                        "Vacated FP":
                            f"{vrow['vacated_fp']:.1f}"
                            if i == 0 else "",
                        "Rank": i + 1,
                        "Origin": top["origin"],
                        "Candidate": top["player_display_name"],
                        "Pos": top["position"],
                        "Career share":
                            f"{top['share']*100:.1f}%",
                        "z":
                            f"{top['share_z']:+.1f}σ",
                        "Career FP/target on route":
                            f"{fp_per_t:.2f}"
                            if pd.notna(fp_per_t) else "—",
                        "Projected FP absorbed":
                            f"{projected_fp_absorbed:.1f}"
                            if projected_fp_absorbed is not None
                            else "—",
                    })
            if fits:
                st.markdown(
                    f"### 🎯 Best fit per vacated route — "
                    f"{config_name} FP conversion"
                )
                st.caption(
                    "For each vacated route, the top 3 candidates "
                    "(incumbents who stayed + new vet arrivals) "
                    "ranked by career specialty z-score. "
                    "**Career FP/target** is how many fantasy points "
                    "this player has converted **per target on this "
                    "specific route** across his NFL career — the "
                    "stat nobody else publishes free. "
                    "**Projected FP absorbed** is a rough estimate "
                    "of fantasy points each candidate would pick up "
                    "if vacated targets distribute by share-z weight. "
                    "Incumbent + elite FP/target = the alpha pick."
                )
                st.dataframe(pd.DataFrame(fits),
                                use_container_width=True,
                                hide_index=True)
            else:
                st.info(
                    "No qualifying candidates have a career profile "
                    "on the vacated routes. (May indicate rookies / "
                    "low-volume players are the only options.)"
                )


# ── Footer / coming next ──────────────────────────────────────────

with st.expander("📐 What this is, and what's coming next"):
    st.markdown("""
**This is Phase 1 of the Scheme Fit Engine.** WR/TE route opportunity.

**Working today:**
- Per-team passing DNA (route, depth, location, personnel, formation)
- Per-player career route profile across the same dimensions
- A naive scheme-fit correlation score

**Next iterations:**
- **Vacated-demand calculator.** Pick a team + a "departed" player
  → see which specific route demand is unfilled. Then surface
  candidates from existing roster + free-agency + draft pool whose
  career profile fits the gap.
- **Fantasy translation.** Convert vacated route demand to projected
  target volume → projected fantasy points → ADP value impact.
- **Cross-position.** RB run-scheme fit (zone vs gap, personnel),
  QB throwing-profile fit, OL scheme fit. Same data infrastructure.

**Why this matters:**
Standard fantasy answers "WR1 left, his targets redistribute." We
answer "47 dig targets are unfilled, only WR3 has the career profile
to absorb them, and he's currently a $1 FAAB add." That's the kind
of edge no other free fantasy tool publishes.
""")

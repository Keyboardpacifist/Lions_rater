"""Fantasy — ADP triangulation.

Shows three signals side-by-side for every fantasy-relevant player:
  - **ADP rank** (where the market is drafting him — Sleeper search_rank,
    re-ranked within position)
  - **Prior-year FP rank** (what he scored last year, in the selected
    scoring config, ranked within position)
  - **GAS rank** (his skill grade from prior year, ranked within position)

When the three agree → fairly priced. When they disagree → the
disagreement *itself* tells you the type of mispricing:
  - Sleeper: GAS + FP both better than ADP → market hasn't caught up
  - Sell-high (context-fueled): FP > GAS → production beating skill;
    risk of regression
  - Buy the dip (context victim): GAS > FP → skill is real but last
    year had bad context (QB/OL/health)
  - Hype risk: ADP high but no NFL track record (rookies / unproven)
  - Overvalued: ADP high but both FP and GAS are worse — fade

This replaces the simple "GAS-vs-FP gap" approach. Triangulation is
more informative because the three-way pattern surfaces the *type*
of mispricing, not just the magnitude.
"""
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

import lib_scoring as fs
from lib_gas_panels import load_gas_data
from lib_shared import inject_css


REPO = Path(__file__).resolve().parent.parent
WEEKLY_PATH = REPO / "data" / "nfl_player_stats_weekly.parquet"
ADP_PATH = REPO / "data" / "fantasy" / "sleeper_adp.parquet"
ATTRIBUTION_PATH = REPO / "data" / "scheme" / "team_route_attribution.parquet"
TRANSITIONS_PATH = REPO / "data" / "scheme" / "roster_transitions.parquet"
PLAYER_ROUTE_PATH = REPO / "data" / "scheme" / "player_route_profile.parquet"

POSITION_TO_GAS = {"QB": "qb", "RB": "rb", "WR": "wr", "TE": "te"}


# ── Page config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Fantasy", page_icon="🏆",
    layout="wide", initial_sidebar_state="expanded",
)
inject_css()


# ── Data loaders ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_weekly() -> pd.DataFrame:
    return pd.read_parquet(WEEKLY_PATH)


@st.cache_data(show_spinner=False)
def load_adp() -> pd.DataFrame:
    if not ADP_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(ADP_PATH)


@st.cache_data(show_spinner=False)
def load_attribution() -> pd.DataFrame:
    if not ATTRIBUTION_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(ATTRIBUTION_PATH)


@st.cache_data(show_spinner=False)
def load_transitions() -> pd.DataFrame:
    if not TRANSITIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(TRANSITIONS_PATH)


@st.cache_data(show_spinner=False)
def load_player_route_profile() -> pd.DataFrame:
    if not PLAYER_ROUTE_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(PLAYER_ROUTE_PATH)


@st.cache_data(show_spinner=False)
def load_all_gas() -> pd.DataFrame:
    rows = []
    for pos_short, pos_long in POSITION_TO_GAS.items():
        df = load_gas_data(pos_long)
        if df is None:
            continue
        df = df[["player_id", "season_year", "gas_score",
                 "gas_label"]].copy()
        df["position"] = pos_short
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_season_fp(season_df: pd.DataFrame) -> pd.DataFrame:
    sum_cols = [
        "completions", "attempts", "passing_yards", "passing_tds",
        "passing_interceptions", "carries", "rushing_yards",
        "rushing_tds", "targets", "receptions", "receiving_yards",
        "receiving_tds", "fantasy_points", "fantasy_points_ppr",
    ]
    avail = [c for c in sum_cols if c in season_df.columns]
    grp = season_df.groupby(
        ["player_id", "player_display_name", "position",
         "team", "season"], as_index=False)
    totals = grp[avail].sum()
    totals["games"] = grp.size()["size"].values

    totals["fp_standard"] = totals["fantasy_points"]
    totals["fp_ppr"] = totals["fantasy_points_ppr"]
    totals["fp_half"] = (totals["fantasy_points"]
                          + totals["fantasy_points_ppr"]) / 2
    is_te = (totals["position"] == "TE").astype(float)
    totals["fp_te_premium"] = (totals["fantasy_points_ppr"]
                                 + 0.5 * totals["receptions"] * is_te)
    return totals


# ── Triangulation logic ──────────────────────────────────────────

def categorize(adp_rank: float, fp_rank: float, gas_rank: float,
                 threshold: int = 8) -> str:
    """Classify the triangulation pattern in 6 buckets.

    Lower rank = better. `threshold` is how many position-rank slots
    of disagreement we tolerate before calling it a real misalignment.
    """
    if pd.isna(adp_rank):
        return "Not drafted"

    no_fp = pd.isna(fp_rank)
    no_gas = pd.isna(gas_rank)

    if no_fp and no_gas:
        return "Hype risk (rookie / unproven)"
    if no_fp:
        # Has GAS but no FP: rookie last year? injury? Trust GAS as the signal.
        if gas_rank < adp_rank - threshold:
            return "Sleeper (skill underrated)"
        return "Hype risk (no FP)"
    if no_gas:
        return "Limited GAS data"

    # All three present — compute the spread
    fp_below_adp = fp_rank > adp_rank + threshold
    gas_below_adp = gas_rank > adp_rank + threshold
    fp_above_adp = fp_rank < adp_rank - threshold
    gas_above_adp = gas_rank < adp_rank - threshold
    fp_above_gas = fp_rank < gas_rank - threshold
    gas_above_fp = gas_rank < fp_rank - threshold

    # 1. All within threshold — aligned
    if not (fp_below_adp or gas_below_adp or fp_above_adp
            or gas_above_adp):
        return "Aligned"

    # 2. Sleeper: skill AND production both better than ADP suggests
    if gas_above_adp and fp_above_adp:
        return "Sleeper"

    # 3. Sell-high: production way better than skill (TD luck / context)
    if fp_above_gas and not gas_above_adp:
        return "Sell-high (context-fueled)"

    # 4. Buy the dip: skill solid, last year's FP lagged
    if gas_above_fp and not fp_above_adp:
        return "Buy the dip (context victim)"

    # 5. Skill alone says undervalued (FP roughly matches ADP)
    if gas_above_adp and not fp_above_adp:
        return "Skill underrated"

    # 6. Both signals worse than ADP — market is too high
    if fp_below_adp and gas_below_adp:
        return "Overvalued"

    return "Mixed signals"


CATEGORY_COLOR = {
    "Sleeper":                       "#16a34a",
    "Buy the dip (context victim)":  "#22c55e",
    "Skill underrated":              "#84cc16",
    "Aligned":                       "#9ca3af",
    "Mixed signals":                 "#a78bfa",
    "Limited GAS data":              "#a78bfa",
    "Hype risk (rookie / unproven)": "#f59e0b",
    "Hype risk (no FP)":             "#f59e0b",
    "Sell-high (context-fueled)":    "#ef4444",
    "Overvalued":                    "#dc2626",
    "Not drafted":                   "#9ca3af",
}


# ── Header ────────────────────────────────────────────────────────

st.title("🏆 Fantasy — ADP triangulation")
st.markdown(
    "Three numbers, side by side: **where the market drafts him** "
    "(ADP), **what he scored last year** (FP), **how he grades as a "
    "football player** (GAS). When all three agree → fairly priced. "
    "When they disagree → the *type* of disagreement tells you "
    "whether to buy, sell, or fade."
)


# ── Sidebar controls ──────────────────────────────────────────────

st.sidebar.header("Settings")

config_name = st.sidebar.selectbox(
    "Scoring system",
    options=[c.name for c in fs.ALL_CONFIGS], index=0,
    help="Affects last-year FP rank. Doesn't change ADP or GAS.",
)
config_to_col = {
    "Standard":   "fp_standard",
    "PPR":        "fp_ppr",
    "Half-PPR":   "fp_half",
    "TE Premium": "fp_te_premium",
}
fp_col = config_to_col[config_name]

weekly = load_weekly()
season_options = sorted(
    [int(s) for s in weekly["season"].dropna().unique()
     if int(s) >= 2020], reverse=True,
)
prior_season = st.sidebar.selectbox(
    "Prior season (for FP + GAS)",
    season_options, index=0,
    help="ADP is current/live. FP and GAS are from this prior season.",
)

position = st.sidebar.selectbox(
    "Position", ["QB", "RB", "WR", "TE", "FLEX (RB/WR/TE)"],
    index=2,
)

threshold = st.sidebar.slider(
    "Mispricing threshold (position-rank slots)",
    3, 20, 8,
    help="How many position-rank slots of disagreement before "
         "calling it a real mispricing.",
)


# ── Build the triangulation table ─────────────────────────────────

adp = load_adp()
if adp.empty:
    st.error(
        "ADP data not loaded. Run `python tools/pull_sleeper_adp.py` "
        "first."
    )
    st.stop()

# 1. Prior-season FP, ranked within position
season_df = weekly[weekly["season"] == prior_season]
if position == "FLEX (RB/WR/TE)":
    pos_filter = ["RB", "WR", "TE"]
else:
    pos_filter = [position]
season_df = season_df[season_df["position"].isin(pos_filter)]

if len(season_df) == 0:
    st.warning(f"No data for {position} in {prior_season}.")
    st.stop()

fp_totals = compute_season_fp(season_df)
fp_totals["fp_rank"] = (
    fp_totals.groupby("position")[fp_col]
             .rank(ascending=False, method="min")
)

# 2. GAS for prior season, ranked within position
gas_all = load_all_gas()
gas_for_season = gas_all[gas_all["season_year"] == prior_season].copy()
gas_for_season["gas_rank"] = (
    gas_for_season.groupby("position")["gas_score"]
                  .rank(ascending=False, method="min")
)

# 3. ADP — already has pos_rank, filter to our positions
adp_filt = adp[adp["position"].isin(pos_filter)].copy()
adp_filt = adp_filt.rename(columns={"pos_rank": "adp_rank"})

# 4. Combine: outer join because we want hype-risk rookies too
combined = adp_filt.merge(
    fp_totals[["player_id", "position", "fp_rank", fp_col, "games"]],
    on=["player_id", "position"], how="outer",
)
combined = combined.merge(
    gas_for_season[["player_id", "position", "gas_rank",
                     "gas_score", "gas_label"]],
    on=["player_id", "position"], how="left",
)

# Player-name fallback (ADP source has full_name; FP source has display_name)
if "player_display_name" in combined.columns:
    combined["display_name"] = combined["player_display_name"].fillna(
        combined["full_name"])
else:
    combined["display_name"] = combined["full_name"]
if "team_x" in combined.columns:
    combined["display_team"] = combined["team_x"].fillna(
        combined.get("team_y", ""))
elif "team" in combined.columns:
    combined["display_team"] = combined["team"]

# Drop rows where we have no display info at all
combined = combined.dropna(subset=["display_name"])

# Categorize
combined["category"] = combined.apply(
    lambda r: categorize(r.get("adp_rank"), r.get("fp_rank"),
                            r.get("gas_rank"), threshold=threshold),
    axis=1,
)


# ── Main triangulation table ──────────────────────────────────────

st.markdown(f"### 📋 Triangulation · {position} · "
              f"{prior_season} → 2025 ADP · {config_name}")

# Filter out junk rows: only keep players with at least 2 of 3 signals
display_df = combined[
    combined[["adp_rank", "fp_rank", "gas_rank"]].notna().sum(axis=1) >= 2
].copy()

# Sort by ADP rank by default
display_df = display_df.sort_values("adp_rank", na_position="last")

table = display_df[[
    "display_name", "display_team", "position",
    "adp_rank", "fp_rank", "gas_rank", "category", fp_col, "gas_score",
]].copy()
table.columns = ["Player", "Team", "Pos", "ADP rank", "FP rank",
                  "GAS rank", "Pattern", "FP", "GAS"]
for col in ("ADP rank", "FP rank", "GAS rank"):
    table[col] = table[col].astype("Float64").round(0)
table["FP"] = table["FP"].astype("Float64").round(1)
table["GAS"] = table["GAS"].astype("Float64").round(1)

st.dataframe(table.head(80), use_container_width=True,
                hide_index=True, height=560)


# ── Category breakdowns ───────────────────────────────────────────

st.markdown("---")
st.markdown("### 🎯 By pattern — what to do with each")

# Filter to qualified players (have FP or GAS — drop pure no-data)
qualified = display_df[
    (display_df["adp_rank"].notna())
    & (display_df["fp_rank"].notna() | display_df["gas_rank"].notna())
].copy()

cat_views = [
    ("🟢 Sleepers", "Sleeper",
     "GAS and FP both rank better than ADP. Market hasn't caught up."),
    ("🟢 Buy the dip (context victim)", "Buy the dip (context victim)",
     "Skill grade is real, but last year's production lagged "
     "(weak QB/OL/health/role). Bet on regression to skill."),
    ("🟢 Skill underrated", "Skill underrated",
     "GAS says better than ADP. FP didn't fully show it. Sneaky bet."),
    ("🔴 Sell-high (context-fueled)", "Sell-high (context-fueled)",
     "Last year's production was above the skill grade. TD luck, "
     "context, or schedule. Trade them at peak."),
    ("🔴 Overvalued", "Overvalued",
     "Both FP and GAS rank worse than ADP. Market is paying for "
     "name brand. Fade."),
    ("🟡 Hype risk", None,  # special — combine both hype categories
     "ADP is high but the player has no FP track record. Pure "
     "projection bet — could pay big or bust."),
]

for label, cat_value, blurb in cat_views:
    if cat_value is None:
        sub = qualified[qualified["category"].str.contains(
            "Hype risk", na=False)]
    else:
        sub = qualified[qualified["category"] == cat_value]
    if sub.empty:
        continue
    sub = sub.sort_values("adp_rank").head(15)

    with st.expander(f"**{label}** ({len(sub)} candidates)",
                       expanded=(cat_value == "Sleeper"
                                  or cat_value == "Buy the dip "
                                                    "(context victim)")):
        st.caption(blurb)
        out = sub[["display_name", "display_team", "position",
                    "adp_rank", "fp_rank", "gas_rank"]].copy()
        out.columns = ["Player", "Team", "Pos", "ADP", "FP", "GAS"]
        for c in ("ADP", "FP", "GAS"):
            out[c] = out[c].astype("Float64").round(0)
        st.dataframe(out, use_container_width=True, hide_index=True)


# ── Methodology ──────────────────────────────────────────────────

with st.expander("📐 How is this computed?", expanded=False):
    st.markdown(f"""
**ADP source:** Sleeper public API — `search_rank` field, re-ranked
within position. Free, refreshed any time `tools/pull_sleeper_adp.py`
is run. Reflects consensus draft position across Sleeper's user base.

**FP rank:** {prior_season} actual fantasy points in **{config_name}**
scoring, ranked within position.

**GAS rank:** {prior_season} GAS Score (our proprietary skill grade),
ranked within position.

**Pattern threshold:** {threshold} position-rank slots. Adjust in the
sidebar to surface bigger or smaller mispricings.

**Caveats:**
- Sleeper `search_rank` is a single overall rank — not split by
  scoring format. We re-rank within position for each comparison.
- ADP shifts daily during draft season; rerun the pull script.
- This is a triangulation against *prior-year* FP/GAS. Forward
  projections (Phase 3) will replace prior-year FP with projected
  next-year FP for an even cleaner comparison.
- Rookies and players returning from injury show up as "Hype risk"
  because they have no NFL FP yet; college GAS / projected GAS
  features will fill that gap.
""")


# ══════════════════════════════════════════════════════════════════
#  Section 2: 🚪 Vacated Demand & Scheme Fit (per-team narratives)
# ══════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("🚪 Vacated demand & scheme fit")
st.markdown(
    "When teams lose receivers, **specific routes lose specific "
    "volume.** This section quantifies the dig-merchant problem: "
    "*\"Team X lost their dig merchant — 47 IN/DIG targets unfilled, "
    "and only Player Y on the existing roster has the career profile "
    "to absorb them.\"* Pick a team and see the story."
)


def _route_row_fp(catches, yards, tds, position, config):
    rec_value = config.reception
    if position == "TE" and config.te_premium_bonus > 0:
        rec_value += config.te_premium_bonus
    return ((catches or 0) * rec_value
            + (yards or 0) * config.rec_yard
            + (tds or 0) * config.rec_td)


attribution = load_attribution()
transitions = load_transitions()
player_route_df = load_player_route_profile()

if attribution.empty or transitions.empty:
    st.info(
        "Scheme data not built yet. Run the scheme builders:\n"
        "```\n"
        "python tools/build_team_passing_fingerprint.py\n"
        "python tools/build_player_route_profile.py\n"
        "python tools/build_team_route_attribution.py\n"
        "python tools/build_roster_transitions.py\n"
        "```"
    )
else:
    # Pre-compute total vacated FP per team to sort the picker by
    # "biggest story this offseason"
    config = fs.CONFIG_BY_NAME[config_name]
    last_year_attr = attribution[attribution["season"] == 2025].copy()
    last_year_attr["row_fp"] = last_year_attr.apply(
        lambda r: _route_row_fp(r.get("catches"), r.get("yards"),
                                  r.get("tds"), r.get("position", ""),
                                  config),
        axis=1,
    )

    departure_ids_per_team = (
        transitions[transitions["transition_type"] == "departure"]
        .dropna(subset=["player_id"])
        .groupby("team")["player_id"].apply(list).to_dict()
    )

    team_vacated_totals = []
    for team, dep_ids in departure_ids_per_team.items():
        team_attr = last_year_attr[last_year_attr["team"] == team]
        vacated_fp = team_attr[
            team_attr["receiver_player_id"].isin(dep_ids)
        ]["row_fp"].sum()
        team_vacated_totals.append({
            "team": team,
            "vacated_fp": float(vacated_fp),
            "n_departures": len(dep_ids),
        })
    team_vacated = pd.DataFrame(team_vacated_totals).sort_values(
        "vacated_fp", ascending=False)

    # Team picker (default: biggest story)
    team_options = team_vacated["team"].tolist()
    if not team_options:
        st.info("No team transition data available.")
    else:
        # Show top 5 biggest stories at top
        st.markdown(f"**Biggest offseason transitions ({config_name}):**")
        big_stories = team_vacated.head(5).copy()
        big_stories["vacated_fp"] = big_stories["vacated_fp"].round(1)
        big_stories.columns = [
            "Team", f"Vacated {config_name} FP", "Departures",
        ]
        st.dataframe(big_stories, use_container_width=True,
                        hide_index=True, height=215)

        selected_scheme_team = st.selectbox(
            "Pick a team to drill into",
            team_options,
            index=0,
            help="Defaults to the biggest vacated-FP event of the offseason.",
        )

        team_trans = transitions[
            transitions["team"] == selected_scheme_team
        ]
        team_deps = team_trans[
            team_trans["transition_type"] == "departure"
        ].dropna(subset=["player_id"])
        dep_ids = team_deps["player_id"].tolist()

        team_attr = last_year_attr[
            last_year_attr["team"] == selected_scheme_team
        ]

        # Departure summary
        dep_summary = (
            team_attr[team_attr["receiver_player_id"].isin(dep_ids)]
            .groupby(["receiver_player_id", "player_display_name",
                       "position"], as_index=False)
            .agg(targets=("targets", "sum"),
                 fp=("row_fp", "sum"))
            .sort_values("fp", ascending=False)
        )
        dep_summary["fp"] = dep_summary["fp"].round(1)

        # Vacated by route (with FP)
        vacated = (
            team_attr[team_attr["receiver_player_id"].isin(dep_ids)]
            .groupby("route", as_index=False)
            .agg(vacated_targets=("targets", "sum"),
                 vacated_fp=("row_fp", "sum"))
        )
        team_route_total = (
            team_attr.groupby("route", as_index=False)
                     .agg(team_total_fp=("row_fp", "sum"))
        )
        vacated = vacated.merge(team_route_total, on="route", how="left")
        vacated["vacated_fp_pct"] = (
            (vacated["vacated_fp"] / vacated["team_total_fp"]) * 100
        )
        vacated = vacated.sort_values("vacated_fp", ascending=False)

        # Internal absorbers — last year's roster minus departures
        last_year_recv = (
            team_attr.groupby(
                ["receiver_player_id", "player_display_name",
                 "position"], as_index=False)
            .agg(prior_targets=("targets", "sum"))
        )
        last_year_recv = last_year_recv[
            last_year_recv["prior_targets"] >= 10
        ]
        incumbent_ids = [
            pid for pid in last_year_recv["receiver_player_id"]
            if pid not in dep_ids
        ]
        # Vet arrivals
        vet_arr_ids = team_trans[
            (team_trans["transition_type"] == "arrival")
            & (team_trans["is_rookie"] == False)
        ]["player_id"].dropna().tolist()
        candidate_ids = incumbent_ids + vet_arr_ids

        # Career FP/target on each route for candidates
        cand_attr = attribution[
            attribution["receiver_player_id"].isin(candidate_ids)
        ].copy()
        cand_attr["row_fp"] = cand_attr.apply(
            lambda r: _route_row_fp(r.get("catches"), r.get("yards"),
                                      r.get("tds"),
                                      r.get("position", ""), config),
            axis=1,
        )
        career_fp_per_route = (
            cand_attr.groupby(["receiver_player_id", "route"],
                                as_index=False)
            .agg(career_targets=("targets", "sum"),
                 career_fp=("row_fp", "sum"))
        )
        career_fp_per_route["career_fp_per_target"] = (
            career_fp_per_route["career_fp"]
            / career_fp_per_route["career_targets"]
        )

        # Player names for candidates
        if not player_route_df.empty:
            name_lookup = player_route_df[[
                "player_id", "player_display_name", "position",
            ]].drop_duplicates(subset="player_id")
        else:
            name_lookup = (
                attribution[["receiver_player_id",
                              "player_display_name", "position"]]
                .drop_duplicates(subset="receiver_player_id")
                .rename(columns={"receiver_player_id": "player_id"})
            )
        career_fp_per_route = career_fp_per_route.merge(
            name_lookup.rename(
                columns={"player_id": "receiver_player_id"}),
            on="receiver_player_id", how="left",
        )
        career_fp_per_route["origin"] = career_fp_per_route[
            "receiver_player_id"
        ].apply(lambda pid: "Incumbent"
                  if pid in incumbent_ids else "New (FA/trade)")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown(
                f"**📤 Departures** "
                f"({len(dep_summary)} significant)")
            if dep_summary.empty:
                st.caption("Nobody significant left.")
            else:
                disp = dep_summary[[
                    "player_display_name", "position",
                    "targets", "fp",
                ]].copy()
                disp.columns = ["Player", "Pos", "Targets",
                                  f"{config_name} FP"]
                st.dataframe(disp, use_container_width=True,
                                hide_index=True, height=240)

        with col_b:
            st.markdown(
                f"**🎯 Vacated demand by route** "
                f"({vacated['vacated_fp'].sum():.1f} {config_name} "
                "FP total)"
            )
            if vacated.empty:
                st.caption("No route demand vacated.")
            else:
                disp = vacated[[
                    "route", "vacated_targets", "vacated_fp",
                    "vacated_fp_pct",
                ]].copy()
                disp.columns = ["Route", "Targets",
                                  f"{config_name} FP",
                                  "% of team's FP"]
                disp[f"{config_name} FP"] = disp[
                    f"{config_name} FP"].round(1)
                disp["% of team's FP"] = disp[
                    "% of team's FP"].round(1)
                st.dataframe(disp, use_container_width=True,
                                hide_index=True, height=240)

        # Best fit per vacated route
        st.markdown(f"### 🎯 Best absorbers per vacated route — "
                      f"{config_name} conversion")
        st.caption(
            "For each route, the **top 3 candidates** ranked by "
            "career FP/target on that specific route. Tags show "
            "**Incumbent** (already on team) vs **New** (FA/trade). "
            "**Incumbent + elite FP/target = the alpha pick** the "
            "market hasn't yet priced into ADP."
        )
        fits = []
        for _, vrow in vacated.head(8).iterrows():
            if vrow["vacated_fp"] <= 0:
                continue
            cands = career_fp_per_route[
                (career_fp_per_route["route"] == vrow["route"])
                & (career_fp_per_route["career_targets"] >= 5)
            ].sort_values("career_fp_per_target", ascending=False)
            if cands.empty:
                continue
            for i, (_, c) in enumerate(cands.head(3).iterrows()):
                fits.append({
                    "Route": vrow["route"] if i == 0 else "",
                    f"Vacated FP": (f"{vrow['vacated_fp']:.1f}"
                                       if i == 0 else ""),
                    "Rank": i + 1,
                    "Origin": c["origin"],
                    "Candidate": c["player_display_name"],
                    "Pos": c["position"],
                    f"Career {config_name} FP/target":
                        f"{c['career_fp_per_target']:.2f}",
                    "Career targets on route":
                        int(c["career_targets"]),
                })
        if fits:
            st.dataframe(pd.DataFrame(fits),
                            use_container_width=True,
                            hide_index=True)
        else:
            st.info("No candidates with career history on the "
                      "vacated routes (rookies / low-volume only).")


# ══════════════════════════════════════════════════════════════════
#  Section 3: 🏆 Per-route FP conversion leaderboard
# ══════════════════════════════════════════════════════════════════

st.markdown("---")
st.header("🏆 Per-route FP conversion — league-wide")
st.markdown(
    "**The stat nobody else publishes.** For each route type, the "
    "league's top **fantasy-points-per-target converters** across "
    "their NFL careers. Bateman's career PPR/target on IN/DIG is "
    "**2.99** — top in the league. Use this to identify route "
    "specialists at the player level (great for finding "
    "DFS-stack pieces, dynasty trade targets, and waiver-wire "
    "specialists)."
)

if attribution.empty:
    st.info("Run scheme builders to populate this section.")
else:
    config = fs.CONFIG_BY_NAME[config_name]

    # Compute career FP per (player, route) league-wide
    full_attr = attribution.copy()
    full_attr["row_fp"] = full_attr.apply(
        lambda r: _route_row_fp(r.get("catches"), r.get("yards"),
                                  r.get("tds"),
                                  r.get("position", ""), config),
        axis=1,
    )
    leaderboard = (
        full_attr.groupby(
            ["receiver_player_id", "player_display_name", "position",
             "route"], as_index=False)
        .agg(targets=("targets", "sum"),
             fp=("row_fp", "sum"))
    )
    leaderboard["fp_per_target"] = (
        leaderboard["fp"] / leaderboard["targets"]
    )

    # Filter controls
    routes = sorted(leaderboard["route"].unique())
    if "" in routes:
        routes.remove("")
    col_r, col_p, col_n = st.columns([2, 2, 1])
    with col_r:
        selected_route = st.selectbox(
            "Route", routes,
            index=routes.index("IN/DIG") if "IN/DIG" in routes else 0,
            key="lb_route",
        )
    with col_p:
        position_filter = st.multiselect(
            "Positions", ["WR", "TE", "RB"],
            default=["WR", "TE"],
            key="lb_position",
        )
    with col_n:
        min_targets = st.slider(
            "Min career targets on route", 5, 100, 20,
            key="lb_min_targets",
        )

    sub = leaderboard[
        (leaderboard["route"] == selected_route)
        & (leaderboard["position"].isin(position_filter))
        & (leaderboard["targets"] >= min_targets)
    ].sort_values("fp_per_target", ascending=False)

    if sub.empty:
        st.info("No qualifying players. Lower the min-targets filter.")
    else:
        disp = sub.head(20)[[
            "player_display_name", "position", "targets",
            "fp", "fp_per_target",
        ]].copy()
        disp.columns = [
            "Player", "Pos", "Career targets",
            f"Career {config_name} FP",
            f"{config_name} FP / target",
        ]
        disp[f"Career {config_name} FP"] = disp[
            f"Career {config_name} FP"].round(1)
        disp[f"{config_name} FP / target"] = disp[
            f"{config_name} FP / target"].round(2)

        st.markdown(f"**Top 20 {config_name} converters on "
                      f"{selected_route} routes** "
                      f"(min {min_targets} career targets):")
        st.dataframe(disp, use_container_width=True,
                        hide_index=True, height=560)

        # Quick "narrative" callout
        leader = sub.iloc[0]
        st.success(
            f"**The {selected_route} {config_name} king:** "
            f"{leader['player_display_name']} "
            f"({leader['position']}) — "
            f"**{leader['fp_per_target']:.2f} FP/target** across "
            f"{int(leader['targets'])} career targets on this route."
        )

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
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

import lib_scoring as fs
from lib_gas_panels import load_gas_data
from lib_shared import inject_css
from lib_top_nav import render_home_button


REPO = Path(__file__).resolve().parent.parent
WEEKLY_PATH = REPO / "data" / "nfl_player_stats_weekly.parquet"
ADP_PATH = REPO / "data" / "fantasy" / "sleeper_adp.parquet"
ATTRIBUTION_PATH = REPO / "data" / "scheme" / "team_route_attribution.parquet"
TRANSITIONS_PATH = REPO / "data" / "scheme" / "roster_transitions.parquet"
PLAYER_ROUTE_PATH = REPO / "data" / "scheme" / "player_route_profile.parquet"
TEAM_QB_PATH = REPO / "data" / "scheme" / "team_qb_profile.parquet"
QB_TRAJECTORY_PATH = REPO / "data" / "scheme" / "qb_trajectory.parquet"

POSITION_TO_GAS = {"QB": "qb", "RB": "rb", "WR": "wr", "TE": "te"}


# ── Page config ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Fantasy", page_icon="🏆",
    layout="wide", initial_sidebar_state="expanded",
)
inject_css()


render_home_button()  # ← back to landing
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
def load_team_qb_profile() -> pd.DataFrame:
    if not TEAM_QB_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(TEAM_QB_PATH)


@st.cache_data(show_spinner=False)
def load_qb_trajectory() -> pd.DataFrame:
    if not QB_TRAJECTORY_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(QB_TRAJECTORY_PATH)


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
    """Aggregate weekly stats to ONE row per (player_id, season).

    Players who switch teams mid-season (trades, FA reassignments)
    have multiple weekly rows under different team codes. We sum
    stats across teams so each player gets a single season-total
    row. The 'team' column shows the team where the player played
    the most weeks (their primary team).
    """
    sum_cols = [
        "completions", "attempts", "passing_yards", "passing_tds",
        "passing_interceptions", "carries", "rushing_yards",
        "rushing_tds", "targets", "receptions", "receiving_yards",
        "receiving_tds", "fantasy_points", "fantasy_points_ppr",
    ]
    avail = [c for c in sum_cols if c in season_df.columns]
    grp = season_df.groupby(
        ["player_id", "player_display_name", "position", "season"],
        as_index=False)
    totals = grp[avail].sum()
    totals["games"] = grp.size()["size"].values

    # Primary team = team they played most weeks for that season
    team_weeks = (
        season_df.groupby(["player_id", "team"]).size()
                   .reset_index(name="weeks")
                   .sort_values("weeks", ascending=False)
                   .drop_duplicates("player_id")
                   [["player_id", "team"]]
    )
    totals = totals.merge(team_weeks, on="player_id", how="left")

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




# ── Hoisted helpers (used across tabs) ─────────────────────────

def _route_row_fp(catches, yards, tds, position, config):
    rec_value = config.reception
    if position == "TE" and config.te_premium_bonus > 0:
        rec_value += config.te_premium_bonus
    return ((catches or 0) * rec_value
            + (yards or 0) * config.rec_yard
            + (tds or 0) * config.rec_td)




@st.cache_data(show_spinner="Crunching league-wide alpha…")
def compute_league_wide_alpha(config_name: str) -> pd.DataFrame:
    """For every team, compute its top alpha candidates (incumbents +
    vet arrivals ranked by projected absorbed FP × QB-tendency match).
    Returns a single long-format DataFrame across all 32 teams.
    Cached so the heavy loop only runs once per scoring config."""
    config = fs.CONFIG_BY_NAME[config_name]
    attribution = pd.read_parquet(ATTRIBUTION_PATH)
    transitions = pd.read_parquet(TRANSITIONS_PATH)
    team_qb = (pd.read_parquet(TEAM_QB_PATH)
                  if TEAM_QB_PATH.exists() else pd.DataFrame())
    if PLAYER_ROUTE_PATH.exists():
        prdf = pd.read_parquet(PLAYER_ROUTE_PATH)
        name_lookup = prdf[[
            "player_id", "player_display_name", "position",
        ]].drop_duplicates(subset="player_id")
    else:
        name_lookup = (
            attribution[["receiver_player_id",
                              "player_display_name", "position"]]
            .drop_duplicates(subset="receiver_player_id")
            .rename(columns={"receiver_player_id": "player_id"})
        )

    # Vectorize: pre-compute per-row FP for the whole attribution table
    full_attr = attribution.copy()
    full_attr["row_fp"] = full_attr.apply(
        lambda r: _route_row_fp(
            r.get("catches"), r.get("yards"), r.get("tds"),
            r.get("position", ""), config),
        axis=1,
    )

    # Per-player cap as a fraction of team's total redistributable FP.
    # No single player realistically absorbs more than ~25% of a team's
    # vacated production — finite snap/target share + roster competition
    # always dilute one-player monopolies.
    PER_PLAYER_CAP_PCT = 0.25

    out_rows = []
    for team in transitions["team"].dropna().unique():
        team_trans = transitions[transitions["team"] == team]
        deps_df = team_trans[
            team_trans["transition_type"] == "departure"
        ]
        dep_ids = deps_df["player_id"].dropna().tolist()
        if not dep_ids:
            continue

        team_attr = full_attr[
            (full_attr["team"] == team) & (full_attr["season"] == 2025)
        ]
        if team_attr.empty:
            continue

        # Vacated by route — apply QB-tendency leakage. Routes the QB
        # doesn't throw at league-avg rate VANISH a portion: vacated
        # demand on those routes won't fully redistribute.
        qb_for_team_local = (
            team_qb[(team_qb["team"] == team)
                       & (team_qb["season"] == 2025)]
            if not team_qb.empty else pd.DataFrame()
        )
        qb_z_map = (
            dict(zip(qb_for_team_local["route"],
                       qb_for_team_local["share_z"]))
            if not qb_for_team_local.empty else {}
        )

        def _keep_factor(route: str) -> float:
            """0.4 + 0.3·qb_z, clipped to [0.25, 1.0]. QB throws above
            avg → keep most. QB throws below avg → big chunk vanishes."""
            qb_z = qb_z_map.get(route)
            if qb_z is None:
                return 0.7   # neutral default
            raw = 0.4 + 0.3 * float(qb_z)
            return max(0.25, min(1.0, raw))

        vac = (
            team_attr[team_attr["receiver_player_id"].isin(dep_ids)]
            .groupby("route", as_index=False)
            .agg(vacated_fp=("row_fp", "sum"))
        )
        vac = vac[vac["vacated_fp"] > 0].copy()
        if vac.empty:
            continue
        vac["redistributable_fp"] = vac.apply(
            lambda r: r["vacated_fp"] * _keep_factor(r["route"]),
            axis=1,
        )

        # Incumbents (last-year cohort minus departures)
        last_year = (
            team_attr.groupby(
                "receiver_player_id", as_index=False)
            .agg(prior=("targets", "sum"))
        )
        last_year = last_year[last_year["prior"] >= 10]
        incumbent_ids = [
            pid for pid in last_year["receiver_player_id"]
            if pid not in dep_ids
        ]
        vet_arr_ids = team_trans[
            (team_trans["transition_type"] == "arrival")
            & (team_trans["is_rookie"] == False)
        ]["player_id"].dropna().tolist()
        candidate_ids = incumbent_ids + vet_arr_ids
        if not candidate_ids:
            continue

        # Career FP/target on each route for candidates
        cand_attr = full_attr[
            full_attr["receiver_player_id"].isin(candidate_ids)
        ]
        cand_career = (
            cand_attr.groupby(
                ["receiver_player_id", "route"], as_index=False)
            .agg(career_targets=("targets", "sum"),
                 career_fp=("row_fp", "sum"))
        )
        cand_career["fpt"] = (
            cand_career["career_fp"]
            / cand_career["career_targets"].clip(lower=1)
        )

        # Total redistributable across the team — used for per-player cap
        team_total_redistributable = float(vac["redistributable_fp"].sum())
        per_player_cap = team_total_redistributable * PER_PLAYER_CAP_PCT

        # For each candidate, sum projected absorbed FP across vacated
        # routes where they're a top-3 specialist
        for cand_id in candidate_ids:
            relevant = []
            qb_friendly_count = 0
            qb_total_routes = 0
            for _, vrow in vac.iterrows():
                rcands = cand_career[
                    (cand_career["route"] == vrow["route"])
                    & (cand_career["career_targets"] >= 5)
                ].sort_values("fpt", ascending=False).head(3)
                if cand_id not in rcands["receiver_player_id"].values:
                    continue
                this_fpt = float(rcands[
                    rcands["receiver_player_id"] == cand_id
                ]["fpt"].iloc[0])
                top3_total = float(rcands["fpt"].sum() or 1)
                # Use REDISTRIBUTABLE FP (after QB leakage), not raw vacated
                est_absorbed = (
                    vrow["redistributable_fp"]
                    * (this_fpt / top3_total)
                )
                qb_z = qb_z_map.get(vrow["route"])
                if qb_z is not None:
                    qb_total_routes += 1
                    if qb_z >= 0.0:
                        qb_friendly_count += 1
                relevant.append({
                    "route": vrow["route"],
                    "vacated_fp": vrow["vacated_fp"],
                    "est_absorbed": est_absorbed,
                    "qb_z": qb_z,
                    "fpt": this_fpt,
                })
            if not relevant:
                continue

            raw_total = sum(r["est_absorbed"] for r in relevant)
            # Hard cap: no single player absorbs > PER_PLAYER_CAP_PCT
            # of team's redistributable FP. Realistic ceiling on
            # one-player monopolies.
            total_absorbed = min(raw_total, per_player_cap)
            qb_match_ratio = (
                qb_friendly_count / qb_total_routes
                if qb_total_routes > 0 else 0
            )
            if qb_match_ratio >= 0.6:
                verdict = "🚀 STOCK UP"
            elif qb_friendly_count == 0 and qb_total_routes > 0:
                verdict = "⚠️ CAUTION"
            else:
                verdict = "👀 WATCH"

            cand_meta = name_lookup[
                name_lookup["player_id"] == cand_id
            ]
            if cand_meta.empty:
                continue
            cname = cand_meta.iloc[0]["player_display_name"]
            cpos = cand_meta.iloc[0]["position"]
            origin = ("Incumbent" if cand_id in incumbent_ids
                        else "New (FA/trade)")

            top_routes = sorted(
                relevant, key=lambda r: -r["est_absorbed"]
            )[:3]
            top_routes_str = ", ".join(
                f"{r['route']} ({r['est_absorbed']:.0f})"
                for r in top_routes
            )

            out_rows.append({
                "team": team,
                "player_id": cand_id,
                "Player": cname,
                "Pos": cpos,
                "Team": team,
                "Origin": origin,
                "Projected absorbed FP": round(total_absorbed, 1),
                "Verdict": verdict,
                "QB-route match %": round(qb_match_ratio * 100, 0),
                "Top routes inherited": top_routes_str,
                "n_routes": len(relevant),
            })

    if not out_rows:
        return pd.DataFrame()
    df = pd.DataFrame(out_rows).sort_values(
        "Projected absorbed FP", ascending=False
    ).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df





# ══════════════════════════════════════════════════════════════════
#  Tabs — four alpha lenses
# ══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 ADP Triangulation",
    "🔍 Usage Autopsy",
    "🛫 QB Trajectory",
    "📈 Volume Alpha",
    "⚖️ Camp Battles",
    "🏆 Route Conversion",
])


with tab1:
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
    # Multi-team-season players (trades) have one GAS row per team. We
    # games-weight to a single per-(player, position) value so the
    # triangulation join doesn't multiply rows.
    gas_all = load_all_gas()
    gas_season_raw = gas_all[gas_all["season_year"] == prior_season].copy()

    # Games-weighted GAS per (player_id, position).
    if "games" in gas_season_raw.columns:
        gas_season_raw["games"] = gas_season_raw["games"].fillna(1).astype(float)
    else:
        gas_season_raw["games"] = 1.0
    gas_season_raw["weighted_gas"] = (
        gas_season_raw["gas_score"] * gas_season_raw["games"]
    )

    _agg = gas_season_raw.groupby(
        ["player_id", "position"], as_index=False
    ).agg(
        weighted_sum=("weighted_gas", "sum"),
        games_total=("games", "sum"),
    )
    _agg["gas_score"] = (
        _agg["weighted_sum"] / _agg["games_total"].clip(lower=1)
    )

    # Take label + confidence from the team where the player had most games
    _primary = (
        gas_season_raw.sort_values(
            ["player_id", "position", "games"],
            ascending=[True, True, False])
        .drop_duplicates(subset=["player_id", "position"])
        [["player_id", "position", "gas_label"]]
    )

    gas_for_season = _agg[["player_id", "position", "gas_score"]].merge(
        _primary, on=["player_id", "position"], how="left",
    )
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




with tab2:
    # ══════════════════════════════════════════════════════════════════
    #  Section 2: 🔍 USAGE AUTOPSY
    # ══════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.header("🔍 Usage Autopsy")
    st.markdown(
        "**A forensic per-team breakdown of where target volume went, "
        "what's missing, and whether the QB's throwing tendencies will "
        "actually fill the holes.** Three layers of signal:"
    )
    st.markdown(
        "- 📤 **What walked out** — departures × per-route load × FP\n"
        "- 🎯 **What's vacated by route** — total opportunity, "
        "broken down by route type\n"
        "- 🏈 **Will the QB even throw it?** — the team's primary QB's "
        "throwing tendency on each vacated route. **If the QB doesn't "
        "throw deep, vacated GO targets vanish — they don't redistribute.**"
    )


    attribution = load_attribution()
    transitions = load_transitions()
    player_route_df = load_player_route_profile()
    team_qb_df = load_team_qb_profile()

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

            # ── 🏆 LEAGUE-WIDE ALPHA LEADERBOARD ───────────────────────
            st.markdown(
                "### 🏆 League-wide alpha leaderboard")
            st.caption(
                "**Top stock-up candidates across all 32 teams**, ranked "
                "by **projected absorbed fantasy points** from this "
                "offseason's roster turnover. Filter by verdict to find "
                "the cleanest plays. **🚀 STOCK UP = the QB's career "
                "tendencies favor the routes this player would inherit.** "
                "Use the filters to narrow down."
            )
            league_alpha = compute_league_wide_alpha(config_name)
            if league_alpha.empty:
                st.info("No alpha candidates surfaced. Check that scheme "
                          "data is built.")
            else:
                la_col1, la_col2, la_col3, la_col4 = st.columns(
                    [1.5, 1, 1.5, 1])
                with la_col1:
                    pos_filter = st.multiselect(
                        "Position",
                        sorted(league_alpha["Pos"].dropna().unique()),
                        default=["WR", "TE"],
                        key="la_pos",
                    )
                with la_col2:
                    origin_filter = st.multiselect(
                        "Origin",
                        sorted(league_alpha["Origin"].unique()),
                        default=sorted(league_alpha["Origin"].unique()),
                        key="la_origin",
                    )
                with la_col3:
                    verdict_options = sorted(
                        league_alpha["Verdict"].unique())
                    verdict_filter = st.multiselect(
                        "Verdict", verdict_options,
                        default=verdict_options,
                        key="la_verdict",
                    )
                with la_col4:
                    min_absorbed = st.number_input(
                        f"Min projected {config_name} FP",
                        min_value=0.0, max_value=100.0, value=10.0,
                        step=5.0,
                        key="la_min_fp",
                    )

                filtered = league_alpha[
                    league_alpha["Pos"].isin(pos_filter)
                    & league_alpha["Origin"].isin(origin_filter)
                    & league_alpha["Verdict"].isin(verdict_filter)
                    & (league_alpha["Projected absorbed FP"]
                       >= min_absorbed)
                ].copy()
                filtered.insert(
                    0, "#", range(1, len(filtered) + 1))
                display_cols = [
                    "#", "Player", "Pos", "Team", "Origin",
                    "Verdict", "Projected absorbed FP",
                    "QB-route match %", "Top routes inherited",
                ]
                st.markdown(
                    f"**{len(filtered)} candidates** match your filters "
                    f"(of {len(league_alpha)} league-wide).")
                st.dataframe(
                    filtered[display_cols].head(40),
                    use_container_width=True, hide_index=True,
                    height=560,
                )

                # Headline narrative for the very top candidate
                if not filtered.empty:
                    top = filtered.iloc[0]
                    st.success(
                        f"**🥇 The single biggest opportunity:** "
                        f"{top['Player']} ({top['Pos']} · "
                        f"{top['Team']} · {top['Origin']}) — "
                        f"projected to absorb "
                        f"**{top['Projected absorbed FP']:.1f} {config_name} "
                        f"fantasy points** across "
                        f"vacated routes ({top['Top routes inherited']}). "
                        f"Verdict: {top['Verdict']}."
                    )

            st.markdown("---")
            st.markdown("### 🔍 Drill into a specific team")

            selected_scheme_team = st.selectbox(
                "Pick a team to drill into",
                team_options,
                index=0,
                help="Defaults to the biggest vacated-FP event of the offseason.",
            )

            # Team-anchored header so every section below is clearly
            # tied to the selected team — no guessing what we're looking at
            team_total_vacated = team_vacated[
                team_vacated["team"] == selected_scheme_team
            ]["vacated_fp"].iloc[0]
            st.markdown(
                f"<div style='background:#1f2937;border-left:5px solid "
                f"#fbbf24;padding:14px 20px;border-radius:6px;"
                f"margin:18px 0;color:white;'>"
                f"<div style='font-size:11px;letter-spacing:2px;"
                f"color:#fbbf24;font-weight:700;'>NOW ANALYZING</div>"
                f"<div style='font-size:24px;font-weight:800;"
                f"margin-top:4px;'>{selected_scheme_team} · 2025 → 2026 "
                f"offseason</div>"
                f"<div style='font-size:14px;color:#d1d5db;margin-top:4px;'>"
                f"<b>{team_total_vacated:.1f}</b> {config_name} fantasy "
                f"points walked out the door. Below: who left, what "
                f"routes are open, and which players on the current "
                f"roster are best positioned to absorb the demand.</div>"
                f"</div>",
                unsafe_allow_html=True,
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
                    f"**📤 {selected_scheme_team} departures** "
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
                # Enrich vacated demand with primary QB's throwing tendency
                # on each route — the cross-reference that says
                # "will the QB even throw it?"
                qb_for_team = pd.DataFrame()
                if not team_qb_df.empty:
                    qb_for_team = team_qb_df[
                        (team_qb_df["team"] == selected_scheme_team)
                        & (team_qb_df["season"] == 2025)
                    ]
                qb_name = (
                    qb_for_team["passer_player_name"].iloc[0]
                    if not qb_for_team.empty
                    and qb_for_team["passer_player_name"].notna().any()
                    else "—"
                )
                qb_lookup = (
                    qb_for_team[["route", "share", "share_z"]]
                    .rename(columns={
                        "share": "qb_share",
                        "share_z": "qb_share_z",
                    })
                    if not qb_for_team.empty else
                    pd.DataFrame(columns=["route", "qb_share",
                                                "qb_share_z"])
                )

                st.markdown(
                    f"**🎯 {selected_scheme_team} — vacated demand by route**"
                    f" ({vacated['vacated_fp'].sum():.1f} {config_name} "
                    f"FP total · QB: **{qb_name}**)"
                )
                if vacated.empty:
                    st.caption("No route demand vacated.")
                else:
                    disp = vacated.merge(qb_lookup, on="route", how="left")

                    # VACUUM / DISAPPEARING / BALANCED status per route
                    def _status(row):
                        fp = float(row["vacated_fp"])
                        qb_z = row.get("qb_share_z")
                        if fp <= 0:
                            return "—"
                        if pd.isna(qb_z):
                            return "?"
                        if qb_z >= 0.3:
                            return "🟢 VACUUM"   # QB throws it; will be filled
                        if qb_z >= -0.3:
                            return "🟡 NEUTRAL"
                        return "🔴 VANISHING"     # QB doesn't throw it

                    disp["status"] = disp.apply(_status, axis=1)
                    disp["qb_share_pct"] = (
                        disp["qb_share"].astype("Float64") * 100
                    ).round(1)

                    disp = disp[[
                        "route", "vacated_targets", "vacated_fp",
                        "vacated_fp_pct", "qb_share_pct", "status",
                    ]].copy()
                    disp.columns = ["Route", "Targets",
                                      f"{config_name} FP",
                                      "% of team's FP",
                                      "QB throws %", "Status"]
                    disp[f"{config_name} FP"] = disp[
                        f"{config_name} FP"].round(1)
                    disp["% of team's FP"] = disp[
                        "% of team's FP"].round(1)
                    st.dataframe(disp, use_container_width=True,
                                    hide_index=True, height=300)
                    st.caption(
                        "**🟢 VACUUM** = QB throws this route above-avg, "
                        "vacated demand WILL be filled (real opportunity). "
                        "**🟡 NEUTRAL** = league-average QB tendency. "
                        "**🔴 VANISHING** = QB doesn't throw this route — "
                        "vacated targets won't redistribute, they vanish."
                    )

            # ── 📋 AUTO-GENERATED TAKEAWAYS (top 3) ──────────────────
            # For each candidate, estimate total projected absorbed FP
            # across all vacated routes where they're a top-3 specialist.
            # Surface the top 3 candidates as a "Stock UP" narrative card.
            candidate_summaries = []
            for cand_id in (incumbent_ids + vet_arr_ids):
                cand_name_row = name_lookup[
                    name_lookup["player_id"] == cand_id
                ]
                if cand_name_row.empty:
                    continue
                cname = cand_name_row.iloc[0]["player_display_name"]
                cpos = cand_name_row.iloc[0]["position"]
                origin_tag = ("Incumbent" if cand_id in incumbent_ids
                                else "New (FA/trade)")
                origin_emoji = ("🟢" if origin_tag == "Incumbent"
                                  else "🆕")

                # Routes where this candidate is top-3 specialist among
                # the team's absorption pool, AND there's real vacated
                # demand on that route AND QB tendency supports it
                relevant_routes = []
                for _, vrow in vacated.iterrows():
                    if vrow["vacated_fp"] <= 0:
                        continue
                    rcands = career_fp_per_route[
                        (career_fp_per_route["route"] == vrow["route"])
                        & (career_fp_per_route["career_targets"] >= 5)
                    ].sort_values("career_fp_per_target", ascending=False)
                    if rcands.empty:
                        continue
                    top3 = rcands.head(3)
                    if cand_id not in top3["receiver_player_id"].values:
                        continue
                    # QB tendency on this route
                    qb_z = None
                    if not qb_for_team.empty:
                        qbm = qb_for_team[
                            qb_for_team["route"] == vrow["route"]]
                        if not qbm.empty:
                            qb_z = float(qbm["share_z"].iloc[0])
                    # Estimated absorbed FP: career FP/target × (vacated
                    # targets × this candidate's share within top 3)
                    this_fpt = float(rcands[
                        rcands["receiver_player_id"] == cand_id
                    ]["career_fp_per_target"].iloc[0])
                    total_top3_fpt = top3["career_fp_per_target"].sum()
                    share_of_top3 = (this_fpt / total_top3_fpt
                                        if total_top3_fpt > 0 else 0)
                    est_absorbed_fp = (
                        vrow["vacated_fp"] * share_of_top3
                    )
                    relevant_routes.append({
                        "route": vrow["route"],
                        "vacated_fp": float(vrow["vacated_fp"]),
                        "career_fp_per_target": this_fpt,
                        "est_absorbed_fp": est_absorbed_fp,
                        "qb_z": qb_z,
                    })

                if not relevant_routes:
                    continue
                total_absorbed = sum(r["est_absorbed_fp"]
                                        for r in relevant_routes)
                candidate_summaries.append({
                    "player_id": cand_id,
                    "name": cname,
                    "pos": cpos,
                    "origin_tag": origin_tag,
                    "origin_emoji": origin_emoji,
                    "total_absorbed_fp": total_absorbed,
                    "routes": sorted(relevant_routes,
                                        key=lambda r: -r["est_absorbed_fp"]),
                })

            # Rank by total projected absorbed FP, top 3
            candidate_summaries.sort(
                key=lambda s: -s["total_absorbed_fp"])
            top_takeaways = candidate_summaries[:3]

            if top_takeaways:
                st.markdown(
                    f"### 📋 Top takeaways · {selected_scheme_team} "
                    f"({config_name})")
                st.caption(
                    "Auto-generated observations from the data. "
                    "**Top candidates ranked by projected absorbed "
                    "fantasy points** (their share of top-fit slots × "
                    "vacated FP, weighted by career conversion rate)."
                )
                for i, t in enumerate(top_takeaways):
                    rank_emoji = ["🥇", "🥈", "🥉"][i]
                    primary_routes = t["routes"][:3]
                    routes_str = ", ".join(
                        f"{r['route']} ({r['est_absorbed_fp']:.1f} {config_name})"
                        for r in primary_routes
                    )
                    # Determine narrative label based on route+QB match
                    qb_friendly = sum(
                        1 for r in primary_routes
                        if r["qb_z"] is not None and r["qb_z"] >= 0.0
                    )
                    if qb_friendly >= len(primary_routes) * 0.6:
                        verdict = "🚀 **STOCK UP**"
                        verdict_color = "#16a34a"
                    elif qb_friendly == 0:
                        verdict = "⚠️ **CAUTION** (QB tendencies don't favor)"
                        verdict_color = "#f59e0b"
                    else:
                        verdict = "👀 **WATCH**"
                        verdict_color = "#3b82f6"

                    st.markdown(
                        f"""<div style='border-left:4px solid {verdict_color};
                            background:rgba(31,41,55,0.4);padding:14px 18px;
                            border-radius:6px;margin:10px 0;'>
                            <div style='font-size:13px;font-weight:700;
                                        color:#fbbf24;letter-spacing:1px;'>
                                {rank_emoji} {verdict}
                            </div>
                            <div style='font-size:18px;font-weight:800;
                                        color:#f3f4f6;margin-top:4px;'>
                                {t['origin_emoji']} {t['name']}
                                <span style='font-weight:500;color:#9ca3af;
                                              font-size:14px;'>
                                  ({t['pos']} · {t['origin_tag']})
                                </span>
                            </div>
                            <div style='font-size:14px;color:#d1d5db;
                                        margin-top:8px;line-height:1.5;'>
                                <b>Projected absorbed: ~{t['total_absorbed_fp']:.1f}
                                  {config_name} FP</b> across {len(t['routes'])}
                                  vacated routes.
                                <br>Top route loads: {routes_str}.
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            # Best fit per vacated route
            st.markdown(
                f"### 🎯 {selected_scheme_team} — who absorbs each "
                f"vacated route? ({config_name} conversion)"
            )
            st.caption(
                f"For each route the **{selected_scheme_team}** lost "
                "volume on, the top 3 players (current roster + new "
                "vet adds) ranked by **career fantasy points per target "
                "on that exact route**. "
                "🟢 **Incumbent** = already on the team last year (stock-up "
                "candidates). 🆕 **New** = arrived this offseason via "
                "FA / trade. **An incumbent with elite FP/target is the "
                "buy signal that ADP hasn't caught yet.**"
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




with tab3:
    # ══════════════════════════════════════════════════════════════════
    #  Section 2.5: 🛫 QB TRAJECTORY TAILWINDS
    # ══════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.header("🛫 QB Trajectory Tailwinds")
    st.markdown(
        "**Vacancy alpha is only one source of fantasy edge.** Receivers "
        "also gain when their QB is about to play *better*. This module "
        "scores every team's primary QB on three trajectory axes — "
        "**Y2/Y3 sophomore leap**, **return from injury**, **aging "
        "decline** — and surfaces the receivers attached to rising QBs."
    )

    qb_traj_df = load_qb_trajectory()
    if qb_traj_df.empty:
        st.info(
            "QB trajectory data not built yet. Run:\n"
            "```\npython tools/build_qb_trajectory.py\n```"
        )
    else:
        # ── QB-level table ───────────────────────────────────────────
        st.markdown("### 📋 Per-team QB trajectory grades")
        st.caption(
            "Sorted by projected GAS delta. Y2 leap = rookies stepping "
            "into year two. Injury recovery = QB missed games last year "
            "AND has a higher career peak. Aging drag = NFL years_exp ≥ "
            "12 with declining trajectory."
        )
        show = qb_traj_df.copy()
        show = show.sort_values("trajectory_delta", ascending=False)
        show_cols = [
            "trajectory_label", "team", "qb_name", "nfl_years_exp",
            "last_games", "last_gas", "peak_gas", "projected_gas",
            "trajectory_delta", "rationale",
        ]
        show.columns = list(show.columns)  # keep originals
        rename = {
            "trajectory_label": "Verdict",
            "team": "Team",
            "qb_name": "QB",
            "nfl_years_exp": "NFL yrs",
            "last_games": "2025 games",
            "last_gas": "2025 GAS",
            "peak_gas": "Peak GAS",
            "projected_gas": "Proj 2026 GAS",
            "trajectory_delta": "Δ GAS",
            "rationale": "Why",
        }
        st.dataframe(
            show[show_cols].rename(columns=rename),
            use_container_width=True, hide_index=True, height=400,
        )

        st.caption(
            "⚠️ **Caveats:** the model assumes the 2025 primary QB is "
            "still the 2026 primary. Teams with QB battles or retirements "
            "(PIT post-Rodgers, WAS Daniels return, ARI Murray return) "
            "should be treated with extra uncertainty. Camp Battles "
            "module will let you override these assumptions."
        )

        # ── Receiver leaderboard: receivers on rising-QB teams ──────
        st.markdown("### 🎯 Receivers attached to rising-QB teams")
        st.caption(
            "Ranks each rising QB's top 2025 receivers by **estimated "
            "fantasy-point tailwind** = (2025 PPR FP) × (Δ GAS / 100). "
            "A +27 GAS leap (Lamar) on a 150-FP receiver projects ~+40 "
            "PPR of QB-driven upside before any vacancy redistribution."
        )

        rising = qb_traj_df[qb_traj_df["trajectory_delta"] > 0]
        if rising.empty:
            st.info("No teams with rising QB trajectories.")
        else:
            # Build last-year FP per receiver per team from attribution
            attr_local = load_attribution()
            config_local = fs.CONFIG_BY_NAME[config_name]
            attr_2025 = attr_local[attr_local["season"] == 2025].copy()
            attr_2025["row_fp"] = attr_2025.apply(
                lambda r: _route_row_fp(
                    r.get("catches"), r.get("yards"), r.get("tds"),
                    r.get("position", ""), config_local),
                axis=1,
            )
            rec_team_fp = (
                attr_2025.groupby(
                    ["team", "receiver_player_id",
                     "player_display_name", "position"],
                    as_index=False)["row_fp"].sum()
                .rename(columns={"row_fp": "fp_2025"})
            )
            # Only keep receivers with meaningful prior-year volume
            rec_team_fp = rec_team_fp[rec_team_fp["fp_2025"] >= 20]

            merged = rec_team_fp.merge(
                rising[["team", "qb_name", "trajectory_delta",
                           "trajectory_label"]],
                on="team", how="inner",
            )
            merged["tailwind_fp"] = (
                merged["fp_2025"] * (merged["trajectory_delta"] / 100.0)
            ).round(1)
            merged = merged.sort_values(
                "tailwind_fp", ascending=False).reset_index(drop=True)

            # Filter UI
            col1, col2 = st.columns(2)
            with col1:
                pos_filter = st.multiselect(
                    "Position", ["WR", "TE", "RB"],
                    default=["WR", "TE", "RB"], key="qb_traj_pos")
            with col2:
                min_tail = st.slider(
                    "Min tailwind FP", 0, 30, 5, step=1,
                    key="qb_traj_mintail")

            filt = merged[
                merged["position"].isin(pos_filter)
                & (merged["tailwind_fp"] >= min_tail)
            ].copy()
            filt.insert(0, "Rank", range(1, len(filt) + 1))
            display = filt[[
                "Rank", "player_display_name", "position", "team",
                "qb_name", "trajectory_label", "fp_2025",
                "trajectory_delta", "tailwind_fp",
            ]].rename(columns={
                "player_display_name": "Player",
                "position": "Pos",
                "team": "Team",
                "qb_name": "QB",
                "trajectory_label": "QB Verdict",
                "fp_2025": "2025 PPR FP",
                "trajectory_delta": "Δ GAS",
                "tailwind_fp": "Est. tailwind PPR",
            })
            st.dataframe(display, use_container_width=True,
                            hide_index=True, height=560)




with tab4:
    # ══════════════════════════════════════════════════════════════════
    #  Section 4: 📈 VOLUME AMPLIFICATION ALPHA
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("📈 Volume Amplification Alpha")
    st.markdown(
        "**Receivers gain when their team simply throws the ball more "
        "total.** Defensive PPG-allowed regresses to the league mean "
        "(empirically ~60%/year) — teams whose defense was *elite* in "
        "2025 project to give up more points in 2026, trail more, and "
        "pass more. Conversely, teams with bad defenses regress UP, "
        "lead more, run more.\n\n"
        "This is the lens that catches **DEN** alpha — Bo Nix steady, "
        "Broncos defense regresses → more passing volume → receivers up."
    )

    VOLUME_PATH = REPO / "data" / "scheme" / "volume_alpha.parquet"

    @st.cache_data(show_spinner=False)
    def _load_volume() -> pd.DataFrame:
        if not VOLUME_PATH.exists():
            return pd.DataFrame()
        return pd.read_parquet(VOLUME_PATH)

    vdf = _load_volume()
    if vdf.empty:
        st.info(
            "Volume Alpha not built yet. Run:\n"
            "```\npython tools/build_volume_alpha.py\n```"
        )
    else:
        st.markdown("### 📋 Per-team projected 2026 volume")
        st.caption(
            "Sorted by projected delta in pass attempts. Sign tells "
            "you direction (more / fewer attempts), magnitude is the "
            "estimated raw attempt count over a 17-game season."
        )
        rename = {
            "volume_label": "Verdict",
            "team": "Team",
            "pass_attempts_2025": "2025 attempts",
            "points_allowed_2025": "2025 PPG allowed",
            "league_avg_points_allowed": "Lg avg PPG",
            "def_regression_pts": "Δ to mean (PPG)",
            "attempts_delta": "Δ attempts",
            "proj_pass_attempts_2026": "Proj 2026 attempts",
            "rationale": "Why",
        }
        cols = ["volume_label", "team", "pass_attempts_2025",
                "points_allowed_2025", "league_avg_points_allowed",
                "def_regression_pts", "proj_pass_attempts_2026",
                "attempts_delta", "rationale"]
        st.dataframe(
            vdf[cols].rename(columns=rename),
            use_container_width=True, hide_index=True, height=560,
        )

        # ── Receiver-level Volume Alpha leaderboard ───────────────
        st.markdown("### 🎯 Receivers benefiting from team volume")
        st.caption(
            "For each rising-volume team, project the extra pass "
            "attempts × the receiver's 2025 target share × their "
            "career PPR/target conversion. Receivers on declining-"
            "volume teams get a negative tailwind."
        )

        attr_local_v = load_attribution()
        config_local_v = fs.CONFIG_BY_NAME[config_name]
        attr_25 = attr_local_v[attr_local_v["season"] == 2025].copy()
        attr_25["row_fp"] = attr_25.apply(
            lambda r: _route_row_fp(
                r.get("catches"), r.get("yards"), r.get("tds"),
                r.get("position", ""), config_local_v),
            axis=1,
        )
        # Per-receiver: 2025 team targets and career FP/target
        rec_team_25 = (
            attr_25.groupby(
                ["team", "receiver_player_id",
                 "player_display_name", "position"], as_index=False)
            .agg(targets_2025=("targets", "sum"))
        )
        # Career rates across all available seasons. Build on a fresh
        # frame with row_fp computed; attr_local_v itself doesn't carry
        # row_fp (only the 2025-filtered attr_25 does).
        attr_full = attr_local_v.copy()
        attr_full["row_fp"] = attr_full.apply(
            lambda r: _route_row_fp(
                r.get("catches"), r.get("yards"), r.get("tds"),
                r.get("position", ""), config_local_v),
            axis=1,
        )
        rec_career = (
            attr_full.groupby("receiver_player_id", as_index=False)
            .agg(career_targets=("targets", "sum"),
                 career_fp=("row_fp", "sum"))
        )
        rec_career["fp_per_target"] = (
            rec_career["career_fp"]
            / rec_career["career_targets"].clip(lower=1)
        )

        # Team total 2025 targets to compute share
        team_total_25 = (
            attr_25.groupby("team", as_index=False)
            .agg(team_targets_2025=("targets", "sum"))
        )
        rec = rec_team_25.merge(team_total_25, on="team", how="left")
        rec["target_share"] = (
            rec["targets_2025"] / rec["team_targets_2025"].clip(lower=1)
        )
        # Filter to meaningful share
        rec = rec[rec["targets_2025"] >= 10]

        # Bring in team volume delta
        rec = rec.merge(
            vdf[["team", "attempts_delta", "volume_label"]],
            on="team", how="left",
        )
        # Pass attempts ≈ targets (close enough; sacks/scrambles
        # subtract from attempts but not from targets, so this slightly
        # underestimates the receiver effect — fine for v1).
        rec["projected_added_targets"] = (
            rec["attempts_delta"] * rec["target_share"]
        ).round(1)

        # Career FP/target conversion
        rec = rec.merge(
            rec_career[["receiver_player_id", "fp_per_target"]],
            on="receiver_player_id", how="left",
        )
        rec["volume_tailwind_fp"] = (
            rec["projected_added_targets"]
            * rec["fp_per_target"].fillna(1.5)
        ).round(1)

        rec = rec.sort_values(
            "volume_tailwind_fp", ascending=False).reset_index(drop=True)

        # ── Filter UI ─────────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            pos_pick_v = st.multiselect(
                "Position", ["WR", "TE", "RB"],
                default=["WR", "TE", "RB"], key="vol_alpha_pos")
        with col_b:
            min_share = st.slider(
                "Min 2025 target share (%)", 0, 30, 5, step=1,
                key="vol_alpha_min_share")

        filt = rec[
            rec["position"].isin(pos_pick_v)
            & (rec["target_share"] * 100 >= min_share)
        ].copy()
        filt.insert(0, "Rank", range(1, len(filt) + 1))
        show = filt[[
            "Rank", "player_display_name", "position", "team",
            "volume_label", "targets_2025", "target_share",
            "attempts_delta", "projected_added_targets",
            "fp_per_target", "volume_tailwind_fp",
        ]].rename(columns={
            "player_display_name": "Player",
            "position": "Pos",
            "team": "Team",
            "volume_label": "Team Verdict",
            "targets_2025": "2025 Tgts",
            "target_share": "Share",
            "attempts_delta": "Team Δ Att",
            "projected_added_targets": "Δ Tgts",
            "fp_per_target": "Career FP/Tgt",
            "volume_tailwind_fp": "Tailwind FP",
        })
        # Format Share as percentage in display
        show["Share"] = (show["Share"] * 100).round(1)
        st.dataframe(
            show, use_container_width=True, hide_index=True,
            height=560,
        )


with tab5:
    # ══════════════════════════════════════════════════════════════════
    #  Section 5: ⚖️ CAMP BATTLES — user-overridable QB picks
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.header("⚖️ Camp Battles")
    st.markdown(
        "**The model assumes the 2025 primary QB is also the 2026 "
        "primary.** That breaks for teams in transition: Mariota was "
        "WAS's primary because Daniels was hurt, Brissett got ARI's "
        "snaps because Murray was hurt, Rodgers's PIT one-year deal "
        "is up. Pick the 2026 starter you actually expect — the QB "
        "Trajectory leaderboard re-projects under your assumption."
    )

    QB_GAS_PATH = REPO / "data" / "qb_gas_seasons.parquet"

    @st.cache_data(show_spinner=False)
    def _load_qb_gas_seasons() -> pd.DataFrame:
        if not QB_GAS_PATH.exists():
            return pd.DataFrame()
        return pd.read_parquet(QB_GAS_PATH)

    @st.cache_data(show_spinner=False)
    def _load_team_qbs_roster() -> pd.DataFrame:
        """All QBs on each team's CURRENT (2026 offseason) roster
        per Sleeper. Using Sleeper rather than nflverse rosters
        because the latter is a 2025-season snapshot — Murray was
        traded to MIN this offseason, but nflverse still shows him
        on ARI. Sleeper reflects current depth charts.
        """
        adp = load_adp()
        if adp.empty:
            return pd.DataFrame()
        qbs = adp[adp["position"] == "QB"].dropna(
            subset=["team"]).copy()
        # Sleeper rookies often have no gsis_id (our crosswalk only
        # covers prior-NFL players). Drop them for now — they have
        # no career data anyway, so they can't drive a trajectory
        # projection. v2: handle rookies as "no-data" picks.
        qbs = qbs.rename(columns={"player_id": "gsis_id"})
        qbs = qbs.dropna(subset=["gsis_id"])
        qbs["status"] = "ACT"
        return qbs[["team", "gsis_id", "full_name", "status",
                    "years_exp"]].copy()

    qb_gas_full = _load_qb_gas_seasons()
    team_qbs = _load_team_qbs_roster()
    traj_df_local = load_qb_trajectory()

    if (qb_gas_full.empty or team_qbs.empty
            or traj_df_local.empty):
        st.info(
            "Upstream QB data missing. Run:\n"
            "```\npython tools/build_qb_trajectory.py\n```"
        )
    else:
        # ── Trajectory computation for an arbitrary picked QB ─────
        def _compute_trajectory_pick(qb_id: str, qb_name: str,
                                       team: str,
                                       nfl_yrs: float) -> dict:
            """Mirror tools/build_qb_trajectory.py for one (qb, team)
            pick. Returns the same row schema as qb_trajectory.parquet
            so it can drop into the existing display table."""
            INJURY_GAMES_THRESHOLD = 16
            PEAK_RECOVERY_RATIO = 0.85
            AGING_VET_YEARS = 12
            DEEP_VET_YEARS = 14

            def _y2_leap(g):
                return 15.0 if g < 50 else (10.0 if g < 60 else 5.0)

            def _y3_leap(g):
                return 7.0 if g < 50 else (4.0 if g < 60 else 2.0)

            career = qb_gas_full[
                qb_gas_full["player_id"] == qb_id
            ].sort_values("season_year")
            if career.empty:
                return {
                    "team": team, "qb_player_id": qb_id,
                    "qb_name": qb_name, "n_seasons": 0,
                    "nfl_years_exp": nfl_yrs,
                    "last_games": 0,
                    "last_gas": float("nan"),
                    "peak_gas": float("nan"),
                    "y2_leap_bump": 0.0, "y3_leap_bump": 0.0,
                    "injury_recovery_bump": 0.0, "aging_drag": 0.0,
                    "projected_gas": float("nan"),
                    "trajectory_delta": 0.0,
                    "trajectory_label": "❓ ROOKIE / NO DATA",
                    "rationale": "No NFL GAS career data on file "
                                  "(rookie or never started)",
                }
            last_row = career.iloc[-1]
            last_gas = float(last_row["gas_score"])
            last_games = int(last_row["games"])
            n_seasons = len(career)
            peak_gas = float(career["gas_score"].max())

            y2 = y3 = 0.0
            if nfl_yrs <= 1 and n_seasons == 1:
                y2 = _y2_leap(last_gas)
            elif nfl_yrs <= 2 and n_seasons == 2:
                y3 = _y3_leap(last_gas)
            peak_gap = peak_gas - last_gas
            inj = 0.0
            if (last_games < INJURY_GAMES_THRESHOLD
                    and peak_gap >= 5 and peak_gas > 60):
                inj = peak_gap * PEAK_RECOVERY_RATIO
            aging = 0.0
            if nfl_yrs >= DEEP_VET_YEARS:
                aging = -3.0
            elif nfl_yrs >= AGING_VET_YEARS:
                aging = -2.0
            if (nfl_yrs >= AGING_VET_YEARS
                    and last_gas < peak_gas - 5):
                aging -= 2.0
            proj = last_gas + y2 + y3 + inj + aging
            delta = proj - last_gas
            label = ("🚀 RISING" if delta >= 5 else
                     "⬇️ DECLINING" if delta <= -3 else
                     "➡️ STABLE")
            bits = []
            if y2: bits.append(f"Y2 leap +{y2:.0f}")
            if y3: bits.append(f"Y3 step +{y3:.0f}")
            if inj: bits.append(f"injury recovery +{inj:.1f}")
            if aging: bits.append(f"aging {aging:.0f}")
            if not bits: bits.append("stable trajectory")
            return {
                "team": team, "qb_player_id": qb_id,
                "qb_name": qb_name, "n_seasons": n_seasons,
                "nfl_years_exp": nfl_yrs,
                "last_games": last_games,
                "last_gas": round(last_gas, 1),
                "peak_gas": round(peak_gas, 1),
                "y2_leap_bump": round(y2, 1),
                "y3_leap_bump": round(y3, 1),
                "injury_recovery_bump": round(inj, 1),
                "aging_drag": round(aging, 1),
                "projected_gas": round(proj, 1),
                "trajectory_delta": round(delta, 1),
                "trajectory_label": label,
                "rationale": "; ".join(bits),
            }

        # ── Build per-team option list from CURRENT Sleeper roster ─
        opts_by_team: dict[str, list[tuple[str, str, float]]] = {}
        for team, sub in team_qbs.groupby("team"):
            opts = list(zip(sub["gsis_id"], sub["full_name"],
                            sub["years_exp"].fillna(0).astype(float)))
            opts.sort(key=lambda o: -o[2])  # vets first
            opts_by_team[team] = opts

        # ── Init defaults — 2025 primary IF still on the roster ───
        # Sleeper reflects current 2026 offseason rosters, so primaries
        # who got traded (Murray ARI→MIN) or retired (Rodgers PIT)
        # won't be on the team anymore. In those cases, default pick
        # falls to the most senior current QB on the team.
        primary_2025 = dict(zip(traj_df_local["team"],
                                  traj_df_local["qb_player_id"]))
        primary_2025_name = dict(zip(traj_df_local["team"],
                                       traj_df_local["qb_name"]))
        defaults: dict[str, str] = {}
        primary_departed: dict[str, str] = {}  # team → old primary name
        for team, opts in opts_by_team.items():
            if not opts:
                continue
            current_ids = {o[0] for o in opts}
            old_primary = primary_2025.get(team)
            if old_primary and old_primary in current_ids:
                defaults[team] = old_primary
            else:
                # 2025 primary departed (trade / retirement / cut)
                defaults[team] = opts[0][0]   # most senior current QB
                if old_primary:
                    primary_departed[team] = (
                        primary_2025_name.get(team, "?"))

        if "qb_picks" not in st.session_state:
            st.session_state["qb_picks"] = {}
        for t, qid in defaults.items():
            st.session_state["qb_picks"].setdefault(t, qid)

        # ── Auto-detect "contested" teams ─────────────────────────
        # Three flags:
        #   1. 2025 primary departed (trade / retirement)
        #   2. 2025 primary nfl_years_exp >= 14 (retirement risk)
        #   3. 2025 primary missed >6 games last year (was a fill-in)
        #   4. Roster has a vet (years_exp >= 5) other than the
        #      2025 primary — meaningful backup or new acquisition
        contested_teams = []
        for team, opts in opts_by_team.items():
            if team in primary_departed:
                contested_teams.append(team)
                continue
            row = traj_df_local[traj_df_local["team"] == team]
            if row.empty:
                continue
            row = row.iloc[0]
            primary_yrs = float(row["nfl_years_exp"] or 0)
            primary_games = int(row["last_games"])
            others = [o for o in opts if o[0] != row["qb_player_id"]]
            n_vet_others = sum(1 for o in others if o[2] >= 5)
            if (primary_yrs >= 14
                    or primary_games <= 10
                    or n_vet_others >= 1):
                contested_teams.append(team)

        st.markdown("### 🏈 Pick the 2026 starting QB per team")
        st.caption(
            f"**{len(contested_teams)} teams flagged as contested** "
            "(aging vet, injury fill-in, primary departed, or vet "
            "backup on roster). Override below; all other teams "
            "default to their 2025 primary. Picks persist for this "
            "session only."
        )

        if primary_departed:
            departed_lines = [
                f"**{old_name}** ({tm}) — pick a 2026 starter"
                for tm, old_name in primary_departed.items()
            ]
            st.warning(
                "🚨 **2025 primary no longer on team's roster:**\n\n"
                + "\n\n".join(f"• {line}" for line in departed_lines)
            )

        with st.expander(f"Show all 32 teams (not just contested)",
                            expanded=False):
            show_all = st.checkbox("Show every team", value=False,
                                       key="cb_show_all")

        teams_to_render = (sorted(team_qbs["team"].unique())
                              if st.session_state.get("cb_show_all")
                              else sorted(contested_teams))

        # Render in 3-column grid
        n_cols = 3
        rows_needed = (len(teams_to_render) + n_cols - 1) // n_cols
        idx = 0
        for _ in range(rows_needed):
            cols = st.columns(n_cols)
            for c in cols:
                if idx >= len(teams_to_render):
                    break
                team = teams_to_render[idx]
                opts = opts_by_team.get(team, [])
                if not opts:
                    idx += 1
                    continue
                primary_id = defaults.get(team)
                primary_name_short = traj_df_local[
                    traj_df_local["team"] == team
                ]["qb_name"].iloc[0]
                # Build display strings for the dropdown
                labels = []
                ids = []
                for gid, name, yrs in opts:
                    star = "⭐ " if gid == primary_id else ""
                    labels.append(f"{star}{name} ({int(yrs)} yr)")
                    ids.append(gid)
                # Find current pick index
                cur = st.session_state["qb_picks"].get(
                    team, primary_id)
                try:
                    default_idx = ids.index(cur)
                except ValueError:
                    default_idx = 0
                with c:
                    pick = c.selectbox(
                        f"**{team}** (2025: {primary_name_short})",
                        options=range(len(labels)),
                        format_func=lambda i, lab=labels: lab[i],
                        index=default_idx,
                        key=f"qbpick_{team}",
                    )
                    st.session_state["qb_picks"][team] = ids[pick]
                idx += 1

        # ── Recompute trajectory under user picks ─────────────────
        st.markdown("### 📋 QB Trajectory under your picks")
        modified_rows = []
        for team in sorted(team_qbs["team"].unique()):
            picked_id = st.session_state["qb_picks"].get(team)
            if not picked_id:
                continue
            # Lookup name + years_exp
            roster_match = team_qbs[
                (team_qbs["team"] == team)
                & (team_qbs["gsis_id"] == picked_id)
            ]
            if roster_match.empty:
                # fallback to traj_df_local
                m = traj_df_local[
                    traj_df_local["qb_player_id"] == picked_id
                ]
                if m.empty:
                    continue
                qb_name = m.iloc[0]["qb_name"]
                nfl_yrs = float(m.iloc[0]["nfl_years_exp"] or 0)
            else:
                qb_name = roster_match.iloc[0]["full_name"]
                nfl_yrs = float(
                    roster_match.iloc[0]["years_exp"] or 0)
            modified_rows.append(
                _compute_trajectory_pick(
                    picked_id, qb_name, team, nfl_yrs))

        modified_df = pd.DataFrame(modified_rows).sort_values(
            "trajectory_delta", ascending=False)

        # Side-by-side: only highlight teams where the pick changed
        changed = modified_df[
            modified_df["qb_player_id"]
            != modified_df["team"].map(defaults)
        ]
        if not changed.empty:
            st.success(
                f"You overrode {len(changed)} team(s). The QB "
                "Trajectory leaderboard now reflects those picks."
            )
        else:
            st.caption(
                "No overrides yet — the table below mirrors the "
                "default QB Trajectory tab."
            )

        # Display the (possibly-modified) trajectory table
        rename_cb = {
            "trajectory_label": "Verdict",
            "team": "Team",
            "qb_name": "QB",
            "nfl_years_exp": "NFL yrs",
            "last_games": "Last games",
            "last_gas": "Last GAS",
            "peak_gas": "Peak GAS",
            "projected_gas": "Proj 2026 GAS",
            "trajectory_delta": "Δ GAS",
            "rationale": "Why",
        }
        show_cb = ["trajectory_label", "team", "qb_name",
                    "nfl_years_exp", "last_games", "last_gas",
                    "peak_gas", "projected_gas",
                    "trajectory_delta", "rationale"]
        st.dataframe(
            modified_df[show_cb].rename(columns=rename_cb),
            use_container_width=True, hide_index=True, height=560,
        )

        st.caption(
            "💡 Receiver tailwind tables on **🛫 QB Trajectory** and "
            "**📈 Volume Alpha** still use the default primary-QB "
            "assumption. v2 will pipe Camp Battles picks through "
            "those leaderboards too."
        )

        # ══════════════════════════════════════════════════════════
        #  Receiver/TE camp battles — WR1 / WR2 / WR3 / TE1 picker
        # ══════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 🎯 WR1 / WR2 / WR3 + TE1 picker")
        st.markdown(
            "Receiver and TE roles are contested any time a team had "
            "free-agent arrivals, drafted a meaningful rookie, or had "
            "a star return from injury. Pick who you think wins each "
            "slot — defaults to the leading 2025 receiver still on "
            "the roster. **Picks don't yet propagate to the Usage "
            "Autopsy / Volume Alpha leaderboards (v2 work).**"
        )

        # Pull current Sleeper receivers + 2025 target volume
        adp_full = load_adp()
        recv_pool = adp_full[
            adp_full["position"].isin(["WR", "TE"])
            & adp_full["team"].notna()
            & adp_full["player_id"].notna()
        ].copy()
        recv_pool = recv_pool.rename(
            columns={"player_id": "gsis_id"})

        # 2025 targets per receiver, from attribution
        attr_for_battles = load_attribution()
        if not attr_for_battles.empty:
            tgts_25 = (
                attr_for_battles[attr_for_battles["season"] == 2025]
                .groupby("receiver_player_id", as_index=False)
                .agg(targets_2025=("targets", "sum"))
                .rename(columns={
                    "receiver_player_id": "gsis_id"})
            )
            recv_pool = recv_pool.merge(
                tgts_25, on="gsis_id", how="left")
            recv_pool["targets_2025"] = (
                recv_pool["targets_2025"].fillna(0))
        else:
            recv_pool["targets_2025"] = 0

        # Detect MEANINGFUL contested teams. Two qualifying signals:
        #   (A) STAR arrival — career_targets >= 300 (Pittman 722,
        #       Mike Evans 1304, DJ Moore 990 etc.). Filters out
        #       depth adds (Tutu Atwell 180, Tyler Johnson 145).
        #   (B) Tight WR1-vs-WR2 race: top-2 incumbents within 15
        #       targets AND both >= 80 in 2025 (real competition).
        trans_for_battles = load_transitions()
        teams_with_arrivals: set[str] = set()
        if not trans_for_battles.empty:
            arrivals_recv = trans_for_battles[
                (trans_for_battles["transition_type"] == "arrival")
                & (trans_for_battles["position"]
                   .isin(["WR", "TE"]))
                & (trans_for_battles["career_targets"]
                   .fillna(0) >= 300)
            ]
            teams_with_arrivals = set(
                arrivals_recv["team"].dropna().unique())

        close_share_teams: set[str] = set()
        if not recv_pool.empty:
            for tm, sub in recv_pool.groupby("team"):
                top2 = sub.nlargest(2, "targets_2025")
                if len(top2) < 2:
                    continue
                gap = (top2.iloc[0]["targets_2025"]
                       - top2.iloc[1]["targets_2025"])
                both_significant = (
                    top2.iloc[0]["targets_2025"] >= 80
                    and top2.iloc[1]["targets_2025"] >= 80)
                if gap <= 15 and both_significant:
                    close_share_teams.add(tm)

        contested_recv_teams = sorted(
            teams_with_arrivals | close_share_teams)
        st.caption(
            f"**{len(contested_recv_teams)} teams flagged as "
            "contested** (had a vet WR/TE arrival, or top-2 "
            "incumbents within 30 targets of each other in 2025)."
        )

        recv_show_all = st.checkbox(
            "Show every team (not just contested)", value=False,
            key="cb_recv_show_all")
        recv_teams_to_render = (
            sorted(recv_pool["team"].unique())
            if recv_show_all else contested_recv_teams)

        if "wr_te_picks" not in st.session_state:
            st.session_state["wr_te_picks"] = {}

        # Per-team rendering
        for team in recv_teams_to_render:
            sub = recv_pool[recv_pool["team"] == team].copy()
            wrs = sub[sub["position"] == "WR"].sort_values(
                "targets_2025", ascending=False)
            tes = sub[sub["position"] == "TE"].sort_values(
                "targets_2025", ascending=False)
            if wrs.empty and tes.empty:
                continue

            with st.expander(f"⚖️ {team}", expanded=False):
                cols = st.columns(4)
                slots = [
                    ("WR1", wrs, cols[0]),
                    ("WR2", wrs, cols[1]),
                    ("WR3", wrs, cols[2]),
                    ("TE1", tes, cols[3]),
                ]
                team_picks = st.session_state["wr_te_picks"].setdefault(
                    team, {})
                for slot, pool, col in slots:
                    if pool.empty:
                        with col:
                            st.caption(f"**{slot}**")
                            st.write("—")
                        continue
                    # Default = nth-ranked by targets (n = 1, 2, 3)
                    default_rank = (
                        1 if slot == "WR1" else
                        2 if slot == "WR2" else
                        3 if slot == "WR3" else 1)
                    if len(pool) >= default_rank:
                        default_pid = pool.iloc[
                            default_rank - 1]["gsis_id"]
                    else:
                        default_pid = pool.iloc[0]["gsis_id"]
                    cur_pid = team_picks.get(slot, default_pid)
                    # Build options sorted by 2025 targets
                    pool_ids = pool["gsis_id"].tolist()
                    pool_labels = [
                        f"{r['full_name']} ({int(r['targets_2025'])} "
                        f"tgts '25)"
                        for _, r in pool.iterrows()
                    ]
                    try:
                        default_idx = pool_ids.index(cur_pid)
                    except ValueError:
                        default_idx = 0
                    with col:
                        pick = col.selectbox(
                            f"**{slot}**",
                            options=range(len(pool_labels)),
                            format_func=(
                                lambda i, lab=pool_labels: lab[i]),
                            index=default_idx,
                            key=f"wrte_{team}_{slot}",
                        )
                        team_picks[slot] = pool_ids[pick]

        # Summary of receiver/TE picks
        n_overrides = 0
        for tm, picks in st.session_state.get(
                "wr_te_picks", {}).items():
            sub = recv_pool[recv_pool["team"] == tm]
            if sub.empty:
                continue
            wrs = sub[sub["position"] == "WR"].sort_values(
                "targets_2025", ascending=False)
            tes = sub[sub["position"] == "TE"].sort_values(
                "targets_2025", ascending=False)
            for slot, pid in picks.items():
                pool = wrs if slot.startswith("WR") else tes
                if pool.empty:
                    continue
                default_rank = (
                    1 if slot == "WR1" else
                    2 if slot == "WR2" else
                    3 if slot == "WR3" else 1)
                if len(pool) >= default_rank:
                    default_pid = pool.iloc[
                        default_rank - 1]["gsis_id"]
                    if pid != default_pid:
                        n_overrides += 1
        if n_overrides:
            st.success(
                f"You overrode {n_overrides} receiver/TE slot(s). "
                "These picks are saved in this session and will feed "
                "into the v2 propagation work.")
        else:
            st.caption(
                "No receiver/TE overrides yet — defaults reflect "
                "the leading 2025 target-getter still on each "
                "team's current roster.")


with tab6:
    # ══════════════════════════════════════════════════════════════════
    #  Section 6: 🏆 Per-route FP conversion leaderboard
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

            # ── YoY trend view ──────────────────────────────────────
            # Career numbers can hide single-bad-year drops (Bateman
            # 2.99 → 2.51 after one weak 2025). Show season-by-season
            # to separate "real decline" from "noisy small samples."
            st.markdown("---")
            st.markdown("### 📈 Year-over-year trend — same route, "
                          "same metric")
            st.caption(
                f"How each top-{config_name}-converter on "
                f"**{selected_route}** has performed season by season. "
                "Uses a per-season minimum-targets filter so single-"
                "game spikes don't dominate. **Use this to ask: is "
                "Bateman in actual decline, or did one bad 2025 drag "
                "his career number down?**"
            )

            top_n = st.slider(
                "Top N to chart", 3, 12, 6, key="yoy_top_n")
            min_per_season = st.slider(
                "Min targets per season (filter out small samples)",
                1, 15, 3, key="yoy_min_season_tgts")

            top_ids = sub.head(top_n)["receiver_player_id"].tolist()
            top_names = (sub.head(top_n)
                            .set_index("receiver_player_id")
                            ["player_display_name"].to_dict())

            yoy = (
                full_attr[
                    (full_attr["route"] == selected_route)
                    & (full_attr["receiver_player_id"].isin(top_ids))
                ]
                .groupby(
                    ["receiver_player_id", "player_display_name",
                     "season"], as_index=False)
                .agg(targets=("targets", "sum"),
                     fp=("row_fp", "sum"))
            )
            yoy = yoy[yoy["targets"] >= min_per_season]
            yoy["fp_per_target"] = (yoy["fp"]
                                       / yoy["targets"].clip(lower=1))

            if yoy.empty:
                st.info("No per-season data passes the filter. "
                        "Lower the min-targets-per-season slider.")
            else:
                # Plotly line chart
                fig = go.Figure()
                for pid in top_ids:
                    sub_y = yoy[yoy["receiver_player_id"] == pid
                                  ].sort_values("season")
                    if sub_y.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=sub_y["season"],
                        y=sub_y["fp_per_target"],
                        mode="lines+markers",
                        name=top_names.get(pid, pid),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "Season: %{x}<br>"
                            f"{config_name} FP/Tgt: %{{y:.2f}}<br>"
                            "Targets: %{customdata}<extra></extra>"),
                        customdata=sub_y["targets"],
                    ))
                fig.update_layout(
                    title=(f"{config_name} FP/target on "
                            f"{selected_route} — top {top_n} converters"),
                    xaxis_title="Season",
                    yaxis_title=f"{config_name} FP / target",
                    height=420,
                    margin=dict(t=50, b=40, l=50, r=20),
                    legend=dict(
                        orientation="h", yanchor="bottom",
                        y=1.05, xanchor="left", x=0),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Wide-format YoY table
                wide = (
                    yoy.pivot_table(
                        index=["player_display_name"],
                        columns="season",
                        values="fp_per_target",
                    ).round(2)
                )
                # Append career rate as right-most column for context
                career_rate = (
                    sub.set_index("player_display_name")
                       .loc[wide.index, "fp_per_target"]
                       .round(2)
                )
                wide["Career"] = career_rate
                # Sort by career rate descending
                wide = wide.sort_values("Career", ascending=False)
                wide = wide.reset_index().rename(
                    columns={"player_display_name": "Player"})
                # Sort season columns chronologically (Career last)
                season_cols = sorted(
                    [c for c in wide.columns
                     if c not in ("Player", "Career")])
                wide = wide[["Player"] + season_cols + ["Career"]]
                st.markdown(
                    f"**Per-season {config_name} FP/target — "
                    f"{selected_route} (NaN = below {min_per_season}-"
                    "target season minimum):**")
                st.dataframe(
                    wide, use_container_width=True,
                    hide_index=True,
                )

                # Auto-narrative: who is trending where?
                # Compare 2025 to player's prior 3-yr avg.
                trends = []
                for pid in top_ids:
                    s = yoy[yoy["receiver_player_id"] == pid
                              ].sort_values("season")
                    if len(s) < 2:
                        continue
                    last_yr = int(s["season"].max())
                    if last_yr < 2025:
                        continue
                    last_rate = float(
                        s[s["season"] == last_yr]
                            ["fp_per_target"].iloc[0])
                    prior_rates = s[s["season"] < last_yr]
                    if prior_rates.empty:
                        continue
                    prior_avg = float(
                        prior_rates["fp_per_target"].mean())
                    delta = last_rate - prior_avg
                    trends.append({
                        "Player": top_names.get(pid, pid),
                        f"2025 {config_name}/Tgt": round(last_rate, 2),
                        f"Prior-yrs avg {config_name}/Tgt":
                            round(prior_avg, 2),
                        "Δ vs prior": round(delta, 2),
                        "Trend": (
                            "🚀 Trending up" if delta >= 0.3 else
                            "⬇️ Trending down" if delta <= -0.3 else
                            "➡️ Steady"),
                    })
                if trends:
                    tdf = pd.DataFrame(trends).sort_values(
                        "Δ vs prior", ascending=False)
                    st.markdown(
                        "**2025 vs. prior career average — who's "
                        "trending where?**")
                    st.dataframe(
                        tdf, use_container_width=True,
                        hide_index=True,
                    )

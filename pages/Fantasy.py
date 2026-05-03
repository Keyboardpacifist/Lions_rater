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

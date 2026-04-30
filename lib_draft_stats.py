"""Per-prospect season + career stats display for the Draft page.

For each prospect we render a season-by-season table mixing the
counting stats fans expect (yards, TDs, attempts) with the advanced
metrics that exist for the position (EPA splits, usage rates,
per-game rates for defenders, line-unit stats for OL).

Keep the column count tight: 6-10 cols per position so the table
stays readable inside the expander.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st


def _z_to_pctl(z) -> int | None:
    if z is None or pd.isna(z):
        return None
    z_clip = max(-3.0, min(3.0, float(z)))
    pct = round(50.0 + 50.0 * math.erf(z_clip / math.sqrt(2)))
    return min(99, max(1, pct))


# Stats whose row should be followed by a national-percentile column.
# Key = raw column name (so per-position labels like "Yds" don't
# collide); value = the z-col on the per-season row.
_PCTL_BY_COL = {
    # QB
    "completion_pct":      "completion_pct_z",
    "pass_yards":          "pass_yards_z",
    "pass_tds":            "pass_tds_z",
    "yards_per_attempt":   "yards_per_attempt_z",
    "rush_yards_total":    "rush_yards_total_z",
    # Skill (WR / TE)
    "receptions":          "receptions_total_z",
    "rec_yards":           "rec_yards_total_z",
    "yards_per_rec":       "yards_per_rec_z",
    "rec_tds":             "rec_tds_total_z",
    # RB
    "rush_carries":        "carries_total_z",
    "rush_yards":          "rush_yards_total_z",
    "yards_per_carry":     "yards_per_carry_z",
    "rush_tds":            "rush_tds_total_z",
    # CFBD-advanced (skill positions)
    "epa_per_play_avg":    "epa_per_play_avg_z",
    "epa_per_pass_avg":    "epa_per_pass_avg_z",
    "epa_third_down_avg":  "epa_third_down_avg_z",
    "usage_pass":          "usage_pass_z",
    "usage_overall":       "usage_overall_z",
    "usage_third_down":    "usage_third_down_z",
    # Defense
    "tackles_total":       "tackles_per_game_z",
    "tackles_solo":        "solo_tackles_per_game_z",
    "tfl":                 "tfl_per_game_z",
    "sacks":               "sacks_per_game_z",
    "qb_hurries":          "qb_hurries_per_game_z",
    "passes_deflected":    "pd_per_game_z",
    "interceptions":       "int_per_game_z",
    "sacks_per_game":      "sacks_per_game_z",
    "pressure_rate":       "pressure_rate_z",
}

_DATA = Path(__file__).resolve().parent / "data"
_COLLEGE = _DATA / "college"

# (label, source_col, fmt, agg)
# agg ∈ {"sum", "mean", "max", None}. None = blank in career row.
_STATS_DISPLAY = {
    "QB": [
        ("Att",       "pass_att",            "{:.0f}",   "sum"),
        ("Cmp%",      "completion_pct",      "{:.1%}",   "mean"),
        ("Pass Yds",  "pass_yards",          "{:,.0f}",  "sum"),
        ("Pass TD",   "pass_tds",            "{:.0f}",   "sum"),
        ("INT",       "pass_ints",           "{:.0f}",   "sum"),
        ("Y/A",       "yards_per_attempt",   "{:.1f}",   "mean"),
        ("Rush Yds",  "rush_yards_total",    "{:,.0f}",  "sum"),
        ("Rush TD",   "rush_tds",            "{:.0f}",   "sum"),
        ("EPA/play",  "epa_per_play_avg",    "{:+.2f}",  "mean"),
        ("3rd-Dn EPA","epa_third_down_avg",  "{:+.2f}",  "mean"),
    ],
    "WR": [
        ("Rec",       "receptions",          "{:.0f}",   "sum"),
        ("Yds",       "rec_yards",           "{:,.0f}",  "sum"),
        ("Y/R",       "yards_per_rec",       "{:.1f}",   "mean"),
        ("TD",        "rec_tds",             "{:.0f}",   "sum"),
        ("Long",      "rec_long",            "{:.0f}",   "max"),
        ("EPA/play",  "epa_per_play_avg",    "{:+.2f}",  "mean"),
        ("EPA/tgt",   "epa_per_pass_avg",    "{:+.2f}",  "mean"),
        ("Pass-Use%", "usage_pass",          "{:.0%}",   "mean"),
        ("3rd-Dn Use%","usage_third_down",   "{:.0%}",   "mean"),
    ],
    "TE": [  # same set as WR
        ("Rec",       "receptions",          "{:.0f}",   "sum"),
        ("Yds",       "rec_yards",           "{:,.0f}",  "sum"),
        ("Y/R",       "yards_per_rec",       "{:.1f}",   "mean"),
        ("TD",        "rec_tds",             "{:.0f}",   "sum"),
        ("Long",      "rec_long",            "{:.0f}",   "max"),
        ("EPA/play",  "epa_per_play_avg",    "{:+.2f}",  "mean"),
        ("EPA/tgt",   "epa_per_pass_avg",    "{:+.2f}",  "mean"),
        ("Pass-Use%", "usage_pass",          "{:.0%}",   "mean"),
    ],
    "RB": [
        ("Att",       "rush_carries",        "{:.0f}",   "sum"),
        ("Yds",       "rush_yards",          "{:,.0f}",  "sum"),
        ("YPC",       "rush_ypc",            "{:.1f}",   "mean"),
        ("TD",        "rush_tds",            "{:.0f}",   "sum"),
        ("Long",      "rush_long",           "{:.0f}",   "max"),
        ("Rec",       "receptions",          "{:.0f}",   "sum"),
        ("Rec Yds",   "rec_yards",           "{:,.0f}",  "sum"),
        ("EPA/play",  "epa_per_play_avg",    "{:+.2f}",  "mean"),
        ("Use%",      "usage_overall",       "{:.0%}",   "mean"),
    ],
    # Defensive positions share the same parquet (college_def_all_seasons)
    # so use the same column spec — fans care about the same lines.
    "_DEF": [
        ("GP",        "games",               "{:.0f}",   "sum"),
        ("Tckl",      "tackles_total",       "{:.0f}",   "sum"),
        ("Solo",      "tackles_solo",        "{:.0f}",   "sum"),
        ("TFL",       "tfl",                 "{:.0f}",   "sum"),
        ("Sacks",     "sacks",               "{:.1f}",   "sum"),
        ("Hurries",   "qb_hurries",          "{:.0f}",   "sum"),
        ("PBU",       "passes_deflected",    "{:.0f}",   "sum"),
        ("INT",       "interceptions",       "{:.0f}",   "sum"),
        ("Sk/G",      "sacks_per_game",      "{:.2f}",   "mean"),
        # pressure_rate in CFBD parquet is actually
        # (sacks + hurries) / games — i.e. pressures-per-game, not a
        # percentage. Format as decimal, not %.
        ("Pressures/G", "pressure_rate",     "{:.2f}",   "mean"),
    ],
    "CB": [], "S": [], "LB": [], "DE": [], "DT": [],  # filled below
    "OL": [
        ("Line Yds",       "line_yards",        "{:.2f}",   "mean"),
        ("Stuff% Avoided", "stuff_rate_avoid",  "{:.1%}",   "mean"),
        ("Power Succ%",    "power_success",     "{:.1%}",   "mean"),
        ("Std-Dn Succ%",   "std_downs_success", "{:.1%}",   "mean"),
        ("Rush PPA",       "rushing_ppa",       "{:+.2f}",  "mean"),
        ("Pass PPA",       "passing_ppa",       "{:+.2f}",  "mean"),
    ],
}
# Defensive positions share one parquet — alias to the unified spec
for _pos in ("CB", "S", "LB", "DE", "DT"):
    _STATS_DISPLAY[_pos] = _STATS_DISPLAY["_DEF"]


_POS_FILES = {
    "QB": "college_qb_all_seasons.parquet",
    "WR": "college_wr_all_seasons.parquet",
    "TE": "college_te_all_seasons.parquet",
    "RB": "college_rb_all_seasons.parquet",
    "OL": "college_ol_roster.parquet",
}

# CFBD-advanced parquets carry the EPA + usage stats that the
# all_seasons primary parquets don't. Per-(player_id, season) join.
_CFBD_ADV_FILES = {
    "QB": "college_qb_cfbd_advanced.parquet",
    "WR": "college_wr_cfbd_advanced.parquet",
    "TE": "college_te_cfbd_advanced.parquet",
    "RB": "college_rb_cfbd_advanced.parquet",
}


@st.cache_data(show_spinner=False)
def get_prospect_seasons(player_id: str, position: str) -> pd.DataFrame:
    """Return all season-rows for the prospect across their career,
    sorted oldest → newest. Merges in the EPA/usage advanced parquet
    for skill positions. Empty if not found."""
    if not player_id:
        return pd.DataFrame()
    if position in _POS_FILES:
        path = _COLLEGE / _POS_FILES[position]
    elif position in ("CB", "S", "LB", "DE", "DT"):
        path = _COLLEGE / "college_def_all_seasons.parquet"
    else:
        return pd.DataFrame()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "player_id" not in df.columns:
        return pd.DataFrame()
    df["player_id"] = df["player_id"].astype(str)
    rows = df[df["player_id"] == str(player_id)].copy()

    # Merge the CFBD-advanced parquet for EPA + usage stats.
    if position in _CFBD_ADV_FILES:
        adv_path = _COLLEGE / _CFBD_ADV_FILES[position]
        if adv_path.exists():
            adv = pd.read_parquet(adv_path)
            adv["player_id"] = adv["player_id"].astype(str)
            adv_cols = [c for c in adv.columns
                         if c.startswith("epa_") or c.startswith("usage_")]
            if adv_cols:
                rows = rows.merge(
                    adv[["player_id", "season"] + adv_cols],
                    on=["player_id", "season"], how="left",
                    suffixes=("", "_adv"),
                )

    return rows.sort_values("season").reset_index(drop=True)


def _agg(values: pd.Series, kind: str):
    vals = values.dropna()
    if vals.empty:
        return None
    if kind == "sum":
        return float(vals.sum())
    if kind == "mean":
        return float(vals.mean())
    if kind == "max":
        return float(vals.max())
    return None


def render_prospect_stats(player_id: str, position: str) -> None:
    """Render the prospect's season-by-season + career stats table.
    Called from the Draft page expander."""
    spec = _STATS_DISPLAY.get(position, [])
    if not spec:
        return
    seasons = get_prospect_seasons(player_id, position)
    if seasons.empty:
        st.caption(
            "_No season-by-season data found in our parquets for this "
            "prospect (likely a true freshman with no 2025 stats yet)._"
        )
        return

    # Per-season rows
    rows = []
    for _, s in seasons.iterrows():
        row = {
            "Season": int(s["season"]) if pd.notna(s.get("season")) else "—",
            "School": s.get("team", "—") or "—",
        }
        for label, col, fmt, _agg_kind in spec:
            v = s.get(col)
            row[label] = fmt.format(v) if pd.notna(v) else "—"
            # Insert national-percentile column immediately after
            # any stat that has a partner z-col mapped.
            z_col = _PCTL_BY_COL.get(col)
            if z_col:
                pct = _z_to_pctl(s.get(z_col))
                row[f"{label} pctl"] = f"{pct}th" if pct else "—"
        rows.append(row)

    # Career row (counting = sum, rates = mean, longs = max).
    # Percentile columns left blank since we'd need a career-summed
    # cohort distribution to z-score against, which we don't have.
    career = {"Season": "Career", "School": "—"}
    for label, col, fmt, agg in spec:
        if agg is None or col not in seasons.columns:
            career[label] = "—"
        else:
            v = _agg(seasons[col], agg)
            career[label] = fmt.format(v) if v is not None else "—"
        if col in _PCTL_BY_COL:
            career[f"{label} pctl"] = "—"
    rows.append(career)

    if position == "OL":
        st.caption(
            "_Note: OL stats here are **team-unit metrics** "
            "(line-yards, stuff-rate avoided, etc.) — not individual "
            "player grades. Every OL on the team-season has the same "
            "values._"
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                  hide_index=True)

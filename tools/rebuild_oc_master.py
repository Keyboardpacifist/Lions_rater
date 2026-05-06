"""Rebuild master_ocs parquets from PBP using actual play-caller attribution.

Source of truth: data/scheme/curation/oc_team_seasons.csv (manually
maintained mapping of play-caller ↔ team-season).

This replaces the externally-generated parquets that incorrectly used
"OC of record" instead of "actual play-caller" — e.g. Pete Carmichael
got credit for Sean Payton's offenses in NO 2006-2021.

Outputs:
    data/master_ocs_with_z.parquet         — career-aggregated, z-scored
    data/master_ocs_2024_with_z.parquet    — single-season 2024
    data/master_ocs_2025_with_z.parquet    — single-season 2025  (NEW)

Schema matches the legacy file so downstream code (OC.py, GAS, etc.)
keeps working without changes.

Filters: only rows where calls_plays=TRUE are kept (admin OCs whose
HC actually called plays don't get credit for that team-season).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP_PATH = REPO / "data" / "game_pbp.parquet"
OC_TEAM_SEASONS_PATH = REPO / "data" / "scheme" / "curation" / "oc_team_seasons.csv"
TEAM_CONTEXT_PATH = REPO / "data" / "team_context.parquet"

OUT_CAREER = REPO / "data" / "master_ocs_with_z.parquet"
OUT_2024 = REPO / "data" / "master_ocs_2024_with_z.parquet"
OUT_2025 = REPO / "data" / "master_ocs_2025_with_z.parquet"
OUT_SEASONS = REPO / "data" / "master_ocs_seasons_with_z.parquet"

# Stats we z-score
Z_STATS = [
    "epa_per_play", "pass_epa_per_play", "rush_epa_per_play",
    "success_rate", "explosive_pass_rate", "explosive_rush_rate",
    "third_down_rate", "red_zone_td_rate", "win_pct",
]


def _zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.mean(); sd = s.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def _aggregate_team_season_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (posteam, season): plays, EPA, success, explosives, situational rates."""
    # Mark explosive plays — PBP stores passing_yards / rushing_yards separately
    pbp = pbp.copy()
    pbp["passing_yards"] = pbp["passing_yards"].fillna(0).astype(float)
    pbp["rushing_yards"] = pbp["rushing_yards"].fillna(0).astype(float)
    pbp["explosive_pass"] = ((pbp["play_type"] == "pass")
                              & (pbp["passing_yards"] >= 20)).astype(int)
    pbp["explosive_rush"] = ((pbp["play_type"] == "run")
                              & (pbp["rushing_yards"] >= 10)).astype(int)
    pbp["is_pass"] = (pbp["play_type"] == "pass").astype(int)
    pbp["is_rush"] = (pbp["play_type"] == "run").astype(int)
    pbp["third_down"] = (pbp["down"].fillna(0) == 3).astype(int)
    pbp["third_conv"] = (pbp["third_down"]
                          & (pbp["first_down"].fillna(0).astype(int) == 1)).astype(int)
    pbp["rz_play"] = (pbp["yardline_100"].fillna(99) <= 20).astype(int)
    pbp["rz_td"] = (pbp["rz_play"] & (pbp["touchdown"].fillna(0).astype(int) == 1)).astype(int)
    pbp["success"] = pbp.get("success", (pbp["epa"] > 0).astype(int))

    grp = pbp.groupby(["posteam", "season"])
    agg = pd.DataFrame({
        "total_plays": grp.size(),
        "epa_sum": grp["epa"].sum(),
        "epa_count": grp["epa"].count(),
        "success_sum": grp["success"].sum(),
        "success_count": grp["success"].count(),
        "pass_epa_sum": grp.apply(lambda g: g.loc[g["is_pass"]==1, "epa"].sum()),
        "pass_count": grp["is_pass"].sum(),
        "rush_epa_sum": grp.apply(lambda g: g.loc[g["is_rush"]==1, "epa"].sum()),
        "rush_count": grp["is_rush"].sum(),
        "explosive_pass": grp["explosive_pass"].sum(),
        "explosive_rush": grp["explosive_rush"].sum(),
        "third_down_plays": grp["third_down"].sum(),
        "third_down_convs": grp["third_conv"].sum(),
        "rz_plays": grp["rz_play"].sum(),
        "rz_tds": grp["rz_td"].sum(),
    }).reset_index()
    return agg


def _aggregate_game_outcomes(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (posteam, season): wins, losses, ties using game_id final scores."""
    games = pbp.dropna(subset=["game_id", "posteam"]).copy()
    games["pos_score"] = np.where(
        games["posteam"] == games["home_team"],
        games["home_score"], games["away_score"]
    )
    games["opp_score"] = np.where(
        games["posteam"] == games["home_team"],
        games["away_score"], games["home_score"]
    )
    games_unique = games.drop_duplicates(subset=["game_id", "posteam"])[
        ["game_id", "posteam", "season", "pos_score", "opp_score"]
    ].copy()
    games_unique["win"] = (games_unique["pos_score"] > games_unique["opp_score"]).astype(int)
    games_unique["loss"] = (games_unique["pos_score"] < games_unique["opp_score"]).astype(int)
    games_unique["tie"] = (games_unique["pos_score"] == games_unique["opp_score"]).astype(int)
    out = games_unique.groupby(["posteam", "season"]).agg(
        wins=("win", "sum"), losses=("loss", "sum"), ties=("tie", "sum"),
    ).reset_index()
    return out


def _attach_oc(team_season_stats: pd.DataFrame, team_outcomes: pd.DataFrame,
               oc_ts: pd.DataFrame) -> pd.DataFrame:
    """Inner-join team-season aggregates to play-caller mapping (calls_plays=TRUE only)."""
    callers = oc_ts[oc_ts["calls_plays"].astype(str).str.upper() == "TRUE"][
        ["oc_name", "team", "season"]
    ].copy()
    callers["season"] = callers["season"].astype(int)

    stats = team_season_stats.merge(
        callers, left_on=["posteam", "season"], right_on=["team", "season"],
        how="inner",
    )
    stats = stats.merge(
        team_outcomes, on=["posteam", "season"], how="left",
    )
    return stats


def _aggregate_career(per_oc_season: pd.DataFrame) -> pd.DataFrame:
    """Sum per-OC career totals from per-(oc, season) sums, then derive rates."""
    grp = per_oc_season.groupby("oc_name")
    out = pd.DataFrame({
        "coordinator": grp.size().index,
        "total_plays": grp["total_plays"].sum().values,
        "epa_sum": grp["epa_sum"].sum().values,
        "epa_count": grp["epa_count"].sum().values,
        "success_sum": grp["success_sum"].sum().values,
        "success_count": grp["success_count"].sum().values,
        "pass_epa_sum": grp["pass_epa_sum"].sum().values,
        "pass_count": grp["pass_count"].sum().values,
        "rush_epa_sum": grp["rush_epa_sum"].sum().values,
        "rush_count": grp["rush_count"].sum().values,
        "explosive_pass": grp["explosive_pass"].sum().values,
        "explosive_rush": grp["explosive_rush"].sum().values,
        "third_down_plays": grp["third_down_plays"].sum().values,
        "third_down_convs": grp["third_down_convs"].sum().values,
        "rz_plays": grp["rz_plays"].sum().values,
        "rz_tds": grp["rz_tds"].sum().values,
        "total_wins": grp["wins"].sum().values,
        "total_losses": grp["losses"].sum().values,
        "ties": grp["ties"].sum().values,
        "seasons": grp.size().values,
        "first_season": grp["season"].min().values,
        "last_season": grp["season"].max().values,
        "teams": grp.apply(lambda g: ", ".join(sorted(g["posteam"].unique()))).values,
    })
    return out


def _derive_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["epa_per_play"] = df["epa_sum"] / df["epa_count"].replace(0, np.nan)
    df["pass_epa_per_play"] = df["pass_epa_sum"] / df["pass_count"].replace(0, np.nan)
    df["rush_epa_per_play"] = df["rush_epa_sum"] / df["rush_count"].replace(0, np.nan)
    df["success_rate"] = df["success_sum"] / df["success_count"].replace(0, np.nan)
    df["explosive_pass_rate"] = df["explosive_pass"] / df["pass_count"].replace(0, np.nan)
    df["explosive_rush_rate"] = df["explosive_rush"] / df["rush_count"].replace(0, np.nan)
    df["third_down_rate"] = df["third_down_convs"] / df["third_down_plays"].replace(0, np.nan)
    df["red_zone_td_rate"] = df["rz_tds"] / df["rz_plays"].replace(0, np.nan)
    games_played = df["total_wins"] + df["total_losses"] + df["ties"] if "total_wins" in df.columns \
        else df["wins"] + df["losses"] + df["ties"]
    win_num = df["total_wins"] + 0.5 * df["ties"] if "total_wins" in df.columns \
        else df["wins"] + 0.5 * df["ties"]
    df["win_pct"] = win_num / games_played.replace(0, np.nan)
    return df


def _add_z_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for stat in Z_STATS:
        if stat in df.columns:
            df[f"{stat}_z"] = _zscore(df[stat])
    return df


def _attach_team_context_proxies(df: pd.DataFrame, oc_ts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate team_context.parquet per OC tenure for roster-quality proxies."""
    df = df.copy()
    if not TEAM_CONTEXT_PATH.exists():
        df["oc_team_rating_avg"] = np.nan
        df["oc_top5_apy_pct_avg"] = np.nan
        return df
    tc = pd.read_parquet(TEAM_CONTEXT_PATH)
    tc["season"] = tc["season"].astype(int)
    keep = [c for c in ["team_rating", "top5_apy_pct_of_cap"] if c in tc.columns]
    if not keep:
        df["oc_team_rating_avg"] = np.nan
        df["oc_top5_apy_pct_avg"] = np.nan
        return df
    callers = oc_ts[oc_ts["calls_plays"].astype(str).str.upper() == "TRUE"][
        ["oc_name", "team", "season"]
    ]
    callers["season"] = callers["season"].astype(int)
    joined = callers.merge(tc[["team", "season"] + keep], on=["team", "season"],
                           how="left")
    agg = joined.groupby("oc_name")[keep].mean().reset_index()
    agg = agg.rename(columns={"team_rating": "oc_team_rating_avg",
                                "top5_apy_pct_of_cap": "oc_top5_apy_pct_avg",
                                "oc_name": "coordinator"})
    df = df.merge(agg, on="coordinator", how="left")
    if "oc_team_rating_avg" not in df.columns:
        df["oc_team_rating_avg"] = np.nan
    if "oc_top5_apy_pct_avg" not in df.columns:
        df["oc_top5_apy_pct_avg"] = np.nan
    return df


def _add_roster_adj(df: pd.DataFrame, predictor_cols: list[str]) -> pd.DataFrame:
    """Add *_adj_z residual z-scores via OLS on roster proxies."""
    df = df.copy()
    haves = [c for c in predictor_cols if c in df.columns]
    if not haves:
        return df
    X = df[haves].to_numpy(dtype=float).copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.nanmean(col) if np.isfinite(np.nanmean(col)) else 0.0
        X[np.isnan(col), j] = m
    for stat in Z_STATS:
        if stat not in df.columns: continue
        y = df[stat].to_numpy(dtype=float)
        valid = ~np.isnan(y)
        if valid.sum() < 5:
            df[f"{stat}_adj_z"] = np.nan
            continue
        Xv = np.hstack([np.ones((valid.sum(), 1)), X[valid]])
        try:
            beta, *_ = np.linalg.lstsq(Xv, y[valid], rcond=None)
        except np.linalg.LinAlgError:
            df[f"{stat}_adj_z"] = np.nan
            continue
        pred = np.hstack([np.ones((len(y), 1)), X]) @ beta
        residual = y - pred
        df[f"{stat}_adj_z"] = _zscore(pd.Series(residual)).values
    return df


def main() -> None:
    print(f"→ loading PBP {PBP_PATH.relative_to(REPO)}")
    pbp = pd.read_parquet(PBP_PATH)
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    pbp = pbp.dropna(subset=["posteam", "season", "epa"])
    pbp["season"] = pbp["season"].astype(int)
    print(f"  scrimmage plays: {len(pbp):,}")

    print(f"→ loading {OC_TEAM_SEASONS_PATH.relative_to(REPO)}")
    oc_ts = pd.read_csv(OC_TEAM_SEASONS_PATH)
    oc_ts["season"] = oc_ts["season"].astype(int)
    callers_full = oc_ts[oc_ts["calls_plays"].astype(str).str.upper() == "TRUE"]
    print(f"  play-caller rows (calls_plays=TRUE): {len(callers_full)}")
    print(f"  unique OCs: {callers_full['oc_name'].nunique()}")
    print()

    print("→ aggregating team-season stats from PBP")
    ts_stats = _aggregate_team_season_stats(pbp)
    ts_outcomes = _aggregate_game_outcomes(pbp)
    print(f"  team-seasons: {len(ts_stats)}")

    # Attach OC names → per-OC-season
    print("→ attaching OC mapping")
    per_oc_season = _attach_oc(ts_stats, ts_outcomes, oc_ts)
    print(f"  per-OC-season rows: {len(per_oc_season)}")
    print()

    # ─────────────────────────────────────────────────────────
    # 1. CAREER aggregate
    # ─────────────────────────────────────────────────────────
    print("→ building CAREER aggregate")
    career_raw = _aggregate_career(per_oc_season)
    career = _derive_rates(career_raw)
    career = _add_z_columns(career)
    career = _attach_team_context_proxies(career, oc_ts)
    career = _add_roster_adj(career,
        predictor_cols=["oc_team_rating_avg", "oc_top5_apy_pct_avg"])
    OUT_CAREER.parent.mkdir(parents=True, exist_ok=True)
    career.to_parquet(OUT_CAREER, index=False)
    print(f"  ✓ wrote {OUT_CAREER.relative_to(REPO)}  rows={len(career)}")
    print()

    # ─────────────────────────────────────────────────────────
    # 1b. PER-(OC, season) file — for year-by-year radar/table
    # ─────────────────────────────────────────────────────────
    print("→ building per-season aggregate (one row per OC × season × team)")
    season_rows_all = per_oc_season.copy()
    season_rows_all = season_rows_all.drop(columns=["team"], errors="ignore")
    season_rows_all = season_rows_all.rename(
        columns={"posteam": "team", "oc_name": "coordinator"})
    season_rows_all["role"] = "OC"
    season_rows_all["side"] = "offense"
    season_rates = _derive_rates(season_rows_all)
    # Z-score within each season's pool (so each year stands alone)
    season_rates_z_parts = []
    for season_year, sub in season_rates.groupby("season"):
        sub = sub.copy()
        for stat in Z_STATS:
            if stat in sub.columns:
                sub[f"{stat}_z"] = _zscore(sub[stat])
        season_rates_z_parts.append(sub)
    season_rates_final = pd.concat(season_rates_z_parts, ignore_index=True)
    season_rates_final = season_rates_final.loc[:, ~season_rates_final.columns.duplicated()]
    season_rates_final.to_parquet(OUT_SEASONS, index=False)
    print(f"  ✓ wrote {OUT_SEASONS.relative_to(REPO)}  rows={len(season_rates_final)}")
    print()

    # ─────────────────────────────────────────────────────────
    # 2. SINGLE-SEASON files (2024 + 2025)
    # ─────────────────────────────────────────────────────────
    for target_season, out_path in [(2024, OUT_2024), (2025, OUT_2025)]:
        season_rows = per_oc_season[per_oc_season["season"] == target_season].copy()
        if season_rows.empty:
            print(f"→ {target_season}: no rows; skipping")
            continue
        print(f"→ building {target_season} single-season aggregate")
        # Drop merge-redundant columns; keep posteam as the canonical "team"
        season_rows = season_rows.drop(columns=["team"], errors="ignore")
        season_rows = season_rows.rename(columns={"posteam": "team",
                                                   "oc_name": "coordinator"})
        season_rows["role"] = "OC"
        season_rows["side"] = "offense"
        rates = _derive_rates(season_rows)
        rates = _add_z_columns(rates)
        rates = _attach_team_context_proxies(rates, oc_ts)
        rates = _add_roster_adj(rates,
            predictor_cols=["oc_team_rating_avg", "oc_top5_apy_pct_avg"])
        # Final dedup safety
        rates = rates.loc[:, ~rates.columns.duplicated()]
        rates.to_parquet(out_path, index=False)
        print(f"  ✓ wrote {out_path.relative_to(REPO)}  rows={len(rates)}")
        print()

    # ─────────────────────────────────────────────────────────
    # Spot checks
    # ─────────────────────────────────────────────────────────
    print("=== Sean Payton — career signals (now incl. NO 2016-2021 + DEN 2023-2024) ===")
    p = career[career["coordinator"] == "Sean Payton"]
    if not p.empty:
        cols = ["coordinator", "teams", "seasons", "first_season", "last_season",
                "total_plays", "win_pct", "epa_per_play", "epa_per_play_z",
                "epa_per_play_adj_z", "third_down_rate_z", "red_zone_td_rate_z"]
        cols = [c for c in cols if c in p.columns]
        for c in cols:
            v = p.iloc[0][c]
            print(f"  {c:25s} = {v}")
    print()

    print("=== Career top 12 by EPA z-score ===")
    top = career.nlargest(12, "epa_per_play_z")[
        ["coordinator", "teams", "seasons", "epa_per_play",
         "epa_per_play_z", "epa_per_play_adj_z"]
    ]
    print(top.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

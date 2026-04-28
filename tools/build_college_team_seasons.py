#!/usr/bin/env python3
"""
Per (school, season) team-strength rollup for college football. Built
by aggregating the existing position-level college parquets — top-N
players per position group, mean of their average z-score, then
z-scored across all team-seasons.

Output: data/college_team_seasons.parquet

Used by pages/CollegeTeam.py for the team profile + comp engine.

Run:
    python tools/build_college_team_seasons.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "college"
OUTPUT = REPO_ROOT / "data" / "college_team_seasons.parquet"

# (group_label, parquet, top_N_to_aggregate, sort_by)
_GROUPS = [
    ("qb",          "college_qb_all_seasons.parquet",  1,  "attempts"),
    ("wr_te_pass",  "college_wr_all_seasons.parquet",  3,  "receptions"),
    ("te",          "college_te_all_seasons.parquet",  1,  "receptions"),
    ("rb",          "college_rb_all_seasons.parquet",  2,  "carries"),
    ("ol",          "college_ol_roster.parquet",        5,  None),  # no obvious sort col
    ("def_all",     "college_def_all_seasons.parquet", 11, "tackles"),
]


def _aggregate_group(df: pd.DataFrame, top_n: int,
                       sort_by: str | None) -> pd.DataFrame:
    """For each (team, season), pick the top-N rows by `sort_by` and
    average each player's mean z-score. Returns one row per
    (team, season) with columns: n_players, avg_z."""
    z_cols = [c for c in df.columns if c.endswith("_z")]
    if not z_cols:
        return pd.DataFrame(columns=["team", "season", "n_players", "avg_z"])
    df = df.copy()
    df["_player_avg_z"] = df[z_cols].mean(axis=1, skipna=True)
    df = df.dropna(subset=["_player_avg_z", "team", "season"])
    if sort_by and sort_by in df.columns:
        df = df.sort_values(["team", "season", sort_by], ascending=[True, True, False])
    out = (
        df.groupby(["team", "season"])
        .head(top_n)
        .groupby(["team", "season"])
        .agg(n_players=("_player_avg_z", "size"),
             avg_z=("_player_avg_z", "mean"))
        .reset_index()
    )
    return out


def main() -> None:
    print("Aggregating college position parquets to team-season strengths…")
    teams: pd.DataFrame | None = None

    for group_label, fname, top_n, sort_by in _GROUPS:
        path = DATA / fname
        if not path.exists():
            print(f"  ⚠️  missing {fname} — skipping")
            continue
        df = pl.read_parquet(path).to_pandas()
        if "season" not in df.columns or "team" not in df.columns:
            print(f"  ⚠️  {fname}: no team/season columns — skipping")
            continue
        agg = _aggregate_group(df, top_n=top_n, sort_by=sort_by)
        agg = agg.rename(columns={
            "avg_z": f"{group_label}_strength",
            "n_players": f"{group_label}_n",
        })
        print(f"  {fname}: {len(agg)} (team, season) rows")
        if teams is None:
            teams = agg
        else:
            teams = teams.merge(agg, on=["team", "season"], how="outer")

    if teams is None or teams.empty:
        raise SystemExit("No team data aggregated.")

    # Z-score each *_strength column across all team-seasons so they're
    # comparable to each other.
    strength_cols = [c for c in teams.columns if c.endswith("_strength")]
    for col in strength_cols:
        s = teams[col].astype(float)
        mu, sigma = s.mean(), s.std(ddof=0)
        if sigma > 0:
            teams[f"{col}_z"] = (s - mu) / sigma
        else:
            teams[f"{col}_z"] = np.nan

    # Composite "overall team strength" — equal-weighted across the
    # six position-group z-scores (defense counted once even though
    # def_all encompasses front + back).
    z_strength_cols = [f"{c}_z" for c in strength_cols
                        if f"{c}_z" in teams.columns]
    if z_strength_cols:
        teams["overall_strength"] = teams[z_strength_cols].mean(axis=1, skipna=True)

    teams = teams.sort_values(["season", "team"]).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    teams.to_parquet(OUTPUT, index=False)
    print(f"\n✓ wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(teams):,} (team, season) rows × {teams.shape[1]} cols")
    print(f"\n  Sample — best 2024 team strengths:")
    yr = teams[teams["season"] == 2024].nlargest(8, "overall_strength")
    if not yr.empty:
        print(yr[["team", "season", "overall_strength"]].to_string(index=False))


if __name__ == "__main__":
    main()

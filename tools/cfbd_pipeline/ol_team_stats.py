"""
Enrich college_ol_roster.parquet with team-level OL quality metrics.

CFBD doesn't publish per-player OL grades in the free API, so we proxy
OL quality with team-level offensive metrics that depend heavily on the
line: run-block line yards, stuff-rate (inverted, lower=better OL),
short-yardage conversion, standard-downs success, rushing PPA, and
passing PPA (pressure tanks pass PPA, so it's a coarse pass-pro proxy).

Each metric is z-scored within the season's FBS pool, then assigned to
EVERY OL on that team-season. So Kadyn Proctor and his backup get the
same Alabama-2024 number — the UI must clearly label these as team-level.

Output: writes back to data/college/college_ol_roster.parquet, adding
6 raw metric columns + 6 z-scored versions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import client

OUTPUT_PATH = (Path(__file__).resolve().parent.parent.parent
               / "data" / "college" / "college_ol_roster.parquet")

# (raw_col_in_output, path_in_cfbd_response, invert_for_z)
# invert=True means "lower raw value = better OL" so we negate the z-score
# to keep the convention "positive z = better".
METRICS = [
    ("line_yards",         ("offense", "lineYards"),                False),
    ("stuff_rate_avoid",   ("offense", "stuffRate"),                True),   # invert
    ("power_success",      ("offense", "powerSuccess"),             False),
    ("std_downs_success",  ("offense", "standardDowns", "successRate"), False),
    ("rushing_ppa",        ("offense", "rushingPlays", "ppa"),      False),
    ("passing_ppa",        ("offense", "passingPlays", "ppa"),      False),
]


def _nested_get(d: dict, path: tuple):
    cur = d
    for k in path:
        if not isinstance(cur, dict): return None
        cur = cur.get(k)
        if cur is None: return None
    return cur


def pull_team_metrics(seasons: list[int], verbose: bool = True) -> pd.DataFrame:
    rows = []
    for year in seasons:
        if verbose: print(f"[{year}] team-advanced...")
        teams = client.team_advanced(year, verbose=verbose)
        for t in teams:
            team_name = t.get("team")
            conf = t.get("conference")
            if not team_name: continue
            entry = {"team": team_name, "season": int(year), "conference": conf}
            for raw_col, path, _invert in METRICS:
                v = _nested_get(t, path)
                entry[raw_col] = float(v) if v is not None else None
            rows.append(entry)
    return pd.DataFrame(rows)


def add_z_scores(team_df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each metric within (season). Invert for stuff-rate-avoid."""
    out = team_df.copy()
    for raw_col, _path, invert in METRICS:
        z_col = f"{raw_col}_z"
        z = pd.Series(np.nan, index=out.index)
        for season, grp in out.groupby("season"):
            vals = grp[raw_col].dropna()
            if len(vals) < 5: continue
            mean, std = vals.mean(), vals.std()
            if not std or std == 0: continue
            z.loc[grp.index] = (grp[raw_col] - mean) / std
        if invert:
            z = -z
        out[z_col] = z
    return out


def run(seasons: list[int] | None = None, verbose: bool = True) -> None:
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"{OUTPUT_PATH} missing — run tools/cfbd_pipeline/ol_roster.py first."
        )
    seasons = seasons or list(range(2014, 2026))

    team_df = pull_team_metrics(seasons, verbose=verbose)
    if team_df.empty:
        print("No team-advanced data pulled — aborting.")
        return
    team_df = add_z_scores(team_df)
    if verbose:
        print(f"\nTeam-advanced rows: {len(team_df)} across {team_df['season'].nunique()} seasons")
        for raw_col, _path, _ in METRICS:
            n_pop = team_df[raw_col].notna().sum()
            n_z = team_df[f"{raw_col}_z"].notna().sum()
            print(f"  {raw_col}: {n_pop} populated, {n_z} z-scored")

    # Merge by (team, season). Drop existing metric columns first so re-runs replace.
    ol = pd.read_parquet(OUTPUT_PATH)
    metric_cols = []
    for raw_col, _, _ in METRICS:
        metric_cols.extend([raw_col, f"{raw_col}_z"])
    metric_cols.append("conference")
    for c in metric_cols:
        if c in ol.columns:
            ol = ol.drop(columns=c)
    keep = ["team", "season", "conference"] + [c for c in metric_cols if c not in ("conference",)]
    keep = [c for c in keep if c in team_df.columns]
    enriched = ol.merge(team_df[keep], on=["team", "season"], how="left")
    if verbose:
        match_rate = enriched["line_yards"].notna().sum() / len(enriched)
        print(f"\nOL roster rows: {len(ol)}")
        print(f"  Match rate: {match_rate:.1%} ({enriched['line_yards'].notna().sum()} OLs got team stats)")
    enriched.to_parquet(OUTPUT_PATH, index=False)
    if verbose:
        print(f"\nWrote {OUTPUT_PATH.name}: {len(enriched)} rows, {len(enriched.columns)} cols")


if __name__ == "__main__":
    run(verbose=True)

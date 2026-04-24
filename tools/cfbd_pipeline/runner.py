"""
CFBD enrichment pipeline.

For each season:
  - Pull /stats/player/season (counting stats by category — long format)
  - Pull /ppa/players/season (per-play PPA = CFBD's EPA)
  - Pull /player/usage (target share, pass/rush usage)
  - Pivot stats to wide, join everything by player_id
  - Write data/college/cfbd_advanced_<season>.parquet

Then merge_and_zscore() combines all seasons into a single per-position
parquet with z-scores added.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import client

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "college"

# Counting stats we want per player. Long format from CFBD has
# (category, statType, stat) — we pivot to wide.
WANTED_STATS = {
    "passing": [
        ("ATT", "pass_att"),
        ("COMPLETIONS", "pass_completions"),
        ("YDS", "pass_yards"),
        ("TD", "pass_tds"),
        ("INT", "pass_ints"),
        ("YPA", "pass_ypa"),
        ("PCT", "pass_completion_pct"),
    ],
    "rushing": [
        ("CAR", "rush_carries"),
        ("YDS", "rush_yards"),
        ("TD", "rush_tds"),
        ("YPC", "rush_ypc"),
        ("LONG", "rush_long"),
    ],
    "receiving": [
        ("REC", "receptions"),
        ("YDS", "rec_yards"),
        ("TD", "rec_tds"),
        ("YPR", "rec_ypr"),
        ("LONG", "rec_long"),
        # Note: targets not exposed by CFBD aggregates — would require
        # parsing /plays text. Defer.
    ],
    "defensive": [
        ("TOT", "tackles_total"),
        ("SOLO", "tackles_solo"),
        ("SACKS", "sacks"),
        ("TFL", "tfl"),
        ("PD", "passes_defended"),
        ("QB HUR", "qb_hurries"),
    ],
    # CFBD puts defensive INTs in their own category, not under "defensive"
    "interceptions": [
        ("INT", "interceptions"),
    ],
}


def pull_season(year: int, verbose: bool = True) -> pd.DataFrame:
    """Pull all CFBD endpoints for one season and return a wide DataFrame
    keyed by player_id."""
    if verbose:
        print(f"\n=== CFBD season {year} ===")

    # 1. Counting stats (long → wide)
    stats = client.stats_player_season(year, verbose=verbose)
    if not stats:
        if verbose:
            print(f"  No stats returned for {year}")
        return pd.DataFrame()

    stats_df = pd.DataFrame(stats)
    # Long-format pivot: keep only the (category, statType) we care about
    rows = {}
    for r in stats:
        pid = str(r.get("playerId", ""))
        if not pid:
            continue
        if pid not in rows:
            rows[pid] = {
                "player_id": pid,
                "player": r.get("player"),
                "team": r.get("team"),
                "conference": r.get("conference"),
                "position": r.get("position"),
                "season": year,
            }
        cat = r.get("category", "")
        stype = r.get("statType", "")
        for keep_cat, mapping in WANTED_STATS.items():
            if cat == keep_cat:
                for src_type, out_col in mapping:
                    if stype == src_type:
                        try:
                            rows[pid][out_col] = float(r.get("stat", 0))
                        except (ValueError, TypeError):
                            pass
    counting_df = pd.DataFrame(list(rows.values()))
    if verbose:
        print(f"  Counting: {len(counting_df)} players")

    # 2. PPA (EPA) per player — wide nested dicts
    ppa = client.ppa_players_season(year, verbose=verbose)
    ppa_rows = []
    for r in ppa:
        avg = r.get("averagePPA") or {}
        tot = r.get("totalPPA") or {}
        ppa_rows.append({
            "player_id": str(r.get("id", "")),
            "epa_per_play_avg": avg.get("all"),
            "epa_per_pass_avg": avg.get("pass"),
            "epa_per_rush_avg": avg.get("rush"),
            "epa_first_down_avg": avg.get("firstDown"),
            "epa_third_down_avg": avg.get("thirdDown"),
            "epa_passing_downs_avg": avg.get("passingDowns"),
            "epa_total": tot.get("all"),
            "epa_total_pass": tot.get("pass"),
            "epa_total_rush": tot.get("rush"),
        })
    ppa_df = pd.DataFrame(ppa_rows)
    if verbose:
        print(f"  PPA: {len(ppa_df)} players")

    # 3. Usage per player — wide nested dict
    usage = client.player_usage(year, verbose=verbose)
    usage_rows = []
    for r in usage:
        u = r.get("usage") or {}
        usage_rows.append({
            "player_id": str(r.get("id", "")),
            "usage_overall": u.get("overall"),
            "usage_pass": u.get("pass"),
            "usage_rush": u.get("rush"),
            "usage_first_down": u.get("firstDown"),
            "usage_third_down": u.get("thirdDown"),
            "usage_standard_downs": u.get("standardDowns"),
            "usage_passing_downs": u.get("passingDowns"),
        })
    usage_df = pd.DataFrame(usage_rows)
    if verbose:
        print(f"  Usage: {len(usage_df)} players")

    # 4. Join
    out = counting_df
    if not ppa_df.empty:
        out = out.merge(ppa_df, on="player_id", how="left")
    if not usage_df.empty:
        out = out.merge(usage_df, on="player_id", how="left")

    if verbose:
        print(f"  Joined: {len(out)} players, {len(out.columns)} cols")

    return out


def pull_seasons(years: list[int], verbose: bool = True) -> pd.DataFrame:
    """Pull a range of seasons and concat."""
    parts = []
    for y in years:
        df = pull_season(y, verbose=verbose)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ── Z-scoring (within position group, within season) ──────────────────


def zscore_within(df: pd.DataFrame, cols: list[str], group_cols: list[str],
                   suffix: str = "_z") -> pd.DataFrame:
    """Z-score `cols` within each `group_cols` partition (e.g., position+season)."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col + suffix] = np.nan
            continue
        # Group-wise z-score
        grouped = df.groupby(group_cols)[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        df[col + suffix] = (df[col] - mean) / std.replace(0, np.nan)
    return df


# Position groupings for college (offensive position broad buckets).
POS_GROUP = {
    "QB": "QB",
    "WR": "WR",
    "TE": "TE",
    "RB": "RB",
    "FB": "RB",
}


def write_position_files(df: pd.DataFrame, verbose: bool = True) -> None:
    """Filter + z-score per position, write enhanced parquets."""
    if df.empty:
        return
    df["pos_group_canonical"] = df["position"].map(POS_GROUP).fillna(df["position"])

    for pos in ["QB", "WR", "TE", "RB"]:
        sub = df[df["pos_group_canonical"] == pos].copy()
        if sub.empty:
            continue

        # Z-score the new advanced cols within season (already a single position group).
        z_cols = [
            "epa_per_play_avg", "epa_per_pass_avg", "epa_per_rush_avg",
            "epa_third_down_avg", "epa_passing_downs_avg",
            "usage_overall", "usage_pass", "usage_rush",
            "usage_third_down", "usage_passing_downs",
        ]
        existing_z = [c for c in z_cols if c in sub.columns]
        sub = zscore_within(sub, existing_z, group_cols=["season"])

        out_path = OUTPUT_DIR / f"college_{pos.lower()}_cfbd_advanced.parquet"
        sub.to_parquet(out_path, index=False)
        if verbose:
            print(f"  Wrote {out_path.name}: {len(sub)} rows, {len(sub.columns)} cols")

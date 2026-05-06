"""Build OC game-script splits dataset (Feature B).

Same OC, three game states (leading by 7+ / tied/one-score / trailing
by 7+). Reveals identity: presses gas when ahead vs folds when behind.

Output: data/oc_gamescript.parquet
Schema (long): oc_name, gamescript, n_plays, epa_per_play,
    success_rate, pass_rate, explosive_pass_rate, explosive_rush_rate,
    no_huddle_rate + per-metric *_z (within-bucket league z).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
OC_TEAM_SEASONS = REPO / "data" / "scheme" / "curation" / "oc_team_seasons.csv"
OUT = REPO / "data" / "oc_gamescript.parquet"

METRIC_COLS = ["epa_per_play", "success_rate", "pass_rate",
               "explosive_pass_rate", "explosive_rush_rate",
               "no_huddle_rate"]


def _gamescript_bucket(diff: float) -> str:
    if pd.isna(diff): return None
    if diff >= 8:  return "leading"
    if diff <= -8: return "trailing"
    return "tied"


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    valid = ~np.isnan(x)
    if valid.sum() < 3:
        return np.full_like(x, np.nan)
    mu = np.nanmean(x); sd = np.nanstd(x[valid], ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(x)
    return (x - mu) / sd


def main() -> None:
    print(f"→ loading PBP {PBP.relative_to(REPO)}")
    pbp = pd.read_parquet(PBP)
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    pbp = pbp.dropna(subset=["posteam", "season", "epa", "score_differential"])
    pbp["season"] = pbp["season"].astype(int)
    pbp["gamescript"] = pbp["score_differential"].apply(_gamescript_bucket)
    pbp = pbp.dropna(subset=["gamescript"])
    print(f"  scrimmage plays w/ gamescript: {len(pbp):,}")

    # Tag with OC mapping
    oc_ts = pd.read_csv(OC_TEAM_SEASONS)
    oc_ts = oc_ts[oc_ts["calls_plays"].astype(str).str.upper() == "TRUE"].copy()
    oc_ts["season"] = oc_ts["season"].astype(int)
    pbp = pbp.merge(
        oc_ts[["oc_name", "team", "season"]],
        left_on=["posteam", "season"], right_on=["team", "season"], how="inner",
    )
    print(f"  plays w/ OC: {len(pbp):,}")

    # Derive per-play features
    pbp["is_pass"] = (pbp["play_type"] == "pass").astype(int)
    pbp["is_rush"] = (pbp["play_type"] == "run").astype(int)
    pbp["explosive_pass"] = (
        (pbp["is_pass"] == 1)
        & (pbp["passing_yards"].fillna(0).astype(float) >= 20)
    ).astype(int)
    pbp["explosive_rush"] = (
        (pbp["is_rush"] == 1)
        & (pbp["rushing_yards"].fillna(0).astype(float) >= 10)
    ).astype(int)
    pbp["is_no_huddle"] = pbp["no_huddle"].fillna(0).astype(int)
    pbp["success"] = pbp.get("success", (pbp["epa"] > 0).astype(int))

    rows = []
    grp = pbp.groupby(["oc_name", "gamescript"])
    for (oc, bucket), g in grp:
        n_pass = int(g["is_pass"].sum())
        n_rush = int(g["is_rush"].sum())
        n = len(g)
        rows.append({
            "oc_name": oc,
            "gamescript": bucket,
            "n_plays": n,
            "n_pass": n_pass,
            "n_rush": n_rush,
            "epa_per_play": float(g["epa"].mean()),
            "success_rate": float(g["success"].mean()),
            "pass_rate": (n_pass / n) if n > 0 else np.nan,
            "explosive_pass_rate": (g["explosive_pass"].sum() / n_pass) if n_pass > 0 else np.nan,
            "explosive_rush_rate": (g["explosive_rush"].sum() / n_rush) if n_rush > 0 else np.nan,
            "no_huddle_rate": float(g["is_no_huddle"].mean()),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  ⚠ no rows produced; aborting"); return

    # Z-score within (gamescript) — comparing OCs in the same game state
    for metric in METRIC_COLS:
        df[f"{metric}_z"] = (
            df.groupby("gamescript")[metric].transform(lambda s: _zscore(s.values))
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n✓ wrote {OUT.relative_to(REPO)}  rows={len(df)}")
    print()

    # Spot checks
    print("=== Game-script identity for top OCs ===")
    show = ["oc_name", "gamescript", "n_plays", "epa_per_play",
            "epa_per_play_z", "pass_rate", "no_huddle_rate"]
    for oc in ["Andy Reid", "Sean McVay", "Ben Johnson", "Sean Payton", "Matt LaFleur"]:
        sub = df[df["oc_name"] == oc].sort_values(
            "gamescript", key=lambda s: s.map({"leading": 0, "tied": 1, "trailing": 2}))
        if not sub.empty:
            print(f"\n--- {oc} ---")
            print(sub[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

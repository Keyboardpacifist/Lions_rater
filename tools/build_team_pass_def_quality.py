#!/usr/bin/env python3
"""
Per (team, season) pass defense quality lookup, computed from the
per-dropback feed. Used by the QB panel's "elite vs. weak competition"
bucket — splits a QB's plays by which quartile of pass D they faced.

Output: data/team_pass_def_quality.parquet

Run:
    python tools/build_team_pass_def_quality.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "data" / "qb_dropbacks.parquet"
OUTPUT = REPO_ROOT / "data" / "team_pass_def_quality.parquet"


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(
            f"Missing {SOURCE.relative_to(REPO_ROOT)} — "
            f"run `python tools/build_qb_dropbacks.py` first."
        )
    db = pd.read_parquet(SOURCE)

    # Regular-season only — playoff samples are too small per defense
    reg = db[db["season_type"] == "REG"].copy()

    # EPA allowed per pass play (lower = better defense). Sample size
    # filter — drop team-seasons with <100 dropbacks faced.
    grouped = (
        reg.groupby(["defteam", "season"])
        .agg(
            dropbacks_faced=("epa", "size"),
            pass_epa_allowed_per_play=("epa", "mean"),
            completion_pct_allowed=("complete_pass", "mean"),
            pressure_rate_allowed=("qb_hit", "mean"),
            sack_rate_forced=("sack", "mean"),
        )
        .reset_index()
        .rename(columns={"defteam": "team"})
    )
    grouped = grouped[grouped["dropbacks_faced"] >= 100].reset_index(drop=True)

    # Per-season ranking — lower EPA allowed = better defense
    grouped["epa_rank"] = (
        grouped.groupby("season")["pass_epa_allowed_per_play"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    grouped["teams_in_season"] = (
        grouped.groupby("season")["team"].transform("count").astype(int)
    )
    grouped["epa_pct_in_season"] = (
        (grouped["epa_rank"] - 1) / (grouped["teams_in_season"] - 1)
    )

    # Quartile bucket: 1 = elite (best 25% of pass Ds), 4 = bad (worst)
    def _bucket(p: float) -> int:
        if p < 0.25: return 1
        if p < 0.50: return 2
        if p < 0.75: return 3
        return 4
    grouped["quality_quartile"] = grouped["epa_pct_in_season"].apply(_bucket)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(OUTPUT, index=False)
    print(f"✓ wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(grouped):,} (team, season) rows")
    print(f"\n  Quartile distribution:")
    print(grouped.groupby("quality_quartile").size().to_string())
    print(f"\n  Sample — best pass Ds 2024:")
    yr = grouped[grouped["season"] == 2024].nsmallest(5, "pass_epa_allowed_per_play")
    print(yr[["team", "pass_epa_allowed_per_play", "epa_rank",
              "quality_quartile"]].to_string(index=False))


if __name__ == "__main__":
    main()

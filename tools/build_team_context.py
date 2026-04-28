#!/usr/bin/env python3
"""
Per (team, season) team-context rollup — roster age, cap concentration,
window indicators, and year-over-year trajectory. Joined to the existing
team_seasons.parquet to power the contention-state classifier and the
"era / cycle" panel on the Team page.

Output: data/team_context.parquet

Run:
    python tools/build_team_context.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
OUTPUT = DATA / "team_context.parquet"

SEASONS = list(range(2016, 2026))
CURRENT_YEAR = 2026


def main() -> None:
    import nflreadpy as nfl

    print("Pulling rosters for age data…")
    rosters_frames = []
    for s in SEASONS:
        try:
            r = nfl.load_rosters([s]).to_pandas()
            r["season"] = s
            rosters_frames.append(r)
        except Exception as e:
            print(f"  {s}: {e}")
    rosters = pd.concat(rosters_frames, ignore_index=True)
    print(f"  {len(rosters):,} roster rows")

    # Compute age at the season's kickoff (Sep 1)
    rosters["birth_date"] = pd.to_datetime(rosters.get("birth_date"),
                                              errors="coerce")
    rosters["age"] = (
        (pd.to_datetime(rosters["season"].astype(str) + "-09-01")
          - rosters["birth_date"]).dt.days / 365.25
    )

    # Per (team, season) age aggregates — weighted by snap counts not
    # available here, so use simple mean of starters (status='ACT' as
    # proxy for active roster).
    if "status" in rosters.columns:
        active = rosters[rosters["status"].astype(str).str.startswith("ACT")]
    else:
        active = rosters
    age_agg = (
        active.dropna(subset=["age"])
        .groupby(["team", "season"])
        .agg(
            roster_size=("age", "size"),
            avg_age=("age", "mean"),
            median_age=("age", "median"),
        )
        .reset_index()
    )
    print(f"  {len(age_agg):,} (team, season) age aggregates")

    # ── Contracts (cap concentration + window indicator) ──
    print("Pulling contracts…")
    contracts = nfl.load_contracts().to_pandas()
    # Active contracts only — filter where contract is still in force
    # at the start of the season we're evaluating.
    contracts["start_year"] = contracts["year_signed"]
    contracts["end_year"] = (
        contracts["year_signed"] + contracts["years"].fillna(0) - 1
    )
    print(f"  {len(contracts):,} contract rows")

    # For each (team, season), find the contracts that were active that
    # year, and compute concentration metrics.
    rows = []
    for s in SEASONS:
        active_c = contracts[(contracts["start_year"] <= s)
                              & (contracts["end_year"] >= s)]
        for team, sub in active_c.groupby("team"):
            top5 = sub.nlargest(5, "apy")
            rows.append({
                "team_name": team,
                "season": s,
                "n_active_contracts": len(sub),
                "total_apy_top10": float(sub.nlargest(10, "apy")["apy"].sum()),
                "top5_apy": float(top5["apy"].sum()),
                "top5_apy_pct_of_cap": float(top5["apy_cap_pct"].sum()),
                "top1_apy_pct_of_cap": float(sub.nlargest(1, "apy_cap_pct")["apy_cap_pct"].sum()),
                "avg_years_remaining_top5": float(
                    (top5["end_year"] - s + 1).mean()
                ),
            })
    cap_agg = pd.DataFrame(rows)

    # Map team-name (Lions) → abbr (DET) for joining. nflverse contracts
    # uses team nicknames; team_seasons uses abbreviations.
    TEAM_NAME_TO_ABBR = {
        "Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF",
        "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE",
        "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
        "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC",
        "Chargers": "LAC", "Rams": "LA", "Raiders": "LV", "Dolphins": "MIA",
        "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
        "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "Seahawks": "SEA",
        "49ers": "SF", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS",
    }
    cap_agg["team"] = cap_agg["team_name"].map(TEAM_NAME_TO_ABBR)
    cap_agg = cap_agg.dropna(subset=["team"])
    cap_agg = cap_agg.drop(columns=["team_name"])

    # ── Combine age + cap ──
    out = age_agg.merge(cap_agg, on=["team", "season"], how="outer")
    out = out.sort_values(["season", "team"]).reset_index(drop=True)

    # Year-over-year trajectory needs the team rating from team_seasons
    ts_path = DATA / "team_seasons.parquet"
    if ts_path.exists():
        ts = pd.read_parquet(ts_path)[["team", "season",
                                         "off_epa_per_play",
                                         "def_epa_per_play"]]
        # Composite "team rating" = off_epa - def_epa (def_epa is what
        # they allow, lower is better, so subtracting makes it
        # "offense better than defense allows" net rating)
        ts["team_rating"] = ts["off_epa_per_play"] - ts["def_epa_per_play"]
        ts_prev = ts.copy()
        ts_prev["next_season"] = ts_prev["season"] + 1
        ts_prev = ts_prev.rename(columns={"team_rating": "prev_team_rating"})[
            ["team", "next_season", "prev_team_rating"]]
        out = out.merge(
            ts[["team", "season", "team_rating"]],
            on=["team", "season"], how="left",
        )
        out = out.merge(
            ts_prev.rename(columns={"next_season": "season"}),
            on=["team", "season"], how="left",
        )
        out["trajectory"] = out["team_rating"] - out["prev_team_rating"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT, index=False)
    print(f"\n✓ wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(out):,} (team, season) rows × {out.shape[1]} cols")
    print()
    print("  Sample — 2024 Lions:")
    lions = out[(out["team"] == "DET") & (out["season"] == 2024)]
    if not lions.empty:
        print(lions.iloc[0].to_string())


if __name__ == "__main__":
    main()

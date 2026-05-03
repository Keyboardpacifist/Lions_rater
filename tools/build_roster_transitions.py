"""Roster transition ledger — who left each team, who joined.

Output: data/scheme/roster_transitions.parquet

Compares last NFL season's actual receivers (from PBP attribution)
against current rosters (from Sleeper) to surface:
  - DEPARTURES: receivers who played for team T last year but
    aren't on T's current roster (retired, traded out, FA out).
  - VET ARRIVALS: NFL veterans on T's current roster who weren't
    on T's last-year cohort (FA in / trade in / depth promotion).
  - ROOKIE ARRIVALS: years_exp == 0 AND no NFL pbp footprint
    last year. Profile is "TBD" — flag for downstream features
    (combine archetype, college route data ingest if/when added).

Schema
------
    team, prior_season, transition_type, player_id, sleeper_id,
    player_display_name, position, is_rookie, prior_team,
    prior_season_targets, career_targets, years_exp

  transition_type: "departure" | "arrival"
  is_rookie: True if years_exp ∈ {0, None} AND player has no PBP
             record in PRIOR_SEASON anywhere in the league
  prior_team: for departures, the team they left; for vet
              arrivals, the team where they had the most targets
              last year (NaN if no prior team)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ATTRIBUTION = REPO / "data" / "scheme" / "team_route_attribution.parquet"
SLEEPER_ADP = REPO / "data" / "fantasy" / "sleeper_adp.parquet"
OUT_DIR = REPO / "data" / "scheme"
OUT = OUT_DIR / "roster_transitions.parquet"

PRIOR_SEASON = 2025
MIN_PRIOR_TARGETS = 10
RECEIVING_POSITIONS = {"WR", "TE", "RB"}


def main() -> None:
    print(f"→ loading attribution + Sleeper rosters...")
    attr = pd.read_parquet(ATTRIBUTION)
    sleeper = pd.read_parquet(SLEEPER_ADP)

    # ── ALL prior-season NFL receivers (any role, any team) ────────
    # Used to distinguish rookies (no last-year footprint) from
    # vet depth adds (had a small role somewhere last year).
    all_last_year_ids = set(
        attr[attr["season"] == PRIOR_SEASON]["receiver_player_id"]
            .unique()
    )
    print(f"  all NFL receivers in {PRIOR_SEASON}: "
          f"{len(all_last_year_ids):,}")

    # ── Last year's significant team cohort ────────────────────────
    last_year = attr[attr["season"] == PRIOR_SEASON].copy()
    last_year_player = (
        last_year.groupby(
            ["team", "receiver_player_id", "player_display_name",
             "position"], as_index=False
        )
        .agg(prior_season_targets=("targets", "sum"))
    )
    last_year_player = last_year_player[
        last_year_player["prior_season_targets"] >= MIN_PRIOR_TARGETS
    ].rename(columns={"receiver_player_id": "player_id"})
    print(f"  cohort (≥{MIN_PRIOR_TARGETS} targets): "
          f"{len(last_year_player):,} player-team pairs")

    # ── Current Sleeper rosters at receiving positions ─────────────
    # NOTE: don't require player_id (gsis_id) — rookies have None
    # there because our gsis crosswalk only includes prior-NFL players.
    # We need rookies in this set to detect "arrival without prior NFL"
    # → is_rookie=True downstream.
    sl = sleeper[
        sleeper["position"].isin(RECEIVING_POSITIONS)
        & sleeper["team"].notna()
    ].copy()
    print(f"  current Sleeper rosters: {len(sl):,} active receivers "
          f"(includes {sl['player_id'].isna().sum():,} unmapped rookies)")

    # ── DEPARTURES ─────────────────────────────────────────────────
    sl_lookup = sl[["player_id", "team"]].rename(
        columns={"team": "current_team"})
    last_year_player = last_year_player.merge(
        sl_lookup, on="player_id", how="left")
    departures = last_year_player[
        last_year_player["current_team"] != last_year_player["team"]
    ].copy()
    departures["transition_type"] = "departure"
    departures["prior_team"] = departures["team"]
    departures["is_rookie"] = False
    print(f"  departures: {len(departures):,}")

    # ── ARRIVALS — split into vet vs rookie ────────────────────────
    last_year_set = set(zip(last_year_player["player_id"],
                                last_year_player["team"]))
    sl["was_in_team_cohort_last_year"] = sl.apply(
        lambda r: (r["player_id"], r["team"]) in last_year_set,
        axis=1,
    )
    sl["was_in_nfl_last_year"] = sl["player_id"].isin(all_last_year_ids)

    arrivals = sl[~sl["was_in_team_cohort_last_year"]].copy()

    # Rookie detection — must satisfy BOTH:
    #   1. No NFL PBP footprint last year, AND
    #   2. years_exp ≤ 0 or null in Sleeper (means "no NFL experience")
    arrivals["is_rookie"] = (
        (~arrivals["was_in_nfl_last_year"])
        & arrivals["years_exp"].apply(
            lambda x: pd.isna(x) or float(x) <= 0)
    )
    arrivals["transition_type"] = "arrival"

    # For VET arrivals, find their primary prior team last year
    vet_mask = ~arrivals["is_rookie"]
    vet_prior = (
        attr[attr["season"] == PRIOR_SEASON]
        .groupby(["receiver_player_id", "team"])["targets"]
        .sum().reset_index()
        .sort_values("targets", ascending=False)
        .drop_duplicates("receiver_player_id")
        .rename(columns={"receiver_player_id": "player_id",
                         "team": "vet_prior_team",
                         "targets": "vet_prior_targets"})
    )
    arrivals = arrivals.merge(
        vet_prior[["player_id", "vet_prior_team"]],
        on="player_id", how="left",
    )
    # Career targets across all NFL years
    career = (
        attr.groupby("receiver_player_id")["targets"]
            .sum().reset_index()
            .rename(columns={"receiver_player_id": "player_id",
                             "targets": "career_targets"})
    )
    arrivals = arrivals.merge(career, on="player_id", how="left")
    print(f"  vet arrivals: {(~arrivals['is_rookie']).sum():,}")
    print(f"  rookie arrivals: {arrivals['is_rookie'].sum():,}")

    # ── Unify schemas + write ──────────────────────────────────────
    dep_out = departures[[
        "team", "player_id", "player_display_name", "position",
        "transition_type", "prior_team", "prior_season_targets",
        "is_rookie",
    ]].copy()
    dep_out["sleeper_id"] = pd.NA
    dep_out["career_targets"] = pd.NA
    dep_out["years_exp"] = pd.NA

    arr_out = arrivals[[
        "team", "player_id", "full_name", "position",
        "transition_type", "vet_prior_team", "career_targets",
        "is_rookie", "sleeper_id", "years_exp",
    ]].rename(columns={"full_name": "player_display_name",
                            "vet_prior_team": "prior_team"})
    arr_out["prior_season_targets"] = pd.NA

    out = pd.concat([dep_out, arr_out], ignore_index=True)
    out["prior_season"] = PRIOR_SEASON
    # Cast numerics to nullable Int64 to avoid object dtype mixing
    for c in ("prior_season_targets", "career_targets"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    out = out[[
        "team", "prior_season", "transition_type",
        "player_id", "sleeper_id", "player_display_name", "position",
        "is_rookie", "prior_team", "prior_season_targets",
        "career_targets", "years_exp",
    ]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()

    # Spot checks
    for sample_team in ["DAL", "BAL", "DET", "PHI"]:
        sub = out[out["team"] == sample_team]
        deps = sub[sub["transition_type"] == "departure"].dropna(
            subset=["prior_season_targets"])
        arrs = sub[sub["transition_type"] == "arrival"]
        rks = arrs[arrs["is_rookie"] == True]
        vets = arrs[arrs["is_rookie"] == False].dropna(
            subset=["career_targets"])
        print(f"=== {sample_team} 2025→2026 ===")
        print(f"  departures: {len(deps)}")
        if len(deps):
            print(deps.nlargest(5, "prior_season_targets")[
                ["player_display_name", "position",
                 "prior_season_targets"]
            ].to_string(index=False))
        print(f"  vet arrivals: {len(vets)}")
        if len(vets):
            print(vets.nlargest(5, "career_targets")[
                ["player_display_name", "position", "prior_team",
                 "career_targets"]
            ].to_string(index=False))
        print(f"  rookie arrivals: {len(rks)}")
        if len(rks):
            print(rks[["player_display_name", "position"]
                       ].head(5).to_string(index=False))
        print()


if __name__ == "__main__":
    main()

"""Build OL GAS Score table.

Output: data/ol_gas_seasons.parquet

Joins league_ol_all_seasons.parquet (already has every needed z-col)
with the OL GAS spec.

NO SOS adjustment in v1 — see lib_ol_gas.py docstring for the
public-data caveats (pass blocking is team-level; run blocking is
position-group-level).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MASTER = REPO / "data" / "league_ol_all_seasons.parquet"
PASS_ADJ = REPO / "data" / "team_pass_block_adjusted.parquet"
RUN_ADJ = REPO / "data" / "team_run_block_adjusted.parquet"
OUT = REPO / "data" / "ol_gas_seasons.parquet"

from lib_ol_gas import compute_ol_gas  # noqa: E402


def main() -> None:
    print("→ loading OL master + SOS-adjusted pass + run blocks...")
    master = pd.read_parquet(MASTER)
    pass_adj = pd.read_parquet(PASS_ADJ)
    run_adj = pd.read_parquet(RUN_ADJ)
    print(f"  master rows: {len(master):,}")
    print(f"  pass-adj team-seasons: {len(pass_adj):,}")
    print(f"  run-adj team-seasons: {len(run_adj):,}")
    print(f"  positions: {master['position'].value_counts().to_dict()}")

    # Filter to actual OL positions (T/G/C/OL). Drop the lone DE typo.
    before = len(master)
    master = master[master["position"].isin(["T", "G", "C", "OL"])
                      ].copy()
    print(f"  filtered to OL positions: {len(master):,}  "
          f"(dropped {before - len(master)})")

    # Join SOS-adjusted pass-block z-cols on (team, season)
    master = master.merge(
        pass_adj.rename(columns={"season": "season_year"})[
            ["team", "season_year",
             "adj_team_sack_rate_z", "adj_team_pressure_rate_z"]],
        on=["team", "season_year"], how="left")
    matched_pass = master["adj_team_sack_rate_z"].notna().sum()
    print(f"  with pass-adj data: {matched_pass:,} / {len(master):,} "
          f"({matched_pass/len(master):.0%})")

    # Join SOS+RB-adjusted run-block z-cols on (team, season)
    master = master.merge(
        run_adj.rename(columns={"season": "season_year"})[
            ["team", "season_year",
             "adj_team_run_epa_z", "adj_team_run_success_z",
             "adj_team_run_explosive_z"]],
        on=["team", "season_year"], how="left")
    matched_run = master["adj_team_run_epa_z"].notna().sum()
    print(f"  with run-adj data: {matched_run:,} / {len(master):,} "
          f"({matched_run/len(master):.0%})")

    graded = compute_ol_gas(master)
    print(f"  graded rows: {len(graded):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    graded.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== Top 15 OL by GAS, 2024 (≥600 snaps) ===")
    cols = ["full_name", "team", "position", "off_snaps",
            "gas_score", "gas_label", "gas_confidence",
            "gas_run_blocking_grade", "gas_pass_blocking_grade",
            "gas_discipline_grade", "gas_durability_grade"]
    cols = [c for c in cols if c in graded.columns]
    s24 = graded[(graded["season_year"] == 2024)
                  & (graded["off_snaps"] >= 600)
                  ].nlargest(15, "gas_score")
    print(s24[cols].to_string(index=False))
    print()
    print("=== Lions OL 2024 ===")
    det = graded[(graded["season_year"] == 2024)
                  & (graded["team"] == "DET")
                  & (graded["off_snaps"] >= 100)].sort_values(
        "gas_score", ascending=False)
    print(det[cols].to_string(index=False))
    print()
    print("=== YoY r ===")
    # gsis_id is mostly None on OL master; pfr_player_id is the
    # populated key.
    df_y = graded[graded["off_snaps"] >= 500][
        ["pfr_player_id", "season_year", "gas_score"]
    ].dropna(subset=["pfr_player_id"]).sort_values(
        ["pfr_player_id", "season_year"]).copy()
    df_y["next"] = df_y.groupby("pfr_player_id")["gas_score"].shift(-1)
    df_y["next_season"] = df_y.groupby("pfr_player_id"
                                          )["season_year"].shift(-1)
    yoy = df_y.dropna(subset=["next"])
    yoy = yoy[yoy["next_season"] - yoy["season_year"] == 1]
    print(f"OL YoY r (≥500 snaps both years): "
          f"{yoy['gas_score'].corr(yoy['next']):.3f}  (n={len(yoy)})")


if __name__ == "__main__":
    main()

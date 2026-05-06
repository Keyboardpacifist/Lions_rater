"""Build OC × QB-archetype interaction matrix (Feature D).

For each (skill-position player, OC, season) cell, we look up the
primary QB on the team that season (most attempts), categorize the
QB by carries/attempts ratio (Mobile dual-threat / Pocket-mobile /
Pocket passer), and measure the player's adj-z lift against their
without-this-OC career baseline.

Aggregate per (OC, qb_archetype, target_position) → mean lift across
all qualifying player-seasons in that cell.

Answers: "Reid's WR lift during Mahomes era vs Alex Smith era?"
       "Does McDaniel's lift on TEs collapse without a mobile QB?"

Output: data/oc_qb_archetype_lift.parquet
Schema: oc_name, qb_archetype, position, n_player_seasons,
        n_sample, mean_lift_z, n_distinct_players
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OC_TEAM_SEASONS = REPO / "data" / "scheme" / "curation" / "oc_team_seasons.csv"
LEAGUE_QB = REPO / "data" / "league_qb_all_seasons.parquet"
OC_PLAYER_LIFT = REPO / "data" / "oc_player_lift.parquet"
OUT = REPO / "data" / "oc_qb_archetype_lift.parquet"

# Skill positions (we don't ask "QB lift by QB archetype" — meaningless)
POSITION_CONFIGS = [
    ("data/wr_sos_adjusted_z.parquet", "data/league_wr_all_seasons.parquet",
     "adj_epa_per_target_z", "targets", "WR"),
    ("data/te_sos_adjusted_z.parquet", "data/league_te_all_seasons.parquet",
     "adj_epa_per_target_z", "targets", "TE"),
    ("data/rb_sos_adjusted_z.parquet", "data/league_rb_all_seasons.parquet",
     "adj_epa_per_rush_z", "carries", "RB"),
]

# Per-season minimum sample for a player-season cell to count
MIN_SEASON_SAMPLE = {"WR": 25, "TE": 15, "RB": 25}

# Min total cell sample for a (OC, qb_archetype, position) to be exposed
MIN_CELL_PLAYER_SEASONS = 2


def categorize_qb(carries: float, attempts: float) -> str | None:
    """QB archetype by carries/attempts ratio.
    Mobile dual-threat: ≥18% carries/att (Lamar, Hurts, Allen, Murray, Daniels)
    Pocket-mobile:      10-18% (Mahomes, Burrow, Stroud, modern hybrids)
    Pocket passer:      <10%  (Goff, Cousins, Brady, Stafford)
    """
    if pd.isna(attempts) or attempts < 100:
        return None
    carries = 0 if pd.isna(carries) else float(carries)
    ratio = carries / float(attempts)
    if ratio >= 0.18:
        return "Mobile dual-threat"
    if ratio >= 0.10:
        return "Pocket-mobile"
    return "Pocket passer"


def primary_qb_per_team_season(qb_df: pd.DataFrame) -> pd.DataFrame:
    sub = qb_df.dropna(subset=["recent_team", "season_year", "attempts"]).copy()
    sub = sub.sort_values("attempts", ascending=False)
    primary = sub.drop_duplicates(subset=["recent_team", "season_year"],
                                    keep="first")
    primary = primary[["recent_team", "season_year",
                       "player_display_name", "attempts", "carries"]].copy()
    primary["qb_archetype"] = primary.apply(
        lambda r: categorize_qb(r.get("carries"), r["attempts"]), axis=1)
    primary = primary.rename(columns={"player_display_name": "qb_name"})
    return primary


def main() -> None:
    print(f"→ loading {OC_TEAM_SEASONS.relative_to(REPO)}")
    oc_ts = pd.read_csv(OC_TEAM_SEASONS)
    oc_ts = oc_ts[oc_ts["calls_plays"].astype(str).str.upper() == "TRUE"].copy()
    oc_ts["season"] = oc_ts["season"].astype(int)
    print(f"  play-caller rows: {len(oc_ts)}")

    print(f"→ loading {LEAGUE_QB.relative_to(REPO)}")
    qbs = pd.read_parquet(LEAGUE_QB)
    primary = primary_qb_per_team_season(qbs)
    print(f"  primary QB per team-season: {len(primary)} rows; "
          f"archetype-mapped: {primary['qb_archetype'].notna().sum()}")
    print(f"  archetype distribution: "
          f"{primary['qb_archetype'].value_counts().to_dict()}")

    print(f"→ loading {OC_PLAYER_LIFT.relative_to(REPO)}")
    pl_lift = pd.read_parquet(OC_PLAYER_LIFT)
    # We need player's without-OC baseline z from this file
    baseline = pl_lift[["oc_name", "player_id", "without_oc_z"]].copy()
    print(f"  oc_player_lift rows: {len(pl_lift)}")

    all_cells = []
    for sos_path, league_path, metric_col, sample_col, position in POSITION_CONFIGS:
        sos_p = REPO / sos_path
        league_p = REPO / league_path
        if not sos_p.exists() or not league_p.exists():
            continue

        sos = pd.read_parquet(sos_p)
        league = pd.read_parquet(league_p)
        league_keep = ["player_id", "season_year"]
        if "player_display_name" in league.columns:
            league_keep.append("player_display_name")
        if "recent_team" in league.columns:
            league_keep.append("recent_team")
        if sample_col in league.columns and sample_col not in sos.columns:
            league_keep.append(sample_col)
        league = league[league_keep].copy()

        df = sos.merge(league, on=["player_id", "season_year"], how="left")
        if metric_col not in df.columns:
            continue

        # Determine sample col after merge
        if sample_col in df.columns:
            sample_col_use = sample_col
        elif "targets" in df.columns:
            sample_col_use = "targets"
        else:
            continue
        df[sample_col_use] = df[sample_col_use].fillna(0).astype(float)
        df = df.dropna(subset=[metric_col]).copy()
        df[metric_col] = df[metric_col].astype(float)

        # Tag with OC via (recent_team, season_year)
        df = df.merge(
            oc_ts[["oc_name", "team", "season"]].rename(
                columns={"team": "recent_team", "season": "season_year"}),
            on=["recent_team", "season_year"], how="left",
        )
        df = df.dropna(subset=["oc_name"]).copy()  # under-OC seasons only

        # Tag with primary QB archetype for that team-season
        df = df.merge(
            primary[["recent_team", "season_year", "qb_archetype", "qb_name"]],
            on=["recent_team", "season_year"], how="left",
        )
        df = df.dropna(subset=["qb_archetype"]).copy()

        # Filter player-seasons with adequate sample
        floor = MIN_SEASON_SAMPLE.get(position, 25)
        df = df[df[sample_col_use] >= floor].copy()

        # Bring in player's without-OC baseline
        df = df.merge(baseline, on=["oc_name", "player_id"], how="left")
        df = df.dropna(subset=["without_oc_z"]).copy()

        # Lift per player-season = season adj_z minus that player's
        # without-this-OC baseline
        df["season_lift"] = df[metric_col] - df["without_oc_z"]
        df["weight"] = df[sample_col_use]

        # Aggregate per (oc_name, qb_archetype, position)
        for (oc, arch), g in df.groupby(["oc_name", "qb_archetype"]):
            n_seasons = len(g)
            n_players = g["player_id"].nunique()
            if n_seasons < MIN_CELL_PLAYER_SEASONS:
                continue
            w_sum = g["weight"].sum()
            mean_lift = ((g["season_lift"] * g["weight"]).sum() / w_sum
                         if w_sum > 0 else float("nan"))
            all_cells.append({
                "oc_name": oc,
                "qb_archetype": arch,
                "position": position,
                "n_player_seasons": n_seasons,
                "n_distinct_players": n_players,
                "n_sample": int(w_sum),
                "mean_lift_z": float(mean_lift),
            })

    out = pd.DataFrame(all_cells)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print()
    print(f"✓ wrote {OUT.relative_to(REPO)}  rows={len(out)}")
    print()

    if out.empty:
        return

    # Spot checks
    print("=== Sample per-OC archetype matrices ===")
    for oc in ["Andy Reid", "Mike McDaniel", "Sean Payton", "Matt LaFleur",
               "Ben Johnson", "Greg Roman"]:
        sub = out[out["oc_name"] == oc]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="qb_archetype", columns="position",
                                values="mean_lift_z")
        print(f"\n--- {oc} ---")
        print(piv.to_string(float_format=lambda x: f"{x:+.2f}"))


if __name__ == "__main__":
    main()

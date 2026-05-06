"""Build OC player-lift dataset (Feature A).

For every (skill-position player, OC) pair where the player has played
both UNDER the OC and elsewhere in their career, compute the delta of
their SOS-adjusted z-score. The player serves as their own control —
this is the cleanest causal inference the data supports for "OC value-
add to a given player."

Method:
1. Tag each player-season with the play-caller via oc_team_seasons.csv.
2. For each (player, OC), split the player's career into with-OC and
   without-OC slices. Sample-weight the SOS-adjusted z-score on each side.
3. Delta = with - without. Apply Bayesian shrinkage (small samples
   pulled toward zero).
4. Filter to pairs that have minimum sample on BOTH sides.

Output: data/oc_player_lift.parquet
Schema: oc_name, position, player_id, player_name, with_oc_z,
        without_oc_z, delta, shrunk_delta, n_with, n_without
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OC_TEAM_SEASONS = REPO / "data" / "scheme" / "curation" / "oc_team_seasons.csv"
OUT_PATH = REPO / "data" / "oc_player_lift.parquet"

# (sos_path, league_path, metric_z_col, sample_col, position_label)
POSITION_CONFIGS = [
    ("data/wr_sos_adjusted_z.parquet", "data/league_wr_all_seasons.parquet",
     "adj_epa_per_target_z", "targets", "WR"),
    ("data/te_sos_adjusted_z.parquet", "data/league_te_all_seasons.parquet",
     "adj_epa_per_target_z", "targets", "TE"),
    ("data/rb_sos_adjusted_z.parquet", "data/league_rb_all_seasons.parquet",
     "adj_epa_per_rush_z", "carries", "RB"),
    ("data/qb_sos_adjusted_z.parquet", "data/league_qb_all_seasons.parquet",
     "adj_pass_epa_per_play_z", "attempts", "QB"),
]

MIN_SAMPLE_PER_SIDE = {"WR": 30, "TE": 20, "RB": 30, "QB": 100}


def _build_position(sos_path, league_path, metric_col, sample_col, position,
                     oc_ts: pd.DataFrame) -> list[dict]:
    sos_p = REPO / sos_path
    league_p = REPO / league_path
    if not sos_p.exists() or not league_p.exists():
        print(f"  ⚠ {position}: missing parquet, skipping")
        return []

    sos = pd.read_parquet(sos_p)
    league = pd.read_parquet(league_p)
    league_keep = ["player_id", "season_year"]
    if "player_display_name" in league.columns:
        league_keep.append("player_display_name")
    if "recent_team" in league.columns:
        league_keep.append("recent_team")
    # Only pull sample col from league if SOS doesn't already have it (avoid
    # _x/_y column collision from the merge).
    if sample_col in league.columns and sample_col not in sos.columns:
        league_keep.append(sample_col)
    league = league[league_keep].copy()

    # Join SOS metrics with team mapping
    df = sos.merge(league, on=["player_id", "season_year"], how="left")
    if metric_col not in df.columns:
        print(f"  ⚠ {position}: metric col {metric_col} missing, skipping")
        return []
    if sample_col not in df.columns:
        # Fall back to whatever sample column is in the SOS file
        if "targets" in df.columns:
            sample_col_use = "targets"
        else:
            print(f"  ⚠ {position}: no sample column, skipping")
            return []
    else:
        sample_col_use = sample_col

    # Tag with OC via (recent_team, season_year)
    df = df.merge(
        oc_ts[["oc_name", "team", "season"]].rename(
            columns={"team": "recent_team", "season": "season_year"}),
        on=["recent_team", "season_year"], how="left",
    )

    df[sample_col_use] = df[sample_col_use].fillna(0).astype(float)
    df = df.dropna(subset=[metric_col]).copy()
    df[metric_col] = df[metric_col].astype(float)

    min_sample = MIN_SAMPLE_PER_SIDE.get(position, 30)
    results = []

    # Players who have played under at least one of our mapped OCs
    coached_player_ocs = (
        df.dropna(subset=["oc_name"])
          .groupby("player_id")["oc_name"].apply(set)
    )

    for player_id, ocs in coached_player_ocs.items():
        player_data = df[df["player_id"] == player_id].copy()
        if "player_display_name" in player_data.columns and player_data["player_display_name"].notna().any():
            player_name = player_data["player_display_name"].dropna().iloc[0]
        else:
            player_name = str(player_id)
        for oc in ocs:
            with_data = player_data[player_data["oc_name"] == oc]
            without_data = player_data[player_data["oc_name"].fillna("") != oc]
            n_with = float(with_data[sample_col_use].sum())
            n_without = float(without_data[sample_col_use].sum())
            if n_with < min_sample or n_without < min_sample:
                continue
            with_z = ((with_data[metric_col] * with_data[sample_col_use]).sum() / n_with
                      if n_with > 0 else np.nan)
            without_z = ((without_data[metric_col] * without_data[sample_col_use]).sum() / n_without
                         if n_without > 0 else np.nan)
            delta = with_z - without_z

            # Bayesian shrinkage by harmonic-mean effective sample
            effective_n = (n_with * n_without) / (n_with + n_without)
            shrink_factor = effective_n / (effective_n + min_sample)
            shrunk_delta = shrink_factor * delta

            results.append({
                "oc_name": oc,
                "position": position,
                "player_id": player_id,
                "player_name": player_name,
                "with_oc_z": with_z,
                "without_oc_z": without_z,
                "delta": delta,
                "shrunk_delta": shrunk_delta,
                "n_with": int(n_with),
                "n_without": int(n_without),
            })
    return results


def main() -> None:
    print(f"→ loading {OC_TEAM_SEASONS.relative_to(REPO)}")
    oc_ts = pd.read_csv(OC_TEAM_SEASONS)
    oc_ts = oc_ts[oc_ts["calls_plays"].astype(str).str.upper() == "TRUE"].copy()
    oc_ts["season"] = oc_ts["season"].astype(int)
    print(f"  play-caller rows: {len(oc_ts)} ({oc_ts['oc_name'].nunique()} OCs)")
    print()

    all_results = []
    for cfg in POSITION_CONFIGS:
        print(f"→ {cfg[4]} (using {cfg[2]} weighted by {cfg[3]})")
        rows = _build_position(*cfg, oc_ts=oc_ts)
        print(f"  {len(rows)} (player, OC) pairs")
        all_results.extend(rows)

    out = pd.DataFrame(all_results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print()
    print(f"✓ wrote {OUT_PATH.relative_to(REPO)}  rows={len(out)}")
    print()

    if out.empty:
        return

    # Per-OC aggregates (weighted by min(n_with, n_without))
    out["weight"] = out[["n_with", "n_without"]].min(axis=1)
    oc_summary = (out.assign(wd=out["shrunk_delta"] * out["weight"])
                     .groupby(["oc_name", "position"])
                     .apply(lambda g: pd.Series({
                         "n_players": len(g),
                         "lift_score": g["wd"].sum() / g["weight"].sum(),
                     }))
                     .reset_index())
    print("=== Per-OC lift summary (top 15 by overall lift) ===")
    overall = (out.assign(wd=out["shrunk_delta"] * out["weight"])
                  .groupby("oc_name")
                  .apply(lambda g: pd.Series({
                      "n_players": len(g),
                      "lift_score": g["wd"].sum() / g["weight"].sum(),
                  }))
                  .reset_index()
                  .sort_values("lift_score", ascending=False))
    print(overall.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Top 10 BOOSTED player-OC pairs (shrunk_delta) ===")
    cols = ["oc_name", "position", "player_name", "with_oc_z",
            "without_oc_z", "shrunk_delta", "n_with", "n_without"]
    print(out.nlargest(10, "shrunk_delta")[cols].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))
    print()

    print("=== Top 10 DRAGGED player-OC pairs (shrunk_delta) ===")
    print(out.nsmallest(10, "shrunk_delta")[cols].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()

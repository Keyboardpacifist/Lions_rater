"""Per-team situational scheme fingerprint.

Output: data/scheme/team_scheme_fingerprint.parquet

Captures the data-derivable side of a team's scheme identity — independent
of route taxonomy (which broke at 2023). Built directly from PBP fields
(down, ydstogo, yardline_100, play_type, no_huddle, defense_man_zone_type,
number_of_pass_rushers) so it's stable across the full 2016-2025 window.

This is the canvas the rebuilt OC page paints on top of, alongside the
human-curated coaching tree + signature concepts and the auto-derived
route fingerprint already in team_passing_fingerprint.parquet.

Dimensions
----------
- dnd_pass_rate:    pass rate by down-and-distance bucket (7 buckets)
- field_pass_rate:  pass rate in red zone / goal-line / backed-up
- shotgun_rate:     shotgun usage by D&D bucket
- tempo:            no-huddle rate (overall + 2-min)
- run_gap_share:    share of runs going to end / tackle / guard
- vs_coverage:      pass-call rate + avg air yards vs man / vs zone (2018+)
- pressure_faced:   share of dropbacks facing 5+ pass rushers (2018+)

Schema (long)
-------------
team, season, dimension, category, n, value, league_value,
value_delta, value_z

  value       = the team's metric (rate or avg)
  league_value= leaguewide same-season-same-cell value
  value_delta = team_value - league_value
  value_z     = z within (season, dimension, category)
  n           = sample size (denominator) for the team-cell

So "DET 2024 dnd_pass_rate / 1st_10 / value=0.62" means DET passed 62%
of 1st & 10s in 2024 — pair with league_value (~0.55) and value_z (+1.4)
to read it as ~1.4σ pass-happy on early downs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
OUT_DIR = REPO / "data" / "scheme"
OUT = OUT_DIR / "team_scheme_fingerprint.parquet"

# Coverage / pressure fields are NGS-derived — only reliable from 2018+.
COVERAGE_MIN_SEASON = 2018


# ----------------------------------------------------------------------
# Bucketers
# ----------------------------------------------------------------------

def _dnd_bucket(down: float, ydstogo: float) -> str:
    if pd.isna(down) or pd.isna(ydstogo):
        return None
    d = int(down)
    y = float(ydstogo)
    if d == 1:
        # Treat 1st & 10 and 1st-and-anything-else (penalty-shifted) together.
        return "1st_10" if y >= 7 else "1st_short"
    if d == 2:
        if y <= 3:
            return "2nd_short"
        if y <= 7:
            return "2nd_med"
        return "2nd_long"
    if d == 3:
        if y <= 3:
            return "3rd_short"
        if y <= 7:
            return "3rd_med"
        return "3rd_long"
    if d == 4:
        return "4th"
    return None


def _field_bucket(yardline_100: float) -> str:
    if pd.isna(yardline_100):
        return None
    y = float(yardline_100)
    if y <= 5:
        return "gl"          # goal-line
    if y <= 20:
        return "rz"          # red zone (excluding GL)
    if y >= 90:
        return "backed_up"   # own 10 or worse
    return None              # rest of field — not a fingerprint cell


# ----------------------------------------------------------------------
# Z-score helper
# ----------------------------------------------------------------------

def _zscore_within(df: pd.DataFrame, group_cols: list[str],
                   value_col: str) -> pd.Series:
    grp = df.groupby(group_cols)[value_col]
    mean = grp.transform("mean")
    std = grp.transform(lambda s: s.std(ddof=0))
    z = (df[value_col] - mean) / std.replace(0, np.nan)
    return z.fillna(0)


# ----------------------------------------------------------------------
# Builders — each returns a long-format DataFrame with the unified schema
# ----------------------------------------------------------------------

def _build_rate_dim(df: pd.DataFrame, group_cols: list[str],
                    cat_col: str, num_col: str, dim_name: str,
                    sample_floor: int = 20) -> pd.DataFrame:
    """Generic rate builder: rate = num.sum() / count() per (team, season, cat).

    df must already be filtered to the relevant universe (e.g. only the plays
    that count for this dimension's denominator).
    """
    agg = (
        df.groupby(["posteam", "season", cat_col])
          .agg(n=(num_col, "size"), num=(num_col, "sum"))
          .reset_index()
    )
    agg = agg[agg["n"] >= sample_floor].copy()
    agg["value"] = agg["num"] / agg["n"]

    # Leaguewide value per (season, category)
    league = (
        df.groupby(["season", cat_col])
          .agg(league_n=(num_col, "size"), league_num=(num_col, "sum"))
          .reset_index()
    )
    league["league_value"] = league["league_num"] / league["league_n"]
    agg = agg.merge(league[["season", cat_col, "league_value"]],
                    on=["season", cat_col])

    agg["value_delta"] = agg["value"] - agg["league_value"]
    agg["value_z"] = _zscore_within(agg, ["season", cat_col], "value")

    out = agg.rename(columns={"posteam": "team", cat_col: "category"})
    out["dimension"] = dim_name
    return out[["team", "season", "dimension", "category", "n",
                "value", "league_value", "value_delta", "value_z"]]


def _build_mean_dim(df: pd.DataFrame, cat_col: str, value_col: str,
                    dim_name: str, sample_floor: int = 20) -> pd.DataFrame:
    """Mean builder: value = value_col.mean() per (team, season, cat)."""
    sub = df.dropna(subset=[value_col])
    agg = (
        sub.groupby(["posteam", "season", cat_col])
           .agg(n=(value_col, "size"), value=(value_col, "mean"))
           .reset_index()
    )
    agg = agg[agg["n"] >= sample_floor].copy()

    league = (
        sub.groupby(["season", cat_col])[value_col].mean()
           .rename("league_value").reset_index()
    )
    agg = agg.merge(league, on=["season", cat_col])
    agg["value_delta"] = agg["value"] - agg["league_value"]
    agg["value_z"] = _zscore_within(agg, ["season", cat_col], "value")

    out = agg.rename(columns={"posteam": "team", cat_col: "category"})
    out["dimension"] = dim_name
    return out[["team", "season", "dimension", "category", "n",
                "value", "league_value", "value_delta", "value_z"]]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    print("→ loading PBP...")
    pbp = pd.read_parquet(PBP)
    pbp["season"] = pbp["season"].astype(int)
    print(f"  total rows: {len(pbp):,}")

    # Restrict to plays where a team actually has the ball with a real D&D.
    # Drop kickoffs, punts, kneels, spikes, no_play (penalty), FGs, XPs.
    real_plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    real_plays = real_plays.dropna(subset=["posteam", "down", "ydstogo"])
    real_plays["down"] = real_plays["down"].astype(int)
    real_plays["is_pass"] = (real_plays["play_type"] == "pass").astype(int)
    real_plays["is_shotgun"] = real_plays["shotgun"].fillna(0).astype(int)
    real_plays["is_no_huddle"] = (
        real_plays["no_huddle"].fillna(0).astype(int)
    )
    real_plays["dnd"] = real_plays.apply(
        lambda r: _dnd_bucket(r["down"], r["ydstogo"]), axis=1
    )
    real_plays["field"] = real_plays["yardline_100"].apply(_field_bucket)

    print(f"  scrimmage plays: {len(real_plays):,}")

    parts = []

    # 1. dnd_pass_rate — pass rate by D&D
    print("→ dnd_pass_rate")
    sub = real_plays.dropna(subset=["dnd"])
    parts.append(_build_rate_dim(
        sub, group_cols=["posteam", "season", "dnd"],
        cat_col="dnd", num_col="is_pass", dim_name="dnd_pass_rate"
    ))

    # 2. field_pass_rate — pass rate in RZ / GL / backed_up
    print("→ field_pass_rate")
    sub = real_plays.dropna(subset=["field"])
    parts.append(_build_rate_dim(
        sub, group_cols=["posteam", "season", "field"],
        cat_col="field", num_col="is_pass", dim_name="field_pass_rate",
        sample_floor=10,  # GL plays are scarce
    ))

    # 3. shotgun_rate by D&D
    print("→ shotgun_rate")
    sub = real_plays.dropna(subset=["dnd"])
    parts.append(_build_rate_dim(
        sub, group_cols=["posteam", "season", "dnd"],
        cat_col="dnd", num_col="is_shotgun", dim_name="shotgun_rate"
    ))

    # 4. tempo — overall no-huddle rate + 2-min no-huddle rate
    print("→ tempo")
    real_plays["tempo_cat"] = "overall"
    overall = _build_rate_dim(
        real_plays, group_cols=["posteam", "season", "tempo_cat"],
        cat_col="tempo_cat", num_col="is_no_huddle", dim_name="tempo",
        sample_floor=200,
    )
    overall["category"] = "overall_no_huddle"
    parts.append(overall)

    # 2-min drill: end of half (q2/q4) with <= 120 sec remaining
    two_min = real_plays[
        (real_plays["qtr"].isin([2, 4]))
        & (real_plays["quarter_seconds_remaining"] <= 120)
    ].copy()
    if len(two_min):
        two_min["tempo_cat"] = "2min_no_huddle"
        two_min_d = _build_rate_dim(
            two_min, group_cols=["posteam", "season", "tempo_cat"],
            cat_col="tempo_cat", num_col="is_no_huddle", dim_name="tempo",
            sample_floor=20,
        )
        parts.append(two_min_d)

    # 5. run_gap_share — what % of runs go end / tackle / guard
    print("→ run_gap_share")
    runs = real_plays[
        (real_plays["is_pass"] == 0)
        & (real_plays["run_gap"].notna())
    ].copy()
    if len(runs):
        # Per (team, season, gap): count / team_total_runs
        team_runs = (
            runs.groupby(["posteam", "season"]).size()
                .rename("team_total").reset_index()
        )
        cell = (
            runs.groupby(["posteam", "season", "run_gap"]).size()
                .rename("n").reset_index()
        )
        cell = cell.merge(team_runs, on=["posteam", "season"])
        cell = cell[cell["team_total"] >= 80].copy()
        cell["value"] = cell["n"] / cell["team_total"]

        league_runs = runs.groupby("season").size().rename("ltot")
        league_cell = (
            runs.groupby(["season", "run_gap"]).size().rename("lnum")
                .reset_index().merge(league_runs, on="season")
        )
        league_cell["league_value"] = league_cell["lnum"] / league_cell["ltot"]
        cell = cell.merge(
            league_cell[["season", "run_gap", "league_value"]],
            on=["season", "run_gap"],
        )
        cell["value_delta"] = cell["value"] - cell["league_value"]
        cell["value_z"] = _zscore_within(cell, ["season", "run_gap"], "value")
        cell = cell.rename(columns={"posteam": "team",
                                     "run_gap": "category"})
        cell["dimension"] = "run_gap_share"
        parts.append(cell[["team", "season", "dimension", "category", "n",
                           "value", "league_value", "value_delta",
                           "value_z"]])

    # 6. vs_coverage — pass-call rate + avg air yards vs man / vs zone (2018+)
    print("→ vs_coverage")
    cov = real_plays[real_plays["season"] >= COVERAGE_MIN_SEASON].copy()
    cov = cov[cov["defense_man_zone_type"].notna()
              & (cov["defense_man_zone_type"].astype(str).str.strip() != "")]
    cov["mz"] = cov["defense_man_zone_type"].map({
        "MAN_COVERAGE": "vs_man",
        "ZONE_COVERAGE": "vs_zone",
    })
    cov = cov.dropna(subset=["mz"])

    # 5a — pass-call rate vs each shell
    parts.append(_build_rate_dim(
        cov, group_cols=["posteam", "season", "mz"],
        cat_col="mz", num_col="is_pass",
        dim_name="vs_coverage_pass_rate",
        sample_floor=80,
    ))

    # 5b — average air yards on dropbacks vs each shell
    pass_only = cov[cov["is_pass"] == 1].copy()
    parts.append(_build_mean_dim(
        pass_only, cat_col="mz", value_col="air_yards",
        dim_name="vs_coverage_avg_ay", sample_floor=40,
    ))

    # 7. pressure_faced — share of dropbacks facing 5+ rushers (2018+)
    print("→ pressure_faced")
    pr = real_plays[
        (real_plays["season"] >= COVERAGE_MIN_SEASON)
        & (real_plays["is_pass"] == 1)
        & (real_plays["number_of_pass_rushers"].notna())
    ].copy()
    pr["five_plus"] = (pr["number_of_pass_rushers"] >= 5).astype(int)
    pr["press_cat"] = "rate_5plus"
    parts.append(_build_rate_dim(
        pr, group_cols=["posteam", "season", "press_cat"],
        cat_col="press_cat", num_col="five_plus",
        dim_name="pressure_faced",
        sample_floor=80,
    ))

    fingerprint = pd.concat(parts, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint.to_parquet(OUT, index=False)
    print()
    print(f"✓ wrote {OUT.relative_to(REPO)}  rows={len(fingerprint):,}")
    print()

    # ------------------------------------------------------------------
    # Spot checks
    # ------------------------------------------------------------------
    cols = ["category", "n", "value", "league_value",
            "value_delta", "value_z"]

    print("=== DET 2024 dnd_pass_rate ===")
    det = fingerprint[
        (fingerprint["team"] == "DET")
        & (fingerprint["season"] == 2024)
        & (fingerprint["dimension"] == "dnd_pass_rate")
    ].sort_values("category")
    print(det[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Most pass-happy 1st & 10 teams 2024 ===")
    top = fingerprint[
        (fingerprint["dimension"] == "dnd_pass_rate")
        & (fingerprint["category"] == "1st_10")
        & (fingerprint["season"] == 2024)
    ].nlargest(5, "value")
    print(top[["team"] + cols].to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Highest no-huddle 2024 ===")
    nh = fingerprint[
        (fingerprint["dimension"] == "tempo")
        & (fingerprint["category"] == "overall_no_huddle")
        & (fingerprint["season"] == 2024)
    ].nlargest(5, "value")
    print(nh[["team"] + cols].to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== DET 2024 run_gap_share ===")
    rg = fingerprint[
        (fingerprint["team"] == "DET")
        & (fingerprint["season"] == 2024)
        & (fingerprint["dimension"] == "run_gap_share")
    ].sort_values("category")
    print(rg[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Most end-run-heavy teams 2024 (outside zone proxy) ===")
    er = fingerprint[
        (fingerprint["dimension"] == "run_gap_share")
        & (fingerprint["category"] == "end")
        & (fingerprint["season"] == 2024)
    ].nlargest(5, "value")
    print(er[["team"] + cols].to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Deepest avg air yards vs man, 2024 ===")
    am = fingerprint[
        (fingerprint["dimension"] == "vs_coverage_avg_ay")
        & (fingerprint["category"] == "vs_man")
        & (fingerprint["season"] == 2024)
    ].nlargest(5, "value")
    print(am[["team"] + cols].to_string(index=False,
          float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()

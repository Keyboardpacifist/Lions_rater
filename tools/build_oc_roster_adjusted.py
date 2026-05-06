"""Add roster-adjusted residual z-scores to the OC parquets.

For each outcome stat (EPA/play, pass EPA, rush EPA, success rate,
explosive pass/rush, third down, red zone, win pct), fit a linear
regression on roster-quality proxies (cap allocation + draft capital,
and where available career proxies from team_context). The residual
= actual outcome − expected-from-roster = the OC's value-add.

Output: same parquets, with new `*_adj_z` columns appended.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

CAREER_PATH = REPO / "data" / "master_ocs_with_z.parquet"
SEASON_PATH = REPO / "data" / "master_ocs_2024_with_z.parquet"
TEAM_CONTEXT_PATH = REPO / "data" / "team_context.parquet"
OC_TEAM_SEASONS_PATH = REPO / "data" / "scheme" / "curation" / "oc_team_seasons.csv"

OUTCOME_STATS = [
    "epa_per_play", "pass_epa_per_play", "rush_epa_per_play",
    "success_rate", "explosive_pass_rate", "explosive_rush_rate",
    "third_down_rate", "red_zone_td_rate", "win_pct",
]


def _safe_lr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Predict y from X via OLS with intercept; impute NaN in X with column means.
    Returns predicted y (same length as input) or fallback `y.mean()` broadcast.
    """
    X = X.astype(float).copy()
    y = y.astype(float).copy()
    # Mean-impute NaN columns
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            m = np.nanmean(col)
            X[np.isnan(col), j] = m if np.isfinite(m) else 0.0
    valid = ~np.isnan(y)
    if valid.sum() < 5:
        return np.full_like(y, np.nanmean(y))
    Xv = np.hstack([np.ones((X[valid].shape[0], 1)), X[valid]])
    yv = y[valid]
    try:
        beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    except np.linalg.LinAlgError:
        return np.full_like(y, np.nanmean(y))
    Xall = np.hstack([np.ones((X.shape[0], 1)), X])
    pred = Xall @ beta
    return pred


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x); sd = np.nanstd(x, ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(x)
    return (x - mu) / sd


def adjust_dataframe(df: pd.DataFrame, predictor_cols: list[str],
                     outcome_cols: list[str]) -> pd.DataFrame:
    """Compute residual *_adj_z columns. Modifies df in place AND returns it.
    Predictors are mean-imputed for missing values.
    """
    out = df.copy()
    if not predictor_cols:
        return out
    have = [c for c in predictor_cols if c in out.columns]
    if not have:
        print(f"  ⚠ no predictor columns present, skipping adjustment")
        return out
    X = out[have].to_numpy()
    for stat in outcome_cols:
        if stat not in out.columns:
            continue
        y = out[stat].to_numpy()
        pred = _safe_lr(X, y)
        residual = y - pred
        adj_z = _zscore(residual)
        out[f"{stat}_adj_z"] = adj_z
        out[f"{stat}_predicted"] = pred
        out[f"{stat}_residual"] = residual
    return out


def build_career_roster_proxies(career: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-OC career roster proxies from team_context.parquet,
    using oc_team_seasons.csv to know which (team, season) cells belong
    to each OC. Returns the career df with two new columns:
        oc_team_rating_avg, oc_top5_apy_pct_avg
    OCs whose tenure isn't in oc_team_seasons.csv get NaN for both.
    """
    if not TEAM_CONTEXT_PATH.exists() or not OC_TEAM_SEASONS_PATH.exists():
        career["oc_team_rating_avg"] = np.nan
        career["oc_top5_apy_pct_avg"] = np.nan
        return career

    tc = pd.read_parquet(TEAM_CONTEXT_PATH)
    oc_ts = pd.read_csv(OC_TEAM_SEASONS_PATH)
    oc_ts["season"] = oc_ts["season"].astype(int)
    tc["season"] = tc["season"].astype(int)

    keep_cols = [c for c in ["team_rating", "top5_apy_pct_of_cap"] if c in tc.columns]
    if not keep_cols:
        career["oc_team_rating_avg"] = np.nan
        career["oc_top5_apy_pct_avg"] = np.nan
        return career

    joined = oc_ts.merge(tc[["team", "season"] + keep_cols],
                         on=["team", "season"], how="left")
    agg = joined.groupby("oc_name")[keep_cols].mean().reset_index()
    agg = agg.rename(columns={
        "team_rating": "oc_team_rating_avg",
        "top5_apy_pct_of_cap": "oc_top5_apy_pct_avg",
    })

    out = career.merge(agg, left_on="coordinator", right_on="oc_name",
                       how="left").drop(columns=["oc_name"], errors="ignore")
    if "oc_team_rating_avg" not in out.columns:
        out["oc_team_rating_avg"] = np.nan
    if "oc_top5_apy_pct_avg" not in out.columns:
        out["oc_top5_apy_pct_avg"] = np.nan
    return out


def main() -> None:
    # ─────────────────────────────────────────────────────────
    # 2024-only file: predictors already baked in (off_cap_pct,
    # off_draft_capital, off_cap_share). Quick win.
    # ─────────────────────────────────────────────────────────
    print(f"→ loading {SEASON_PATH.relative_to(REPO)}")
    season = pd.read_parquet(SEASON_PATH)
    season_predictors = [c for c in
        ["off_cap_pct", "off_draft_capital", "off_cap_share"]
        if c in season.columns]
    print(f"  predictors: {season_predictors}")
    season_adj = adjust_dataframe(season, season_predictors, OUTCOME_STATS)
    season_adj.to_parquet(SEASON_PATH, index=False)
    print(f"  ✓ wrote {SEASON_PATH.relative_to(REPO)} ({len(season_adj)} rows)")
    print()

    # ─────────────────────────────────────────────────────────
    # Career file: enrich with per-OC roster proxies aggregated from
    # team_context.parquet via oc_team_seasons.csv mapping. OCs not
    # in the mapping get NaN proxies → fall back to no-adjustment
    # (residuals = actual - mean).
    # ─────────────────────────────────────────────────────────
    print(f"→ loading {CAREER_PATH.relative_to(REPO)}")
    career = pd.read_parquet(CAREER_PATH)
    career = build_career_roster_proxies(career)
    n_with_proxies = career["oc_team_rating_avg"].notna().sum()
    print(f"  enriched career rows with team-context proxies: {n_with_proxies}/{len(career)}")
    career_predictors = [c for c in
        ["oc_team_rating_avg", "oc_top5_apy_pct_avg"]
        if c in career.columns]
    print(f"  predictors: {career_predictors}")
    career_adj = adjust_dataframe(career, career_predictors, OUTCOME_STATS)
    career_adj.to_parquet(CAREER_PATH, index=False)
    print(f"  ✓ wrote {CAREER_PATH.relative_to(REPO)} ({len(career_adj)} rows)")
    print()

    # ─────────────────────────────────────────────────────────
    # Spot checks
    # ─────────────────────────────────────────────────────────
    print("=== 2024 — biggest roster-adjustment SHIFTS (raw vs adjusted EPA/play z) ===")
    s = season_adj[["coordinator", "team", "epa_per_play",
                    "epa_per_play_z", "epa_per_play_adj_z",
                    "off_cap_pct", "off_draft_capital"]].copy()
    s["shift"] = s["epa_per_play_adj_z"] - s["epa_per_play_z"]
    big = s.reindex(s["shift"].abs().sort_values(ascending=False).index)[:10]
    print(big.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()

    print("=== Career — biggest roster-adjustment SHIFTS for OCs in our mapping ===")
    c = career_adj[career_adj["oc_team_rating_avg"].notna()][[
        "coordinator", "epa_per_play", "epa_per_play_z", "epa_per_play_adj_z",
        "oc_team_rating_avg",
    ]].copy()
    c["shift"] = c["epa_per_play_adj_z"] - c["epa_per_play_z"]
    cbig = c.reindex(c["shift"].abs().sort_values(ascending=False).index)[:10]
    print(cbig.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()

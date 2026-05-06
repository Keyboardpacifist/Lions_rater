"""Per-OC fulcrum / clutch profile.

For each OC × leverage definition, compute:
- fulcrum metrics (EPA/play, success rate) on plays the definition keeps
- non-fulcrum metrics (same OC, plays the definition excludes)
- elevation index = fulcrum − non-fulcrum (does this OC step up or fade?)
- roster-adjusted residual z-scores (for OCs we have proxies for)

Five leverage definitions:
1. wp_volatility   — |wpa| ≥ 0.05 (single-play game-swinging)
2. high_stakes     — 4Q close + RZ + 3rd & medium+ + 2-min drill (rule-based)
3. epa_x_leverage  — every play weighted by |wpa| (continuous)
4. hybrid          — |wpa| × situation_multiplier (composite)
5. all_plays       — baseline, every play equally weighted

Output: data/scheme/oc_fulcrum_profile.parquet
Schema (long format):
    oc_name, leverage_def, metric, n_fulcrum, n_non_fulcrum,
    fulcrum_value, non_fulcrum_value, elevation,
    fulcrum_z, fulcrum_adj_z, elevation_z

OCs use the (team, season) ↔ OC mapping in
data/scheme/curation/oc_team_seasons.csv. OCs not in the mapping
get no fulcrum row (downstream UI shows empty state).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP_PATH = REPO / "data" / "game_pbp.parquet"
OC_TEAM_SEASONS_PATH = REPO / "data" / "scheme" / "curation" / "oc_team_seasons.csv"
CAREER_OC_PATH = REPO / "data" / "master_ocs_with_z.parquet"
OUT_PATH = REPO / "data" / "scheme" / "oc_fulcrum_profile.parquet"

LEVERAGE_DEFS = ["wp_volatility", "high_stakes", "epa_x_leverage",
                 "hybrid", "all_plays"]


def _binary_mask(pbp: pd.DataFrame, definition: str) -> pd.Series | None:
    """Return per-play mask for binary fulcrum definitions, or None for
    weight-based defs."""
    if definition == "wp_volatility":
        return pbp["wpa"].abs().fillna(0) >= 0.05
    if definition == "high_stakes":
        q4_close = (pbp["qtr"].fillna(0) == 4) & pbp["pos_wp"].between(0.2, 0.8)
        rz = pbp["yardline_100"].fillna(99) <= 20
        third_med_long = (pbp["down"].fillna(0) == 3) & (pbp["ydstogo"].fillna(0) >= 4)
        # 2-min drill: end of half (q2 or q4) with ≤120 sec remaining
        two_min = (pbp["qtr"].fillna(0).isin([2, 4])
                   & (pbp["quarter_seconds_remaining"].fillna(999) <= 120))
        return q4_close | rz | third_med_long | two_min
    if definition == "all_plays":
        return pd.Series(True, index=pbp.index)
    return None  # weight-based defs handled separately


def _continuous_weight(pbp: pd.DataFrame, definition: str) -> pd.Series | None:
    """Return per-play weight for weight-based fulcrum definitions, or None."""
    if definition == "epa_x_leverage":
        return pbp["wpa"].abs().fillna(0.0)
    if definition == "hybrid":
        base = pbp["wpa"].abs().fillna(0.0)
        rz = pbp["yardline_100"].fillna(99) <= 20
        third_med_long = (pbp["down"].fillna(0) == 3) & (pbp["ydstogo"].fillna(0) >= 4)
        q4_close = (pbp["qtr"].fillna(0) == 4) & pbp["pos_wp"].between(0.2, 0.8)
        two_min = (pbp["qtr"].fillna(0).isin([2, 4])
                   & (pbp["quarter_seconds_remaining"].fillna(999) <= 120))
        situ = (rz | third_med_long | q4_close | two_min).astype(float)
        # 1.5x weight for any "money situation", 1.0x otherwise; multiply by |wpa|
        return base * (1.0 + 0.5 * situ)
    return None


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    valid = ~np.isnan(x)
    if valid.sum() < 3:
        return np.full_like(x, np.nan)
    mu = np.nanmean(x); sd = np.nanstd(x[valid], ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(x)
    return (x - mu) / sd


def _safe_lr_residual(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS residual with mean-imputed missing predictors. NaN if too few data."""
    X = X.astype(float).copy(); y = y.astype(float).copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            m = np.nanmean(col)
            X[np.isnan(col), j] = m if np.isfinite(m) else 0.0
    valid = ~np.isnan(y)
    if valid.sum() < 5:
        return np.full_like(y, np.nan)
    Xv = np.hstack([np.ones((X[valid].shape[0], 1)), X[valid]])
    yv = y[valid]
    try:
        beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    except np.linalg.LinAlgError:
        return np.full_like(y, np.nan)
    pred = np.hstack([np.ones((X.shape[0], 1)), X]) @ beta
    return y - pred


def main() -> None:
    print(f"→ loading PBP {PBP_PATH.relative_to(REPO)}")
    pbp = pd.read_parquet(PBP_PATH)
    print(f"  total rows: {len(pbp):,}")

    # Restrict to scrimmage plays only (no kickoffs/punts/kneels)
    pbp = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    pbp = pbp.dropna(subset=["posteam", "season", "epa"])
    pbp["season"] = pbp["season"].astype(int)
    # Offense-relative win probability — PBP only stores home_wp/away_wp.
    pbp["pos_wp"] = np.where(
        pbp["posteam"] == pbp["home_team"],
        pbp["home_wp"], pbp["away_wp"],
    )
    print(f"  scrimmage plays: {len(pbp):,}")

    # Join OC mapping
    print(f"→ loading {OC_TEAM_SEASONS_PATH.relative_to(REPO)}")
    oc_ts = pd.read_csv(OC_TEAM_SEASONS_PATH)
    oc_ts["season"] = oc_ts["season"].astype(int)
    pbp = pbp.merge(
        oc_ts[["oc_name", "team", "season"]],
        left_on=["posteam", "season"], right_on=["team", "season"],
        how="inner",
    )
    print(f"  plays with OC mapping: {len(pbp):,} "
          f"({pbp['oc_name'].nunique()} OCs)")

    # Load career roster proxies for adjustment
    career = pd.read_parquet(CAREER_OC_PATH)
    proxy_cols = [c for c in ["oc_team_rating_avg", "oc_top5_apy_pct_avg"]
                  if c in career.columns]
    proxies = career[["coordinator"] + proxy_cols].rename(
        columns={"coordinator": "oc_name"})

    # Per-OC: success column already exists in PBP
    if "success" not in pbp.columns:
        pbp["success"] = (pbp["epa"] > 0).astype(int)

    rows = []
    for definition in LEVERAGE_DEFS:
        mask = _binary_mask(pbp, definition)
        weight = _continuous_weight(pbp, definition)

        if mask is not None:
            # Binary partition
            for metric_col, metric_name in [("epa", "epa_per_play"),
                                             ("success", "success_rate")]:
                # Fulcrum stats per OC
                fulcrum_grp = pbp[mask].groupby("oc_name")[metric_col]
                fulcrum_mean = fulcrum_grp.mean()
                n_fulcrum = fulcrum_grp.size()

                # Non-fulcrum stats per OC (skip for "all_plays")
                if definition == "all_plays":
                    non_fulcrum_mean = pd.Series(np.nan, index=fulcrum_mean.index)
                    n_non_fulcrum = pd.Series(0, index=fulcrum_mean.index)
                else:
                    nf_grp = pbp[~mask].groupby("oc_name")[metric_col]
                    non_fulcrum_mean = nf_grp.mean()
                    n_non_fulcrum = nf_grp.size()

                idx = fulcrum_mean.index.union(non_fulcrum_mean.index)
                fv = fulcrum_mean.reindex(idx)
                nfv = non_fulcrum_mean.reindex(idx)
                elev = fv - nfv
                for oc in idx:
                    rows.append({
                        "oc_name": oc,
                        "leverage_def": definition,
                        "metric": metric_name,
                        "n_fulcrum": int(n_fulcrum.get(oc, 0)),
                        "n_non_fulcrum": int(n_non_fulcrum.get(oc, 0)),
                        "fulcrum_value": float(fv.get(oc, np.nan)),
                        "non_fulcrum_value": float(nfv.get(oc, np.nan)),
                        "elevation": float(elev.get(oc, np.nan)),
                    })
        else:
            # Continuous weighted: compute weighted mean of metric per OC
            assert weight is not None
            for metric_col, metric_name in [("epa", "epa_per_play"),
                                             ("success", "success_rate")]:
                w = weight.fillna(0.0)
                wm = pbp[metric_col].astype(float) * w
                grp_w = w.groupby(pbp["oc_name"]).sum()
                grp_wm = wm.groupby(pbp["oc_name"]).sum()
                weighted_mean = grp_wm / grp_w.replace(0, np.nan)

                grp_unweighted = pbp.groupby("oc_name")[metric_col].mean()
                grp_n = pbp.groupby("oc_name").size()

                idx = weighted_mean.index.union(grp_unweighted.index)
                fv = weighted_mean.reindex(idx)
                nfv = grp_unweighted.reindex(idx)
                elev = fv - nfv
                for oc in idx:
                    rows.append({
                        "oc_name": oc,
                        "leverage_def": definition,
                        "metric": metric_name,
                        # For weighted defs n_fulcrum is "sum of weights × 100"
                        # so it's interpretable. Round to int.
                        "n_fulcrum": int(round(float(grp_w.get(oc, 0)) * 100)),
                        "n_non_fulcrum": int(grp_n.get(oc, 0)),
                        "fulcrum_value": float(fv.get(oc, np.nan)),
                        "non_fulcrum_value": float(nfv.get(oc, np.nan)),
                        "elevation": float(elev.get(oc, np.nan)),
                    })

    fulcrum_df = pd.DataFrame(rows)

    # Z-scores within (leverage_def, metric)
    fulcrum_df["fulcrum_z"] = (
        fulcrum_df.groupby(["leverage_def", "metric"])["fulcrum_value"]
                  .transform(lambda s: _zscore(s.values))
    )
    fulcrum_df["elevation_z"] = (
        fulcrum_df.groupby(["leverage_def", "metric"])["elevation"]
                  .transform(lambda s: _zscore(s.values))
    )

    # Roster-adjusted z-scores (career proxies)
    fulcrum_df = fulcrum_df.merge(proxies, on="oc_name", how="left")
    if proxy_cols:
        adj_z = []
        for (ld, mt), sub in fulcrum_df.groupby(["leverage_def", "metric"]):
            X = sub[proxy_cols].to_numpy()
            y = sub["fulcrum_value"].to_numpy()
            resid = _safe_lr_residual(X, y)
            adj_z.append(pd.Series(_zscore(resid), index=sub.index,
                                    name="fulcrum_adj_z"))
        fulcrum_df["fulcrum_adj_z"] = pd.concat(adj_z).sort_index()
    else:
        fulcrum_df["fulcrum_adj_z"] = np.nan

    fulcrum_df = fulcrum_df.drop(columns=proxy_cols, errors="ignore")
    out_cols = ["oc_name", "leverage_def", "metric", "n_fulcrum",
                "n_non_fulcrum", "fulcrum_value", "non_fulcrum_value",
                "elevation", "fulcrum_z", "fulcrum_adj_z", "elevation_z"]
    fulcrum_df = fulcrum_df[out_cols]
    fulcrum_df.to_parquet(OUT_PATH, index=False)
    print(f"\n✓ wrote {OUT_PATH.relative_to(REPO)}  rows={len(fulcrum_df):,}  "
          f"OCs={fulcrum_df['oc_name'].nunique()}")
    print()

    # Spot checks
    print("=== Ben Johnson — full clutch profile (all leverage defs) ===")
    bj = fulcrum_df[fulcrum_df["oc_name"] == "Ben Johnson"]
    print(bj.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Best EPA elevation (clutch step-up) — wp_volatility def ===")
    elev = fulcrum_df[(fulcrum_df["leverage_def"] == "wp_volatility")
                      & (fulcrum_df["metric"] == "epa_per_play")]
    top_elev = elev.nlargest(5, "elevation")[["oc_name", "n_fulcrum",
                "fulcrum_value", "non_fulcrum_value", "elevation",
                "elevation_z"]]
    print(top_elev.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    print("=== Worst EPA elevation (fades in clutch) — wp_volatility def ===")
    bot_elev = elev.nsmallest(5, "elevation")[["oc_name", "n_fulcrum",
                "fulcrum_value", "non_fulcrum_value", "elevation",
                "elevation_z"]]
    print(bot_elev.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

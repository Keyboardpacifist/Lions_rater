"""Scheme Shift Alpha — factor 4 of 4.

The premise: when a team's offensive scheme changes year-over-year,
receivers whose career route profile matches the NEW distribution
gain alpha; receivers locked into the OLD distribution lose alpha.

We measure the shift from team route distributions directly, no OC
tracking required. The team's offense IS the data — if it's running
more SLANTs and fewer GOs in 2025 than 2024, that IS a scheme shift,
regardless of who the OC is.

Three outputs (data/scheme/scheme_shift_*.parquet):

1. team_route_drift.parquet — per (team, season-pair), Jensen-Shannon
   divergence between consecutive seasons' route distributions. High
   JSD = team's offense materially changed. Low JSD = continuity.

2. team_route_change.parquet — per (team, route), the share delta
   between 2024 and 2025. Shows which routes a team is running MORE
   of vs LESS of.

3. receiver_scheme_fit.parquet — per (team, receiver), the fit score
   between the receiver's career route profile and the team's 2025
   distribution. Receivers with high fit benefit if the team's
   2026 scheme is similar to 2025 (continuation).

Production model (Phase 2): pair this with an OC-continuity table to
detect "new OC = expected scheme shift" cases. For now, we project
continuation (2026 ≈ 2025) which is correct for ~80% of teams.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TEAM_FP = REPO / "data" / "scheme" / "team_passing_fingerprint.parquet"
PLAYER_RP = REPO / "data" / "scheme" / "player_route_profile.parquet"
ATTRIBUTION = REPO / "data" / "scheme" / "team_route_attribution.parquet"
OUT_DIR = REPO / "data" / "scheme"
DRIFT_OUT = OUT_DIR / "scheme_shift_team_drift.parquet"
CHANGE_OUT = OUT_DIR / "scheme_shift_route_change.parquet"
FIT_OUT = OUT_DIR / "scheme_shift_receiver_fit.parquet"
HARMONIZED_OUT = OUT_DIR / "team_route_harmonized.parquet"

# Harmonized route taxonomy that bridges the 2022→2023 schema break.
# Pre-2023 had 12 categories; 2023+ split a few (HITCH/CURL added a
# variant, OUT split into DEEP/QUICK, IN became IN/DIG, etc.). To
# enable 10-year multi-season visualization, we collapse the post-
# 2023 finer categories back to the pre-2023 12-class taxonomy.
ROUTE_TO_HARMONIZED = {
    # Already-canonical (both eras)
    "ANGLE": "ANGLE",
    "CORNER": "CORNER",
    "CROSS": "CROSS",
    "FLAT": "FLAT",
    "GO": "GO",
    "HITCH": "HITCH",
    "IN": "IN",
    "OUT": "OUT",
    "POST": "POST",
    "SCREEN": "SCREEN",
    "SLANT": "SLANT",
    "WHEEL": "WHEEL",
    # Post-2023 finer categories → collapse to pre-2023 12-class
    "HITCH/CURL": "HITCH",
    "IN/DIG": "IN",
    "DEEP OUT": "OUT",
    "QUICK OUT": "OUT",
    "SHALLOW CROSS/DRAG": "CROSS",
    "TEXAS/ANGLE": "ANGLE",
    "SWING": "FLAT",
}

PRIOR_SEASON = 2025  # last completed season
LOOKBACK_SEASON = 2024  # comparison year

# Route taxonomy changed in 2023 (12 → 13 categories: HITCH split
# from HITCH/CURL, OUT split into DEEP OUT + QUICK OUT, etc.).
# Pre-2023 distributions can't be JSD-compared apples-to-apples
# with post-2023 — drift would reflect taxonomy change, not offense
# change. Restrict drift computation to 2023+ where the schema is
# stable.
TAXONOMY_STABLE_SINCE = 2023


def jensen_shannon_div(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric, bounded distance between two probability
    distributions. Range [0, 1] when using log base 2.
    Returns 0 for identical distributions, 1 for fully disjoint."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    # KL divergence with safe log
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def main() -> None:
    print("→ loading team passing fingerprint...")
    fp_full = pd.read_parquet(TEAM_FP)
    fp_full = fp_full[fp_full["dimension"] == "route"].copy()

    # ── Harmonized timeline 2016-2025 (collapses post-2023 finer
    # categories back to the pre-2023 12-class taxonomy) ──────
    print("→ harmonizing route taxonomy across 2016-2025...")
    fp_full["route_unified"] = fp_full["category"].map(
        ROUTE_TO_HARMONIZED).fillna(fp_full["category"])
    # Re-aggregate by harmonized route
    har = (fp_full.groupby(["team", "season", "route_unified"],
                              as_index=False)
                    .agg(count=("count", "sum")))
    # Re-compute shares per (team, season)
    season_total = (har.groupby(["team", "season"])["count"]
                       .sum().rename("season_total").reset_index())
    har = har.merge(season_total, on=["team", "season"])
    har["share"] = har["count"] / har["season_total"].clip(lower=1)
    har = har.rename(columns={"route_unified": "route"})
    har = har[["team", "season", "route", "count", "share"]]
    har.to_parquet(HARMONIZED_OUT, index=False)
    print(f"  ✓ wrote {HARMONIZED_OUT.relative_to(REPO)} "
          f"({len(har):,} rows, {har['season'].min()}–"
          f"{har['season'].max()})")

    # The drift / change / fit calculations below stay in the
    # post-2023 era — within-taxonomy comparisons only.
    fp = fp_full[fp_full["season"] >= TAXONOMY_STABLE_SINCE].copy()
    print(f"  {len(fp):,} (team, season, route) rows "
          f"({TAXONOMY_STABLE_SINCE}+) for drift/fit calcs")

    # Wide: (team, season) → {route: share}
    wide = fp.pivot_table(
        index=["team", "season"], columns="category",
        values="share", fill_value=0.0,
    ).reset_index()

    # ── Output 1: per-team year-over-year drift (JSD) ──────────
    print(f"→ computing year-over-year drift {LOOKBACK_SEASON}→{PRIOR_SEASON}...")
    routes = [c for c in wide.columns if c not in ("team", "season")]
    drift_rows = []
    for team in sorted(wide["team"].unique()):
        sub = wide[wide["team"] == team].sort_values("season")
        for i in range(1, len(sub)):
            prior = sub.iloc[i - 1]
            curr = sub.iloc[i]
            prior_dist = prior[routes].values.astype(float)
            curr_dist = curr[routes].values.astype(float)
            jsd = jensen_shannon_div(prior_dist, curr_dist)
            drift_rows.append({
                "team": team,
                "from_season": int(prior["season"]),
                "to_season": int(curr["season"]),
                "jsd": float(jsd),
            })
    drift = pd.DataFrame(drift_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drift.to_parquet(DRIFT_OUT, index=False)
    print(f"  ✓ wrote {DRIFT_OUT.relative_to(REPO)}")

    # Print top 10 biggest YoY shifts (any season pair)
    print(f"\n  Biggest league-wide YoY route distribution shifts:")
    big = drift.sort_values("jsd", ascending=False).head(10)
    print(big[["team", "from_season", "to_season",
                "jsd"]].to_string(index=False))

    # ── Output 2: per (team, route) share-change in latest year ─
    print(f"\n→ computing per-route share change "
          f"{LOOKBACK_SEASON}→{PRIOR_SEASON}...")
    cur_year = wide[wide["season"] == PRIOR_SEASON].set_index("team")
    prev_year = wide[wide["season"] == LOOKBACK_SEASON].set_index("team")
    teams_both = sorted(set(cur_year.index) & set(prev_year.index))
    change_rows = []
    for team in teams_both:
        for route in routes:
            cur_share = float(cur_year.loc[team, route])
            prev_share = float(prev_year.loc[team, route])
            change_rows.append({
                "team": team,
                "route": route,
                "share_2024": prev_share,
                "share_2025": cur_share,
                "delta": cur_share - prev_share,
            })
    change = pd.DataFrame(change_rows)
    change.to_parquet(CHANGE_OUT, index=False)
    print(f"  ✓ wrote {CHANGE_OUT.relative_to(REPO)} "
          f"({len(change):,} rows)")

    # ── Output 3: per-receiver scheme fit on each team ─────────
    # For each (team, receiver), compute how well the receiver's
    # career route profile aligns with the team's 2025 distribution.
    # Fit = 1 - JSD(receiver_career, team_2025).
    print(f"\n→ computing receiver scheme fit "
          f"vs team {PRIOR_SEASON} distribution...")
    if not PLAYER_RP.exists():
        print("  player_route_profile not built — skipping fit.")
        return

    prp = pd.read_parquet(PLAYER_RP)
    # prp schema: player_id, player_display_name, position, dimension,
    # category, share, targets, ...  (mirrors team fingerprint shape).
    # Filter to route dimension and align column name with the team
    # fingerprint frame.
    prp = prp[prp["dimension"] == "route"].copy()
    prp = prp.rename(columns={"category": "route"})
    # Group by (player, route) → average career share
    career = (prp.groupby(["player_id", "player_display_name",
                              "position", "route"], as_index=False)
                  .agg(career_share=("share", "mean"),
                       career_targets=("targets", "sum")))
    career = career[career["career_targets"] >= 5]

    # Pivot to player × route matrix
    cw = career.pivot_table(
        index=["player_id", "player_display_name", "position"],
        columns="route", values="career_share", fill_value=0.0,
    ).reset_index()
    # Match column set with team routes
    cw_routes = [c for c in cw.columns
                  if c not in ("player_id", "player_display_name",
                                "position")]
    common_routes = [r for r in cw_routes if r in routes]

    # Each player's team(s) in 2025 — use attribution data
    attr = pd.read_parquet(ATTRIBUTION)
    attr_25 = attr[attr["season"] == PRIOR_SEASON]
    rec_team = (attr_25.groupby(
        ["receiver_player_id", "team"], as_index=False)
        .agg(targets=("targets", "sum")))
    rec_team = (rec_team.sort_values("targets", ascending=False)
                          .drop_duplicates("receiver_player_id"))
    rec_team = rec_team.rename(
        columns={"receiver_player_id": "player_id"})

    # Compute fit per receiver against THEIR team's 2025 distribution
    fit_rows = []
    cur_dist_lookup = {
        team: cur_year.loc[team, common_routes].values.astype(float)
        for team in cur_year.index
    }
    for _, prow in cw.iterrows():
        pid = prow["player_id"]
        team_row = rec_team[rec_team["player_id"] == pid]
        if team_row.empty:
            continue
        team = team_row.iloc[0]["team"]
        if team not in cur_dist_lookup:
            continue
        receiver_dist = prow[common_routes].values.astype(float)
        if receiver_dist.sum() < 0.01:
            continue
        team_dist = cur_dist_lookup[team]
        # Fit score = 1 - JSD (so 1.0 = perfect match, 0.0 = disjoint)
        fit = 1.0 - jensen_shannon_div(receiver_dist, team_dist)
        # Also compute fit vs PRIOR year for delta interpretation
        if team in prev_year.index:
            team_dist_prev = prev_year.loc[team, common_routes].values.astype(float)
            fit_prev = 1.0 - jensen_shannon_div(receiver_dist, team_dist_prev)
        else:
            fit_prev = float("nan")
        fit_rows.append({
            "team": team,
            "player_id": pid,
            "player_display_name": prow["player_display_name"],
            "position": prow["position"],
            "fit_2025": float(fit),
            "fit_2024": float(fit_prev),
            "fit_delta": float(fit - fit_prev) if not np.isnan(fit_prev) else float("nan"),
        })
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_parquet(FIT_OUT, index=False)
    print(f"  ✓ wrote {FIT_OUT.relative_to(REPO)} ({len(fit_df):,} rows)")

    # Spot check — biggest "fit improvers" (their team's drift in 2024→2025
    # made them a better fit). These are the scheme-shift winners.
    print(f"\n  Top 15 receivers whose scheme fit IMPROVED most "
          f"(team's offense moved toward their profile):")
    winners = fit_df.dropna(subset=["fit_delta"]).sort_values(
        "fit_delta", ascending=False).head(15)
    print(winners[["team", "player_display_name", "position",
                    "fit_2024", "fit_2025",
                    "fit_delta"]].round(3).to_string(index=False))

    print(f"\n  Top 10 receivers whose scheme fit DEGRADED most "
          f"(team moved away from their profile — sell candidates):")
    losers = fit_df.dropna(subset=["fit_delta"]).sort_values(
        "fit_delta", ascending=True).head(10)
    print(losers[["team", "player_display_name", "position",
                   "fit_2024", "fit_2025",
                   "fit_delta"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

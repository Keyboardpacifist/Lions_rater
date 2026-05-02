"""Build per-team-season scheme tendency table + league-relative deltas.

Output: data/scheme_deltas.parquet

For each (team, season) computes ~20 offensive + defensive scheme
metrics, then subtracts the league-wide season mean to produce a
delta. Deltas are what matter for cross-season comparisons —
"this defense is +3.4 points more man-coverage than league avg" is
more useful than the raw rate, which drifts with league-wide trends.

Metrics:
  Offense
    pace_secs_per_play       — neutral-script tempo
    plays_per_game           — overall volume
    pass_rate_overall        — pass / (pass + rush)
    pass_rate_neutral        — pass rate in neutral situations
    early_down_pass_rate     — 1st & 2nd down pass rate (single best
                                proxy for "pass-heavy scheme")
    shotgun_rate             — % of plays from shotgun
    no_huddle_rate           — % of plays no-huddle
    rz_pass_rate             — pass rate inside opponent 20
    fourth_down_go_rate      — go-for-it rate on 4th down (eligible)
    avg_air_yards            — depth of target
    short_pass_rate          — passes < 10 air yards
    deep_pass_rate           — passes ≥ 20 air yards
    pass_left_rate / mid / right
  Defense
    blitz_rate               — % of dropbacks with ≥5 rushers
    pressure_rate            — % dropbacks with `was_pressure`
    man_coverage_rate        — % from man (when labeled)
    zone_coverage_rate       — % from zone (when labeled)
    box_loaded_rush_rate     — % rush plays with 8+ in box
    epa_per_play_def         — EPA allowed per play
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
OUT = REPO / "data" / "scheme_deltas.parquet"


def _neutral_mask(df: pd.DataFrame) -> pd.Series:
    """Standard nflfastR neutral-script: WP 20-80%, |score_diff| ≤ 7,
    Q1-Q3."""
    home_wp = df.get("home_wp")
    if home_wp is None:
        # Approximate: |score_diff| ≤ 7 and qtr ≤ 3
        return ((df["score_differential"].abs() <= 7)
                & (df["qtr"] <= 3))
    # Compute offense WP: home if posteam==home_team else away
    pos_wp = pd.Series(0.5, index=df.index, dtype=float)
    is_home = df["posteam"] == df["home_team"]
    pos_wp.loc[is_home] = home_wp[is_home]
    pos_wp.loc[~is_home] = df["away_wp"][~is_home]
    return ((pos_wp.between(0.20, 0.80))
            & (df["score_differential"].abs() <= 7)
            & (df["qtr"] <= 3))


def _safe_rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def offensive_metrics(grp: pd.DataFrame) -> dict:
    """Compute offensive scheme metrics for a (team, season) group."""
    plays = grp[grp["play_type"].isin(["pass", "run"])]
    passes = plays[plays["play_type"] == "pass"]
    rushes = plays[plays["play_type"] == "run"]
    n_plays = len(plays)
    n_pass = len(passes)
    n_rush = len(rushes)

    neutral = _neutral_mask(plays)
    plays_neut = plays[neutral]
    passes_neut = plays_neut[plays_neut["play_type"] == "pass"]

    early_downs = plays[plays["down"].isin([1, 2])]
    rz = plays[plays["yardline_100"] <= 20]

    # Pace = average game-clock seconds elapsed between consecutive
    # neutral plays (proxy: plays per game inverted via 60-min/30-sec)
    games = grp["game_id"].nunique() if "game_id" in grp else 1

    # 4th-down go rate: 4th down plays where we ran a play (vs. punt/FG)
    fourth = grp[grp["down"] == 4]
    fourth_eligible = fourth[fourth["play_type"]
                             .isin(["pass", "run", "punt", "field_goal"])]
    fourth_went = fourth_eligible[fourth_eligible["play_type"]
                                  .isin(["pass", "run"])]

    # Air-yards splits (only valid for pass attempts with non-null air_yards)
    pa = passes[passes["air_yards"].notna()]
    short_pass = pa[pa["air_yards"] < 10]
    deep_pass = pa[pa["air_yards"] >= 20]

    return {
        "n_plays": n_plays,
        "plays_per_game": n_plays / games if games else float("nan"),
        "pass_rate_overall": _safe_rate(n_pass, n_plays),
        "pass_rate_neutral": _safe_rate(len(passes_neut), len(plays_neut)),
        "early_down_pass_rate": _safe_rate(
            (early_downs["play_type"] == "pass").sum(), len(early_downs)),
        "shotgun_rate": _safe_rate(plays["shotgun"].fillna(0).sum(), n_plays),
        "no_huddle_rate": _safe_rate(plays["no_huddle"].fillna(0).sum(),
                                       n_plays),
        "rz_pass_rate": _safe_rate(
            (rz["play_type"] == "pass").sum(), len(rz)),
        "fourth_down_go_rate": _safe_rate(len(fourth_went),
                                            len(fourth_eligible)),
        "avg_air_yards": float(pa["air_yards"].mean()) if len(pa) else float("nan"),
        "short_pass_rate": _safe_rate(len(short_pass), len(pa)),
        "deep_pass_rate": _safe_rate(len(deep_pass), len(pa)),
        "pass_left_rate": _safe_rate(
            (passes["pass_location"] == "left").sum(), n_pass),
        "pass_middle_rate": _safe_rate(
            (passes["pass_location"] == "middle").sum(), n_pass),
        "pass_right_rate": _safe_rate(
            (passes["pass_location"] == "right").sum(), n_pass),
        "epa_per_play_off": float(plays["epa"].mean()) if n_plays else float("nan"),
    }


def defensive_metrics(grp: pd.DataFrame) -> dict:
    """Compute defensive scheme metrics for a (defteam, season) group."""
    plays = grp[grp["play_type"].isin(["pass", "run"])]
    n_plays = len(plays)
    dropbacks = plays[plays["play_type"] == "pass"]
    rushes = plays[plays["play_type"] == "run"]

    n_blitz = (dropbacks["number_of_pass_rushers"].fillna(0) >= 5).sum()
    n_pressure = dropbacks.get("was_pressure", pd.Series(dtype=bool)).fillna(0).sum()

    cov = dropbacks.get("defense_man_zone_type", pd.Series(dtype=str))
    n_man  = (cov.astype(str).str.upper() == "MAN_COVERAGE").sum() \
             if not cov.empty else 0
    n_zone = (cov.astype(str).str.upper() == "ZONE_COVERAGE").sum() \
             if not cov.empty else 0
    n_cov_known = n_man + n_zone

    box_loaded = (rushes.get("defenders_in_box", pd.Series(dtype=float))
                  .fillna(0) >= 8).sum()

    return {
        "blitz_rate": _safe_rate(n_blitz, len(dropbacks)),
        "pressure_rate": _safe_rate(n_pressure, len(dropbacks)),
        "man_coverage_rate": _safe_rate(n_man, n_cov_known),
        "zone_coverage_rate": _safe_rate(n_zone, n_cov_known),
        "coverage_label_rate": _safe_rate(n_cov_known, len(dropbacks)),
        "box_loaded_rush_rate": _safe_rate(box_loaded, len(rushes)),
        "epa_per_play_def": float(plays["epa"].mean()) if n_plays else float("nan"),
    }


def main() -> None:
    print("→ loading game_pbp...")
    df = pd.read_parquet(PBP)
    print(f"  rows: {len(df):,}")
    print(f"  seasons: "
          f"{int(df['season'].min())}–{int(df['season'].max())}")

    seasons = sorted(df["season"].dropna().unique())
    rows: list[dict] = []
    for season in seasons:
        sf = df[df["season"] == season]
        # OFFENSE — group by posteam
        for team, grp in sf.groupby("posteam"):
            if not isinstance(team, str) or not team:
                continue
            r = {"team": team, "season": int(season), "side": "offense"}
            r.update(offensive_metrics(grp))
            rows.append(r)
        # DEFENSE — group by defteam
        for team, grp in sf.groupby("defteam"):
            if not isinstance(team, str) or not team:
                continue
            r = {"team": team, "season": int(season), "side": "defense"}
            r.update(defensive_metrics(grp))
            rows.append(r)

    out = pd.DataFrame(rows)
    print(f"  rows produced: {len(out):,}")

    # Compute league-relative deltas: subtract season mean per side
    metric_cols = [c for c in out.columns
                   if c not in ("team", "season", "side", "n_plays",
                                "plays_per_game")]
    for col in metric_cols:
        if out[col].dtype.kind not in "fi":
            continue
        means = out.groupby(["season", "side"])[col].transform("mean")
        out[f"{col}_delta"] = out[col] - means

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== Sample: 2024 offensive deltas, top 8 most pass-heavy "
          "(early-down) ===")
    sample = (out[(out["season"] == 2024) & (out["side"] == "offense")]
              .sort_values("early_down_pass_rate_delta", ascending=False)
              .head(8))
    print(sample[["team", "season", "early_down_pass_rate",
                  "early_down_pass_rate_delta",
                  "no_huddle_rate", "shotgun_rate"]].to_string())


if __name__ == "__main__":
    main()

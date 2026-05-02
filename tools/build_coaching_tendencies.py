"""Build per-team-season coaching/play-caller tendency table.

Output: data/coaching_tendencies.parquet

The scheme-delta table already captures play-style. This focuses on
*coaching decisions* — game-script behavior and aggression — that are
hardest to find elsewhere and most useful for live/in-game props:

  • pass_rate_leading_7p    — pass rate when leading by 7+ (Q1-3)
  • pass_rate_trailing_7p   — pass rate when trailing by 7+ (Q1-3)
  • pass_rate_q4_trailing   — Q4 pass rate when trailing
  • run_rate_q4_leading     — Q4 run rate when leading 7+ (clock-kill)
  • two_min_drill_plays_pg  — plays in final 2 min of either half / game
  • 4th_down_go_rate_short  — 4th-and-≤2 go rate (sharp aggression metric)
  • 4th_down_go_rate_long   — 4th-and-3+ go rate
  • two_pt_attempt_rate     — 2-point try rate after a TD
  • rz_run_rate             — run rate inside opp 20

These are computed for the offense (`posteam`) only — defensive
coaching is harder to capture in pbp without coverage labels we
already use in scheme deltas.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PBP = REPO / "data" / "game_pbp.parquet"
OUT = REPO / "data" / "coaching_tendencies.parquet"


def _safe(num, den):
    return float(num) / float(den) if den else float("nan")


def coaching_metrics(grp: pd.DataFrame) -> dict:
    plays = grp[grp["play_type"].isin(["pass", "run"])]
    n_plays = len(plays)
    games = grp["game_id"].nunique() if "game_id" in grp else 1

    # Game-script splits
    q123 = plays[plays["qtr"] <= 3]
    leading_7 = q123[q123["score_differential"] >= 7]
    trailing_7 = q123[q123["score_differential"] <= -7]
    q4 = plays[plays["qtr"] == 4]
    q4_lead7 = q4[q4["score_differential"] >= 7]
    q4_trail = q4[q4["score_differential"] < 0]

    # 2-min drills (final 2 minutes of either half)
    two_min = plays[
        ((plays["qtr"] == 2) & (plays["quarter_seconds_remaining"] <= 120))
        | ((plays["qtr"] == 4) & (plays["quarter_seconds_remaining"] <= 120))
    ]

    # 4th-down decisions
    fourth = grp[grp["down"] == 4]
    fourth = fourth[fourth["play_type"].isin(["pass", "run", "punt", "field_goal"])]
    short = fourth[fourth["ydstogo"] <= 2]
    long_ = fourth[fourth["ydstogo"] >= 3]
    short_went = short[short["play_type"].isin(["pass", "run"])]
    long_went = long_[long_["play_type"].isin(["pass", "run"])]

    # 2-point attempts (after a TD)
    two_pt = grp[grp["two_point_conv_result"].notna()]
    # All TDs by this team's offense → eligible for an XP/2pt try
    tds = grp[(grp["touchdown"] == 1)
              & (grp["td_team"] == grp["posteam"])]

    # Red-zone runs
    rz = plays[plays["yardline_100"] <= 20]
    rz_runs = rz[rz["play_type"] == "run"]

    return {
        "n_plays": n_plays,
        "games": games,
        "pass_rate_leading_7p":
            _safe((leading_7["play_type"] == "pass").sum(), len(leading_7)),
        "pass_rate_trailing_7p":
            _safe((trailing_7["play_type"] == "pass").sum(), len(trailing_7)),
        "pass_rate_q4_trailing":
            _safe((q4_trail["play_type"] == "pass").sum(), len(q4_trail)),
        "run_rate_q4_leading":
            _safe((q4_lead7["play_type"] == "run").sum(), len(q4_lead7)),
        "two_min_drill_plays_pg":
            _safe(len(two_min), games),
        "fourth_short_go_rate":
            _safe(len(short_went), len(short)),
        "fourth_long_go_rate":
            _safe(len(long_went), len(long_)),
        "two_pt_attempt_rate":
            _safe(len(two_pt), len(tds)),
        "rz_run_rate":
            _safe(len(rz_runs), len(rz)),
    }


def main() -> None:
    print("→ loading pbp...")
    df = pd.read_parquet(PBP)
    print(f"  rows: {len(df):,}")

    seasons = sorted(df["season"].dropna().unique())
    rows: list[dict] = []
    for season in seasons:
        sf = df[df["season"] == season]
        for team, grp in sf.groupby("posteam"):
            if not isinstance(team, str) or not team:
                continue
            r = {"team": team, "season": int(season)}
            r.update(coaching_metrics(grp))
            rows.append(r)

    out = pd.DataFrame(rows)
    print(f"  rows produced: {len(out):,}")

    # League-relative deltas
    metric_cols = [c for c in out.columns
                   if c not in ("team", "season", "n_plays", "games")]
    for col in metric_cols:
        if out[col].dtype.kind not in "fi":
            continue
        means = out.groupby("season")[col].transform("mean")
        out[f"{col}_delta"] = out[col] - means

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== 2024: most aggressive on 4th-and-short ===")
    sample = (out[out["season"] == 2024]
              .sort_values("fourth_short_go_rate", ascending=False)
              .head(8))
    print(sample[["team", "fourth_short_go_rate",
                  "fourth_short_go_rate_delta",
                  "fourth_long_go_rate",
                  "two_pt_attempt_rate"]].to_string())


if __name__ == "__main__":
    main()

"""Books-vs-Model behavioral baseline — Feature 4.3.

Output: data/books_vs_model.parquet

For every historical game where a key starter was on the injury
report (OUT, DOUBTFUL, or QUESTIONABLE), measures the gap between
the closing spread and the actual game margin from that team's POV.
Aggregated by (position, body_part, status) it reveals where books
systematically over- or under-react to injury news.

Methodology
-----------
For each (season, week, team), classify the team's situation:
  • position_lost — QB / RB1 / WR1 / no key loss
  • status        — OUT / DOUBTFUL / QUESTIONABLE / healthy
  • body_part     — knee / hamstring / etc.

Then compute:
  • spread_from_team_pov  — closing spread (negative = favored)
  • actual_margin         — final margin (positive = win)
  • line_miss             — actual_margin - (-spread)
                            (positive ⇒ team beat the close)
  • cover                 — bool: did team cover?
  • total_actual_minus_close — total scored vs. close

If teams missing QB1 systematically post line_miss > 0, the books
OVER-reacted (the team did better than the line implied) → bet
those teams. If line_miss < 0, books UNDER-reacted (book got it
right or under-shot the impact).

Output schema
-------------
position_lost, status, body_part, n_games,
mean_line_miss, cover_rate, mean_total_miss
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCHEDULES = REPO / "data" / "nfl_schedules.parquet"
INJURIES = REPO / "data" / "nfl_injuries_historical.parquet"
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
OUT = REPO / "data" / "books_vs_model.parquet"

# Body-part normalizer reuse
import sys
sys.path.insert(0, str(REPO))
from lib_injury_cohort import body_part_normalize, report_status_code


def _identify_key_players(ps: pd.DataFrame) -> pd.DataFrame:
    """For each (team, season), find the QB1 / RB1 / WR1 by season-long
    workload. Returns a long-format mapping with player_id + role."""
    qb = (ps[ps["position"] == "QB"]
          .groupby(["team", "season", "player_id"], as_index=False)
          ["attempts"].sum())
    qb1 = (qb.sort_values("attempts", ascending=False)
              .drop_duplicates(["team", "season"]))
    qb1["role"] = "QB"

    wr = (ps[ps["position"] == "WR"]
          .groupby(["team", "season", "player_id"], as_index=False)
          ["targets"].sum())
    wr1 = (wr.sort_values("targets", ascending=False)
              .drop_duplicates(["team", "season"]))
    wr1["role"] = "WR1"

    rb = (ps[ps["position"] == "RB"]
          .groupby(["team", "season", "player_id"], as_index=False)
          ["carries"].sum())
    rb1 = (rb.sort_values("carries", ascending=False)
              .drop_duplicates(["team", "season"]))
    rb1["role"] = "RB1"

    keys = pd.concat([
        qb1[["team", "season", "player_id", "role"]],
        wr1[["team", "season", "player_id", "role"]],
        rb1[["team", "season", "player_id", "role"]],
    ], ignore_index=True)
    return keys


def main() -> None:
    print("→ loading schedules + injuries + player stats...")
    sch = pd.read_parquet(SCHEDULES)
    inj = pd.read_parquet(INJURIES)
    ps = pd.read_parquet(PLAYER_STATS)

    # Limit to seasons with reliable spread + injury overlap (2009+)
    sch = sch[sch["season"] >= 2009].copy()
    sch = sch.dropna(subset=["spread_line", "result"])
    print(f"  scheduled games (2009+, with spread + result): "
          f"{len(sch):,}")

    # Build key-player lookup
    keys = _identify_key_players(ps)
    print(f"  key-player rows: {len(keys):,}")

    # Tag each weekly injury row with whether it concerns a key starter
    inj_tagged = inj.merge(
        keys, left_on=["team", "season", "gsis_id"],
        right_on=["team", "season", "player_id"],
        how="inner",
    )
    inj_tagged["status_code"] = inj_tagged["report_status"].apply(
        report_status_code)
    inj_tagged["body_part"] = inj_tagged["report_primary_injury"].apply(
        body_part_normalize)
    # Drop healthy designations — only the impactful ones matter
    inj_tagged = inj_tagged[inj_tagged["status_code"].isin(
        ["OUT", "DOUBTFUL", "QUESTIONABLE"])]
    print(f"  key-player injury rows (Q/D/O only): "
          f"{len(inj_tagged):,}")

    # For each (season, week, team) take the WORST status across roles
    # (e.g., if QB1 is OUT and WR1 is questionable, we treat the team
    # as "QB OUT" — the bigger bet-relevant signal).
    severity = {"OUT": 3, "DOUBTFUL": 2, "QUESTIONABLE": 1}
    inj_tagged["severity"] = inj_tagged["status_code"].map(severity)
    inj_tagged = inj_tagged.sort_values(
        ["season", "week", "team", "severity"], ascending=[True, True, True, False]
    )
    one_per_team = inj_tagged.drop_duplicates(
        subset=["season", "week", "team"], keep="first")[
        ["season", "week", "team", "role", "status_code", "body_part"]
    ]
    one_per_team = one_per_team.rename(columns={"role": "position_lost",
                                                "status_code": "status"})

    # Build per-team-game spread + result records (one row per team,
    # so we can join injuries on the perspective of the affected team).
    home = sch[["season", "week", "home_team", "away_team",
                "home_score", "away_score", "spread_line",
                "total_line", "total"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opp"})
    home["actual_margin"] = home["home_score"] - home["away_score"]
    # spread_line is from home POV (negative = home favored). Flip sign
    # so spread_from_pov > 0 means TEAM is favored.
    home["spread_from_pov"] = -home["spread_line"]
    home["is_home"] = True

    away = sch[["season", "week", "away_team", "home_team",
                "home_score", "away_score", "spread_line",
                "total_line", "total"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opp"})
    away["actual_margin"] = away["away_score"] - away["home_score"]
    away["spread_from_pov"] = away["spread_line"]
    away["is_home"] = False

    games = pd.concat([home, away], ignore_index=True)
    # line_miss = actual_margin - implied_margin (positive = team beat
    # the close). spread_from_pov is the margin the team was expected
    # to win by (positive = favored, negative = dog).
    games["line_miss"] = games["actual_margin"] - games["spread_from_pov"]
    games["covered"] = games["line_miss"] > 0
    games["total_miss"] = games["total"] - games["total_line"]

    # Attach injury context to each team-game
    g_inj = games.merge(one_per_team,
                        on=["season", "week", "team"], how="left")
    g_inj["position_lost"] = g_inj["position_lost"].fillna("HEALTHY")
    g_inj["status"] = g_inj["status"].fillna("HEALTHY")
    g_inj["body_part"] = g_inj["body_part"].fillna("none")

    print(f"  team-games tagged: {len(g_inj):,}")
    print(f"  with injury status: "
          f"{(g_inj['position_lost'] != 'HEALTHY').sum():,}")

    # Aggregate
    grouped = (g_inj.groupby(["position_lost", "status", "body_part"])
               .agg(n_games=("line_miss", "size"),
                    mean_line_miss=("line_miss", "mean"),
                    cover_rate=("covered", "mean"),
                    mean_total_miss=("total_miss", "mean"),
                    median_actual_margin=("actual_margin", "median"),
                    median_spread=("spread_from_pov", "median"))
               .reset_index())

    grouped = grouped.sort_values(["n_games"],
                                  ascending=False).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== Top 12 cohorts by sample size ===")
    print(grouped.head(12).to_string())
    print()
    print("=== When QB1 is OUT — by body part (n>=10) ===")
    qb_out = (grouped[(grouped["position_lost"] == "QB")
                       & (grouped["status"] == "OUT")
                       & (grouped["n_games"] >= 10)]
              .sort_values("mean_line_miss", ascending=False))
    print(qb_out.to_string())


if __name__ == "__main__":
    main()

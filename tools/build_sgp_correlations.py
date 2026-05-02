"""Build same-game-parlay (SGP) correlation table.

Output: data/sgp_correlations.parquet

For each (team, season), identifies the QB1 (most attempts) and top
pass-catchers (WR1/WR2/WR3/TE1 by targets) and lead RB (most carries),
then computes:

  • Pearson correlation between QB passing yards and each partner's
    receiving yards across the games where both played
  • Conditional probability: P(partner ≥ X yds | QB ≥ Y yds) vs the
    unconditional P(partner ≥ X yds) → "lift"

This is the table that exposes where SGP markets are mis-priced.
Sportsbook SGP engines commonly assume independence; positive
correlation means the combined parlay is under-priced.

Schema
------
team, season, qb_name, partner_name, partner_role,
n_games_both, corr_qb_yds_partner_yds, corr_qb_yds_partner_rec,
qb_yds_p_300, partner_yds_p_75, partner_yds_p_75_given_qb_300,
lift_partner_75_given_qb_300
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
OUT = REPO / "data" / "sgp_correlations.parquet"

QB_YARDS_THRESHOLD = 250
PARTNER_YARDS_THRESHOLD = 75
RB_YARDS_THRESHOLD = 75


def _identify_team_roles(team_df: pd.DataFrame) -> dict[str, str]:
    """For one (team, season) df, return {role: player_id}.
    QB1 by attempts; WR1/2/3 by targets among WRs; TE1 by targets;
    RB1 by carries among RBs."""
    roles: dict[str, str] = {}

    qb = (team_df[team_df["position"] == "QB"]
          .groupby("player_id", as_index=False)["attempts"].sum()
          .sort_values("attempts", ascending=False))
    if not qb.empty and qb.iloc[0]["attempts"] > 50:
        roles["QB1"] = qb.iloc[0]["player_id"]

    wrs = (team_df[team_df["position"] == "WR"]
           .groupby("player_id", as_index=False)["targets"].sum()
           .sort_values("targets", ascending=False))
    for i, slot in enumerate(["WR1", "WR2", "WR3"]):
        if len(wrs) > i and wrs.iloc[i]["targets"] >= 25:
            roles[slot] = wrs.iloc[i]["player_id"]

    tes = (team_df[team_df["position"] == "TE"]
           .groupby("player_id", as_index=False)["targets"].sum()
           .sort_values("targets", ascending=False))
    if not tes.empty and tes.iloc[0]["targets"] >= 25:
        roles["TE1"] = tes.iloc[0]["player_id"]

    rbs = (team_df[team_df["position"] == "RB"]
           .groupby("player_id", as_index=False)["carries"].sum()
           .sort_values("carries", ascending=False))
    if not rbs.empty and rbs.iloc[0]["carries"] >= 50:
        roles["RB1"] = rbs.iloc[0]["player_id"]

    return roles


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 4:
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _process_team_season(team_df: pd.DataFrame, team: str, season: int
                         ) -> list[dict]:
    """Compute correlations for one (team, season) cohort."""
    roles = _identify_team_roles(team_df)
    if "QB1" not in roles:
        return []

    qb_id = roles["QB1"]
    qb_games = team_df[team_df["player_id"] == qb_id][[
        "week", "passing_yards", "passing_tds", "completions",
        "player_display_name"
    ]].rename(columns={
        "passing_yards": "qb_yds",
        "passing_tds": "qb_tds",
        "completions": "qb_comp",
        "player_display_name": "qb_name",
    })
    qb_name = qb_games["qb_name"].iloc[0] if not qb_games.empty else "?"

    out: list[dict] = []
    for slot in ["WR1", "WR2", "WR3", "TE1"]:
        if slot not in roles:
            continue
        pid = roles[slot]
        partner_games = team_df[team_df["player_id"] == pid][[
            "week", "receiving_yards", "receptions", "targets",
            "receiving_tds", "player_display_name"
        ]].rename(columns={
            "receiving_yards": "p_yds",
            "receptions": "p_rec",
            "targets": "p_tgt",
            "receiving_tds": "p_tds",
            "player_display_name": "p_name",
        })
        if partner_games.empty:
            continue
        merged = qb_games.merge(partner_games, on="week", how="inner")
        if len(merged) < 4:
            continue

        partner_name = merged["p_name"].iloc[0]
        n = len(merged)
        corr_yds = _safe_corr(merged["qb_yds"], merged["p_yds"])
        corr_rec = _safe_corr(merged["qb_yds"], merged["p_rec"])

        qb_300 = (merged["qb_yds"] >= QB_YARDS_THRESHOLD)
        p_75 = (merged["p_yds"] >= PARTNER_YARDS_THRESHOLD)
        p_yds_75_uncond = float(p_75.mean())
        if qb_300.sum() > 0:
            p_yds_75_given_qb = float(p_75[qb_300].mean())
        else:
            p_yds_75_given_qb = float("nan")
        lift = (p_yds_75_given_qb - p_yds_75_uncond
                if not np.isnan(p_yds_75_given_qb) else float("nan"))

        out.append({
            "team": team, "season": int(season),
            "qb_name": qb_name,
            "partner_name": partner_name,
            "partner_role": slot,
            "n_games_both": n,
            "corr_qb_yds_partner_yds": corr_yds,
            "corr_qb_yds_partner_rec": corr_rec,
            "qb_yds_p_300": float(qb_300.mean()),
            "partner_yds_p_75": p_yds_75_uncond,
            "partner_yds_p_75_given_qb_300": p_yds_75_given_qb,
            "lift_partner_75_given_qb_300": lift,
        })

    # QB ↔ RB1 — passing-yards vs rushing-yards correlation
    if "RB1" in roles:
        rb_id = roles["RB1"]
        rb_games = team_df[team_df["player_id"] == rb_id][[
            "week", "rushing_yards", "carries", "rushing_tds",
            "player_display_name"
        ]].rename(columns={
            "rushing_yards": "p_yds",
            "carries": "p_carries",
            "rushing_tds": "p_tds",
            "player_display_name": "p_name",
        })
        merged = qb_games.merge(rb_games, on="week", how="inner")
        if len(merged) >= 4:
            partner_name = merged["p_name"].iloc[0]
            n = len(merged)
            corr_yds = _safe_corr(merged["qb_yds"], merged["p_yds"])
            qb_300 = (merged["qb_yds"] >= QB_YARDS_THRESHOLD)
            rb_75 = (merged["p_yds"] >= RB_YARDS_THRESHOLD)
            rb_75_uncond = float(rb_75.mean())
            rb_75_given_qb = (float(rb_75[qb_300].mean())
                              if qb_300.sum() > 0 else float("nan"))
            lift = (rb_75_given_qb - rb_75_uncond
                    if not np.isnan(rb_75_given_qb) else float("nan"))
            out.append({
                "team": team, "season": int(season),
                "qb_name": qb_name,
                "partner_name": partner_name,
                "partner_role": "RB1",
                "n_games_both": n,
                "corr_qb_yds_partner_yds": corr_yds,
                "corr_qb_yds_partner_rec": float("nan"),
                "qb_yds_p_300": float(qb_300.mean()),
                "partner_yds_p_75": rb_75_uncond,
                "partner_yds_p_75_given_qb_300": rb_75_given_qb,
                "lift_partner_75_given_qb_300": lift,
            })

    return out


def main() -> None:
    print("→ loading player stats...")
    df = pd.read_parquet(PLAYER_STATS)
    print(f"  rows: {len(df):,}")
    df = df[df["season"] >= 2016].copy()
    print(f"  rows (2016+): {len(df):,}")
    df["attempts"]     = df["attempts"].fillna(0)
    df["targets"]      = df["targets"].fillna(0)
    df["carries"]      = df["carries"].fillna(0)
    df["passing_yards"]   = df["passing_yards"].fillna(0)
    df["receiving_yards"] = df["receiving_yards"].fillna(0)
    df["rushing_yards"]   = df["rushing_yards"].fillna(0)

    rows: list[dict] = []
    for (team, season), grp in df.groupby(["team", "season"]):
        if not isinstance(team, str) or not team:
            continue
        rows.extend(_process_team_season(grp, team, int(season)))

    out = pd.DataFrame(rows)
    out = out.sort_values(["season", "team", "partner_role"]).reset_index(drop=True)

    print(f"  rows produced: {len(out):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()
    print("=== Highest-correlation QB↔WR1 stacks (n>=10, season>=2022) ===")
    sample = (out[(out["partner_role"] == "WR1")
                  & (out["n_games_both"] >= 10)
                  & (out["season"] >= 2022)]
              .sort_values("corr_qb_yds_partner_yds", ascending=False)
              .head(10))
    print(sample[["team", "season", "qb_name", "partner_name",
                  "n_games_both", "corr_qb_yds_partner_yds",
                  "lift_partner_75_given_qb_300"]].to_string())


if __name__ == "__main__":
    main()

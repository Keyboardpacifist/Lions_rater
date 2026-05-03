"""Audit the league-wide Usage Autopsy alpha output.

Goals
-----
1. Validate the projected-absorbed-FP distribution looks sane after the
   QB-tendency leakage + 25% per-player cap.
2. Spot-check the highest-turnover teams (PIT, NYJ, CLE, etc.).
3. Surface anomalies: extreme rows, cap-binding cases, suspicious
   incumbents/arrivals, missing teams, duplicate players, weird names.
4. Confirm no team is silently dropped (32 teams expected).

Run: python tools/audit_league_alpha.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import lib_scoring as fs  # noqa: E402

ATTR = REPO / "data" / "scheme" / "team_route_attribution.parquet"
TRANS = REPO / "data" / "scheme" / "roster_transitions.parquet"
TEAM_QB = REPO / "data" / "scheme" / "team_qb_profile.parquet"
PLAYER_ROUTE = REPO / "data" / "scheme" / "player_route_profile.parquet"


def _route_row_fp(catches, yards, tds, position, config):
    rec_value = config.reception
    if position == "TE" and config.te_premium_bonus > 0:
        rec_value += config.te_premium_bonus
    return ((catches or 0) * rec_value
            + (yards or 0) * config.rec_yard
            + (tds or 0) * config.rec_td)


def compute(config_name: str) -> pd.DataFrame:
    config = fs.CONFIG_BY_NAME[config_name]
    attribution = pd.read_parquet(ATTR)
    transitions = pd.read_parquet(TRANS)
    team_qb = (pd.read_parquet(TEAM_QB)
                  if TEAM_QB.exists() else pd.DataFrame())
    if PLAYER_ROUTE.exists():
        prdf = pd.read_parquet(PLAYER_ROUTE)
        name_lookup = prdf[[
            "player_id", "player_display_name", "position",
        ]].drop_duplicates(subset="player_id")
    else:
        name_lookup = (
            attribution[["receiver_player_id",
                         "player_display_name", "position"]]
            .drop_duplicates(subset="receiver_player_id")
            .rename(columns={"receiver_player_id": "player_id"})
        )

    full_attr = attribution.copy()
    full_attr["row_fp"] = full_attr.apply(
        lambda r: _route_row_fp(
            r.get("catches"), r.get("yards"), r.get("tds"),
            r.get("position", ""), config),
        axis=1,
    )

    PER_PLAYER_CAP_PCT = 0.25

    out_rows = []
    team_diagnostics = []
    for team in transitions["team"].dropna().unique():
        team_trans = transitions[transitions["team"] == team]
        deps_df = team_trans[
            team_trans["transition_type"] == "departure"
        ]
        dep_ids = deps_df["player_id"].dropna().tolist()
        if not dep_ids:
            team_diagnostics.append({
                "team": team, "issue": "no departures",
                "vacated": 0, "redistributable": 0, "cap": 0})
            continue

        team_attr = full_attr[
            (full_attr["team"] == team) & (full_attr["season"] == 2025)
        ]
        if team_attr.empty:
            team_diagnostics.append({
                "team": team, "issue": "no 2025 attribution",
                "vacated": 0, "redistributable": 0, "cap": 0})
            continue

        qb_for_team_local = (
            team_qb[(team_qb["team"] == team)
                       & (team_qb["season"] == 2025)]
            if not team_qb.empty else pd.DataFrame()
        )
        qb_z_map = (
            dict(zip(qb_for_team_local["route"],
                       qb_for_team_local["share_z"]))
            if not qb_for_team_local.empty else {}
        )

        def _keep_factor(route: str) -> float:
            qb_z = qb_z_map.get(route)
            if qb_z is None:
                return 0.7
            raw = 0.4 + 0.3 * float(qb_z)
            return max(0.25, min(1.0, raw))

        vac = (
            team_attr[team_attr["receiver_player_id"].isin(dep_ids)]
            .groupby("route", as_index=False)
            .agg(vacated_fp=("row_fp", "sum"))
        )
        vac = vac[vac["vacated_fp"] > 0].copy()
        if vac.empty:
            team_diagnostics.append({
                "team": team, "issue": "no vacated routes",
                "vacated": 0, "redistributable": 0, "cap": 0})
            continue
        vac["redistributable_fp"] = vac.apply(
            lambda r: r["vacated_fp"] * _keep_factor(r["route"]),
            axis=1,
        )

        team_total_redistributable = float(vac["redistributable_fp"].sum())
        per_player_cap = (team_total_redistributable
                            * PER_PLAYER_CAP_PCT)
        team_diagnostics.append({
            "team": team,
            "issue": "",
            "vacated": vac["vacated_fp"].sum(),
            "redistributable": team_total_redistributable,
            "cap": per_player_cap,
            "n_qb_routes": len(qb_z_map),
        })

        last_year = (
            team_attr.groupby(
                "receiver_player_id", as_index=False)
            .agg(prior=("targets", "sum"))
        )
        last_year = last_year[last_year["prior"] >= 10]
        incumbent_ids = [
            pid for pid in last_year["receiver_player_id"]
            if pid not in dep_ids
        ]
        vet_arr_ids = team_trans[
            (team_trans["transition_type"] == "arrival")
            & (team_trans["is_rookie"] == False)
        ]["player_id"].dropna().tolist()
        candidate_ids = incumbent_ids + vet_arr_ids
        if not candidate_ids:
            continue

        cand_attr = full_attr[
            full_attr["receiver_player_id"].isin(candidate_ids)
        ]
        cand_career = (
            cand_attr.groupby(
                ["receiver_player_id", "route"], as_index=False)
            .agg(career_targets=("targets", "sum"),
                 career_fp=("row_fp", "sum"))
        )
        cand_career["fpt"] = (
            cand_career["career_fp"]
            / cand_career["career_targets"].clip(lower=1)
        )

        for cand_id in candidate_ids:
            relevant = []
            qb_friendly_count = 0
            qb_total_routes = 0
            for _, vrow in vac.iterrows():
                rcands = cand_career[
                    (cand_career["route"] == vrow["route"])
                    & (cand_career["career_targets"] >= 5)
                ].sort_values("fpt", ascending=False).head(3)
                if cand_id not in rcands["receiver_player_id"].values:
                    continue
                this_fpt = float(rcands[
                    rcands["receiver_player_id"] == cand_id
                ]["fpt"].iloc[0])
                top3_total = float(rcands["fpt"].sum() or 1)
                est_absorbed = (
                    vrow["redistributable_fp"]
                    * (this_fpt / top3_total)
                )
                qb_z = qb_z_map.get(vrow["route"])
                if qb_z is not None:
                    qb_total_routes += 1
                    if qb_z >= 0.0:
                        qb_friendly_count += 1
                relevant.append({
                    "route": vrow["route"],
                    "vacated_fp": vrow["vacated_fp"],
                    "est_absorbed": est_absorbed,
                    "qb_z": qb_z,
                    "fpt": this_fpt,
                })
            if not relevant:
                continue

            raw_total = sum(r["est_absorbed"] for r in relevant)
            total_absorbed = min(raw_total, per_player_cap)
            cap_hit = raw_total > per_player_cap
            qb_match_ratio = (
                qb_friendly_count / qb_total_routes
                if qb_total_routes > 0 else 0
            )

            cand_meta = name_lookup[
                name_lookup["player_id"] == cand_id
            ]
            if cand_meta.empty:
                continue
            cname = cand_meta.iloc[0]["player_display_name"]
            cpos = cand_meta.iloc[0]["position"]
            origin = ("Incumbent" if cand_id in incumbent_ids
                        else "New (FA/trade)")

            out_rows.append({
                "team": team,
                "player_id": cand_id,
                "Player": cname,
                "Pos": cpos,
                "Origin": origin,
                "raw_absorbed": round(raw_total, 1),
                "Projected absorbed FP": round(total_absorbed, 1),
                "cap_hit": cap_hit,
                "n_routes": len(relevant),
                "qb_match_pct": round(qb_match_ratio * 100, 0),
            })

    return (pd.DataFrame(out_rows),
            pd.DataFrame(team_diagnostics))


def main() -> None:
    print("=" * 70)
    print("USAGE AUTOPSY AUDIT — League-wide alpha")
    print("=" * 70)
    df, diag = compute("PPR")

    if df.empty:
        print("FATAL: no rows returned")
        return

    # ── Sanity 1: how many teams are represented? ─────────────────────
    teams_in_output = sorted(df["team"].unique())
    print(f"\n[1] Teams with at least one alpha candidate: "
          f"{len(teams_in_output)} / 32")
    if len(teams_in_output) < 32:
        all_teams = set(diag["team"])
        missing = sorted(all_teams - set(teams_in_output))
        print(f"    Missing: {missing}")
        for t in missing:
            row = diag[diag["team"] == t].iloc[0]
            print(f"      {t}: {row['issue']}")

    # ── Sanity 2: distribution of projected absorbed FP ───────────────
    print(f"\n[2] Projected absorbed FP distribution:")
    print(df["Projected absorbed FP"].describe().round(1).to_string())

    # ── Sanity 3: top 25 rows ──────────────────────────────────────────
    print(f"\n[3] TOP 25 league-wide stock-up candidates:")
    cols = ["team", "Player", "Pos", "Origin", "raw_absorbed",
            "Projected absorbed FP", "cap_hit", "n_routes",
            "qb_match_pct"]
    print(df.head(25)[cols].to_string(index=False))

    # ── Sanity 4: rows where cap is binding ───────────────────────────
    capped = df[df["cap_hit"]].sort_values(
        "raw_absorbed", ascending=False)
    print(f"\n[4] Rows where 25% per-player cap kicked in: "
          f"{len(capped)}")
    if not capped.empty:
        print(capped.head(15)[cols].to_string(index=False))

    # ── Sanity 5: highest-turnover teams (top 10 by vacated) ──────────
    print(f"\n[5] HIGHEST-TURNOVER TEAMS (top 10 by vacated FP):")
    top_teams = diag.sort_values(
        "vacated", ascending=False).head(10)
    print(top_teams[["team", "vacated", "redistributable", "cap"]
                       ].round(1).to_string(index=False))

    # ── Sanity 6: per-team #1 absorber for top turnover teams ─────────
    print(f"\n[6] Per-team #1 absorber for top-10 turnover teams:")
    for team in top_teams["team"].head(10):
        team_rows = df[df["team"] == team].sort_values(
            "Projected absorbed FP", ascending=False).head(3)
        if team_rows.empty:
            continue
        print(f"\n  -- {team} --")
        print(team_rows[
            ["Player", "Pos", "Origin", "raw_absorbed",
             "Projected absorbed FP", "cap_hit", "qb_match_pct"]
        ].to_string(index=False))

    # ── Sanity 7: anomalies: duplicate (player, team), missing names ──
    dup = df.groupby(["team", "player_id"]).size()
    dup = dup[dup > 1]
    if not dup.empty:
        print(f"\n[7] DUPLICATE (team, player_id) ROWS: {len(dup)}")
        print(dup.to_string())
    else:
        print(f"\n[7] No duplicate (team, player_id) rows. ✓")

    miss_name = df[df["Player"].isna()
                       | (df["Player"].astype(str).str.strip() == "")]
    if not miss_name.empty:
        print(f"\n    {len(miss_name)} rows with missing player name.")

    # ── Sanity 8: position distribution ───────────────────────────────
    print(f"\n[8] Position breakdown of leaderboard:")
    print(df["Pos"].value_counts().to_string())

    # ── Sanity 9: extreme rows (>40 PPR) ──────────────────────────────
    extreme = df[df["Projected absorbed FP"] > 40].sort_values(
        "Projected absorbed FP", ascending=False)
    print(f"\n[9] Rows projecting > 40 PPR absorbed: "
          f"{len(extreme)}")
    if not extreme.empty:
        print(extreme[cols].to_string(index=False))

    # ── Sanity 10: zero / near-zero absorption rate ───────────────────
    near_zero = df[df["Projected absorbed FP"] < 1]
    print(f"\n[10] Rows projecting <1 PPR (potentially noise): "
          f"{len(near_zero)}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

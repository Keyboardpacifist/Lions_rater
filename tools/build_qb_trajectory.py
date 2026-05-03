"""Per-team QB trajectory score going into the next season.

Output: data/scheme/qb_trajectory.parquet

The Usage Autopsy is vacancy-driven: it surfaces alpha when receivers
leave. But fantasy alpha also comes from QB upgrades:
  - Sophomore (Y2) leap: rookies typically gain ~10–15 GAS in Y2
  - Y3 stabilization: smaller continued gains
  - Injury recovery: a peak QB returning from a missed year
  - Aging decline: vets 35+ projecting modest regression

This builder grades each team's primary QB (from team_qb_profile) on
those axes and emits a single per-team trajectory score the Fantasy
page consumes to rank receivers attached to rising-trajectory QBs.

Schema
------
    team, prior_season, qb_player_id, qb_name,
    n_seasons, last_gas, peak_gas,
    y2_leap_bump, y3_leap_bump, injury_recovery_bump, aging_drag,
    projected_gas, trajectory_delta, trajectory_label, rationale

trajectory_label:
    🚀 RISING     — projected_gas - last_gas >= +5
    ➡️ STABLE     — within +/- 5
    ⬇️ DECLINING  — projected_gas - last_gas <= -3

The deltas are intentionally conservative; this is a *tailwind signal*,
not a point projection. The Fantasy page uses the SIGN and MAGNITUDE
to rank receivers attached to QBs about to play better/worse.
"""
from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
QB_GAS = REPO / "data" / "qb_gas_seasons.parquet"
TEAM_QB = REPO / "data" / "scheme" / "team_qb_profile.parquet"
OUT_DIR = REPO / "data" / "scheme"
OUT = OUT_DIR / "qb_trajectory.parquet"

PRIOR_SEASON = 2025

# Y2 leap magnitudes — empirically rookies who struggle gain the most;
# rookies who already played at a starter level have less room.
Y2_LEAP_LOW = 15.0   # if rookie GAS < 50
Y2_LEAP_MID = 10.0   # if rookie GAS in [50, 60)
Y2_LEAP_HI = 5.0    # if rookie GAS >= 60

# Y3 leap is meaningfully smaller — most growth happens by Y2
Y3_LEAP_LOW = 7.0
Y3_LEAP_MID = 4.0
Y3_LEAP_HI = 2.0

INJURY_GAMES_THRESHOLD = 16  # missed time if total games <16 (regular
                              # season is 17; <16 = missed at least one).
                              # Set high enough to catch Lamar (13) and
                              # Tua (14), low enough to leave Goff/
                              # Prescott/Mayfield at 17 stable.
PEAK_RECOVERY_RATIO = 0.85   # injured QBs return to ~85% of peak

# Aging — using career length as proxy (we don't have birth dates handy)
AGING_VET_YEARS = 12         # 12+ NFL seasons → meaningful age drag
DEEP_VET_YEARS = 14          # 14+ → larger drag


def _y2_leap(last_gas: float) -> float:
    if last_gas < 50:
        return Y2_LEAP_LOW
    if last_gas < 60:
        return Y2_LEAP_MID
    return Y2_LEAP_HI


def _y3_leap(last_gas: float) -> float:
    if last_gas < 50:
        return Y3_LEAP_LOW
    if last_gas < 60:
        return Y3_LEAP_MID
    return Y3_LEAP_HI


def _label(delta: float) -> str:
    if delta >= 5:
        return "🚀 RISING"
    if delta <= -3:
        return "⬇️ DECLINING"
    return "➡️ STABLE"


def main() -> None:
    print("→ loading qb_gas_seasons + team_qb_profile...")
    g = pd.read_parquet(QB_GAS)
    tq = pd.read_parquet(TEAM_QB)

    # ── Pull actual NFL years_exp from rosters ─────────────────────
    # qb_gas_seasons only spans 2016+, so a QB like Rodgers (21 NFL
    # seasons) appears as 9 in the dataset — aging drag would never
    # fire. nflreadpy gives us the authoritative experience count.
    print(f"→ loading nflverse rosters({PRIOR_SEASON}) for QB ages...")
    rost = nfl.load_rosters(PRIOR_SEASON).to_pandas()
    rost = rost[rost["position"] == "QB"].copy()
    rost = rost.dropna(subset=["gsis_id"])
    nfl_years = (
        rost.drop_duplicates(subset="gsis_id")
            .set_index("gsis_id")["years_exp"]
            .astype(float)
            .to_dict()
    )
    print(f"  pulled years_exp for {len(nfl_years):,} QBs")

    # Per-team primary QB for the prior season
    primary = (
        tq[tq["season"] == PRIOR_SEASON]
        .drop_duplicates(["team", "passer_player_id"])
        [["team", "passer_player_id", "passer_player_name",
          "primary_qb_dropbacks"]]
    )
    print(f"  primary QBs for {PRIOR_SEASON}: {len(primary)}")

    rows = []
    for _, prow in primary.iterrows():
        qb_id = prow["passer_player_id"]
        qb_name = prow["passer_player_name"]
        team = prow["team"]
        career = g[g["player_id"] == qb_id].sort_values("season_year")
        if career.empty:
            # No career rows in qb_gas (very rare — perhaps practice-squad
            # call-up that started a couple games). Skip with neutral.
            rows.append({
                "team": team,
                "prior_season": PRIOR_SEASON,
                "qb_player_id": qb_id,
                "qb_name": qb_name,
                "n_seasons": 0,
                "last_gas": float("nan"),
                "peak_gas": float("nan"),
                "y2_leap_bump": 0.0,
                "y3_leap_bump": 0.0,
                "injury_recovery_bump": 0.0,
                "aging_drag": 0.0,
                "projected_gas": float("nan"),
                "trajectory_delta": 0.0,
                "trajectory_label": "❓ UNKNOWN",
                "rationale": "No QB GAS career data on file",
            })
            continue

        last_row = career.iloc[-1]
        last_gas = float(last_row["gas_score"])
        last_games = int(last_row["games"])
        n_seasons = len(career)
        peak_gas = float(career["gas_score"].max())
        # Authoritative NFL career length (Rodgers = 21, Mahomes = 9, etc.)
        nfl_yrs = float(nfl_years.get(qb_id, n_seasons))

        # ── Y2 / Y3 leap ───────────────────────────────────────────
        # Use NFL years here too so a vet returning from a multi-year
        # absence doesn't get falsely tagged as a Y2 leap candidate.
        y2_bump = 0.0
        y3_bump = 0.0
        if nfl_yrs <= 1 and n_seasons == 1:
            y2_bump = _y2_leap(last_gas)
        elif nfl_yrs <= 2 and n_seasons == 2:
            y3_bump = _y3_leap(last_gas)

        # ── Injury recovery ────────────────────────────────────────
        # Trigger ONLY when last season was games-shortened (proxy
        # for injury) AND there's meaningful headroom to a prior peak.
        # Without the games-played gate, we'd false-fire on QBs who
        # simply had a worse healthy season (Goff/Prescott/Purdy had
        # peak gaps but played 17 games — those are regressions to
        # mean, not injury recoveries).
        injury_bump = 0.0
        peak_gap = peak_gas - last_gas
        if (last_games < INJURY_GAMES_THRESHOLD
                and peak_gap >= 5
                and peak_gas > 60):
            injury_bump = peak_gap * PEAK_RECOVERY_RATIO

        # ── Aging drag ─────────────────────────────────────────────
        aging_drag = 0.0
        if nfl_yrs >= DEEP_VET_YEARS:
            aging_drag = -3.0
        elif nfl_yrs >= AGING_VET_YEARS:
            aging_drag = -2.0

        # Aging vet clearly past peak → deepen the drag
        if (nfl_yrs >= AGING_VET_YEARS
                and last_gas < peak_gas - 5):
            aging_drag -= 2.0

        projected_gas = (last_gas + y2_bump + y3_bump
                            + injury_bump + aging_drag)
        delta = projected_gas - last_gas

        # ── Rationale ─────────────────────────────────────────────
        bits = []
        if y2_bump:
            bits.append(f"Y2 leap +{y2_bump:.0f}")
        if y3_bump:
            bits.append(f"Y3 step +{y3_bump:.0f}")
        if injury_bump:
            bits.append(f"injury recovery +{injury_bump:.1f}")
        if aging_drag:
            bits.append(f"aging {aging_drag:.0f}")
        if not bits:
            bits.append("stable trajectory")
        rationale = "; ".join(bits)

        rows.append({
            "team": team,
            "prior_season": PRIOR_SEASON,
            "qb_player_id": qb_id,
            "qb_name": qb_name,
            "n_seasons": n_seasons,
            "nfl_years_exp": nfl_yrs,
            "last_games": last_games,
            "last_gas": round(last_gas, 1),
            "peak_gas": round(peak_gas, 1),
            "y2_leap_bump": round(y2_bump, 1),
            "y3_leap_bump": round(y3_bump, 1),
            "injury_recovery_bump": round(injury_bump, 1),
            "aging_drag": round(aging_drag, 1),
            "projected_gas": round(projected_gas, 1),
            "trajectory_delta": round(delta, 1),
            "trajectory_label": _label(delta),
            "rationale": rationale,
        })

    out = pd.DataFrame(rows).sort_values(
        "trajectory_delta", ascending=False).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")
    print()

    # Spot check — full leaderboard
    print("=== QB TRAJECTORY LEADERBOARD (going into "
          f"{PRIOR_SEASON + 1}) ===")
    cols = ["team", "qb_name", "n_seasons", "last_gas", "peak_gas",
            "projected_gas", "trajectory_delta", "trajectory_label",
            "rationale"]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()

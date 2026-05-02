"""Build empirical cohort play-rate table.

Joins the historical injury archive (Friday snapshot) to actual game-day
snap counts to answer the only question gamblers really care about:

    "Given a player at THIS position with THIS body-part injury and THIS
    Friday designation, what fraction of comparable historical players
    actually took a snap on Sunday?"

Output: data/injury_cohort_rates.parquet

Schema:
    position, body_part_bucket, report_code, practice_code,
    n_cases, n_played, play_rate, snap_share_if_played

Usage retention is captured by `snap_share_if_played` — when a hurt
player did play, what fraction of his usual snap share did he get?
That feeds directly into prop-bet usage adjustments.

Notes
-----
• Snap counts only exist 2013+; the join is restricted to that window.
• Team abbreviation drift is handled (STL→LAR, OAK→LV, SD→LAC, WSH↔WAS).
• "Played" = offense_snaps + defense_snaps + st_snaps > 0.
• Snap share = position-specific (offense_pct for skill, defense_pct
  for defenders, st_pct for kickers/punters).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
INJURIES = REPO / "data" / "nfl_injuries_historical.parquet"
SNAPS = REPO / "data" / "nfl_snap_counts.parquet"
OUT = REPO / "data" / "injury_cohort_rates.parquet"

# Map historical/legacy team abbreviations onto current ones so the join
# survives relocations and the WSH/WAS spelling flip-flop.
TEAM_FIX = {
    "STL": "LA",  "LAR": "LA",
    "OAK": "LV",
    "SD":  "LAC",
    "WSH": "WAS",
}


def _norm_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().replace(TEAM_FIX)


_SUFFIX_RE = r"\s+(Jr\.?|Sr\.?|II|III|IV|V)$"


def _norm_name(s: pd.Series) -> pd.Series:
    """Strip generational suffix and lowercase so 'Marvin Harrison Jr.'
    matches 'Marvin Harrison'. Punctuation/apostrophes are preserved
    since those line up between the two sources."""
    return (s.astype(str)
              .str.replace(_SUFFIX_RE, "", regex=True)
              .str.strip()
              .str.lower())


def _practice_code(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "NONE"
    s = str(raw).upper().strip()
    if "DID NOT" in s or s == "DNP":
        return "DNP"
    if "LIMITED" in s:
        return "LIMITED"
    if "FULL" in s:
        return "FULL"
    if "OUT" in s:
        return "OUT"
    return "NONE"


def _report_code(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "NONE"
    s = str(raw).upper().strip()
    for k in ("OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE"):
        if k in s:
            return k
    return "NONE"


# Reuse the body-part normalizer from lib_injury_cohort so the cohort
# table's buckets always match what the runtime engine queries with.
import sys
sys.path.insert(0, str(REPO))
from lib_injury_cohort import body_part_normalize  # noqa: E402


def _snap_share_for_position(row) -> float:
    """Return the position-appropriate snap share from a snap-counts row."""
    pos = (row.get("position") or "").upper()
    # Offensive skill / OL → offense_pct
    if pos in {"QB", "RB", "FB", "WR", "TE", "C", "G", "T", "OL", "OT", "OG"}:
        return float(row.get("offense_pct") or 0)
    # Defenders → defense_pct
    if pos in {"DE", "DT", "NT", "DL", "EDGE", "OLB", "ILB", "LB", "MLB",
               "CB", "FS", "SS", "S", "DB"}:
        return float(row.get("defense_pct") or 0)
    if pos in {"K", "P", "LS"}:
        return float(row.get("st_pct") or 0)
    # Default: max of the three
    return max(float(row.get("offense_pct") or 0),
               float(row.get("defense_pct") or 0),
               float(row.get("st_pct") or 0))


def main() -> None:
    print("→ loading injuries + snap counts...")
    inj = pd.read_parquet(INJURIES)
    snp = pd.read_parquet(SNAPS)

    # Snap counts begin 2013; restrict join window
    inj = inj[inj["season"] >= 2013].copy()
    print(f"  injuries (2013+): {len(inj):,}")
    print(f"  snap-count rows: {len(snp):,}")

    inj["team_n"]   = _norm_team(inj["team"])
    inj["season_i"] = inj["season"].astype(int)
    inj["week_i"]   = inj["week"].astype(int)
    inj["name_n"]   = _norm_name(inj["full_name"])
    snp["team_n"]   = _norm_team(snp["team"])
    snp["season_i"] = snp["season"].astype(int)
    snp["week_i"]   = snp["week"].astype(int)
    snp["name_n"]   = _norm_name(snp["player"])

    # Join key: (season, week, team, normalized name)
    join = inj.merge(
        snp[["season_i", "week_i", "team_n", "name_n", "player", "position",
             "offense_snaps", "defense_snaps", "st_snaps",
             "offense_pct", "defense_pct", "st_pct"]],
        on=["season_i", "week_i", "team_n", "name_n"],
        how="left",
        suffixes=("", "_snp"),
    )
    matched = join["player"].notna().sum()
    print(f"  matched: {matched:,} / {len(join):,} "
          f"({matched / max(len(join), 1):.0%})")

    # Compute outcome columns
    snaps_total = (join["offense_snaps"].fillna(0)
                   + join["defense_snaps"].fillna(0)
                   + join["st_snaps"].fillna(0))
    join["played"] = (snaps_total > 0).astype(int)
    # If we had no snap-count match (player wasn't on the gameday roster),
    # treat as did_not_play. That's consistent with the gambler's question.
    join.loc[join["player"].isna(), "played"] = 0
    join["snap_share"] = join.apply(_snap_share_for_position, axis=1)

    # Cohort key columns
    join["body_part"]   = join["report_primary_injury"].apply(body_part_normalize)
    join["report_code"] = join["report_status"].apply(_report_code)
    join["practice_code"] = join["practice_status"].apply(_practice_code)
    join["pos_clean"] = join["position"].astype(str).str.upper().str.strip()

    # Aggregate by cohort
    grouped = (join
               .groupby(["pos_clean", "body_part", "report_code",
                         "practice_code"], dropna=False)
               .agg(n_cases=("played", "size"),
                    n_played=("played", "sum"),
                    snap_share_sum=("snap_share", "sum"))
               .reset_index())
    grouped["play_rate"] = grouped["n_played"] / grouped["n_cases"].clip(lower=1)
    # snap_share_if_played: avg snap share among those who DID play.
    # Build this with a second groupby on the played-only subset.
    played_only = join[join["played"] == 1]
    played_grp = (played_only
                  .groupby(["pos_clean", "body_part", "report_code",
                            "practice_code"], dropna=False)["snap_share"]
                  .mean()
                  .reset_index()
                  .rename(columns={"snap_share": "snap_share_if_played"}))
    grouped = grouped.merge(
        played_grp,
        on=["pos_clean", "body_part", "report_code", "practice_code"],
        how="left",
    )
    grouped = grouped.drop(columns=["snap_share_sum"])
    grouped = grouped.rename(columns={"pos_clean": "position"})

    # Sort by sample size descending so head() is the most-trusted cohorts
    grouped = grouped.sort_values("n_cases", ascending=False).reset_index(drop=True)

    print(f"  cohorts: {len(grouped):,}")
    print(f"  median n: {int(grouped['n_cases'].median()):,}")
    print(f"  cohorts with n>=30: "
          f"{int((grouped['n_cases'] >= 30).sum()):,}")
    print(f"  top-10 cohorts:")
    print(grouped.head(10).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(OUT, index=False)
    print(f"  ✓ wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

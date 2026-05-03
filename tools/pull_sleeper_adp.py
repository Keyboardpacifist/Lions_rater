"""Pull Sleeper public ADP / search-rank data and map to gsis_id.

Output: data/fantasy/sleeper_adp.parquet

Sleeper's free public API (https://api.sleeper.app/v1/players/nfl)
returns a player metadata blob for ~12K NFL players. Each entry has
a `search_rank` field that's effectively the platform-wide ADP signal
(lower = drafted earlier). It's not split by scoring format, so for
v1 we use it as the overall ADP proxy. Position-specific rank is
derived by filtering and re-ranking within position.

Player ID crosswalk: Sleeper uses its own internal player_id. We map
to gsis_id (00-XXXXXXX) by joining on name + position via
nfl_player_stats_weekly.parquet, which has both gsis_id and
player_display_name covering every active NFL player.

Notes
-----
- Run weekly during draft season; the rank changes daily.
- For format-specific ADP (PPR vs Standard vs Best Ball), we'd
  need a different source — Sleeper's search_rank is consensus.
- Free, no auth required.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


REPO = Path(__file__).resolve().parent.parent
PLAYER_STATS = REPO / "data" / "nfl_player_stats_weekly.parquet"
OUT_DIR = REPO / "data" / "fantasy"
OUT_PATH = OUT_DIR / "sleeper_adp.parquet"

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"

# Sleeper uses some legacy team abbreviations that don't match nflverse.
# Normalize to nflverse style so downstream joins (attribution,
# transitions, scheme fingerprint) all see consistent codes.
SLEEPER_TO_NFLVERSE_TEAM = {
    "LAR": "LA",     # Rams: Sleeper LAR → nflverse LA
    "OAK": "LV",     # Raiders: Sleeper still uses legacy OAK
    # Add more here if other mismatches surface (BAL/BLT, ARI/ARZ, etc.)
}


def _normalize_name(s: pd.Series) -> pd.Series:
    """Lower + strip punctuation/spaces + suffix tokens for fuzzy join.

    Handles cases where one source has a suffix and the other doesn't:
      "Luther Burden III" (nflverse) ↔ "Luther Burden" (Sleeper)
      "Marvin Harrison Jr." (nflverse) ↔ "Marvin Harrison" (Sleeper)
    Both normalize to the same key.
    """
    out = s.fillna("").astype(str).str.lower()
    # Strip suffix tokens (jr, sr, ii, iii, iv, v) at word boundaries,
    # optionally with trailing period
    out = out.str.replace(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", regex=True)
    # Strip dots, spaces, hyphens, apostrophes
    out = out.str.replace(r"[\.\s\-']", "", regex=True)
    return out


def main() -> None:
    print("→ pulling Sleeper player metadata...")
    r = requests.get(SLEEPER_PLAYERS_URL, timeout=30)
    r.raise_for_status()
    raw = r.json()
    print(f"  total players: {len(raw):,}")

    rows = []
    for sleeper_id, p in raw.items():
        if not p.get("active"):
            continue
        if not p.get("full_name"):
            continue
        team = p.get("team")
        # Normalize Sleeper team codes to nflverse style
        if team in SLEEPER_TO_NFLVERSE_TEAM:
            team = SLEEPER_TO_NFLVERSE_TEAM[team]
        rows.append({
            "sleeper_id":   sleeper_id,
            "full_name":    p.get("full_name"),
            "position":     p.get("position"),
            "team":         team,
            "years_exp":    p.get("years_exp"),
            "search_rank":  p.get("search_rank"),
            "injury_status": p.get("injury_status"),
            "depth_chart_position": p.get("depth_chart_position"),
        })
    sleeper = pd.DataFrame(rows)
    print(f"  active w/ name: {len(sleeper):,}")

    # ── Map to our gsis_id via name match ──────────────────────────
    print("→ loading nfl_player_stats_weekly for gsis_id crosswalk...")
    pw = pd.read_parquet(PLAYER_STATS,
                            columns=["player_id", "player_display_name",
                                       "position", "season"])
    pw = pw.dropna(subset=["player_id", "player_display_name"])

    # Track each player's latest active NFL season — used to break
    # ties when two players share a normalized name+position
    # (e.g., Marvin Harrison Sr. + Marvin Harrison Jr.).
    latest_season = (
        pw.groupby("player_id")["season"].max()
          .rename("latest_season").reset_index()
    )
    pw = (pw.drop_duplicates(subset=["player_id"])
            .merge(latest_season, on="player_id", how="left"))
    print(f"  player crosswalk rows: {len(pw):,}")

    # Normalize names on both sides
    sleeper["nname"] = _normalize_name(sleeper["full_name"])
    pw["nname"] = _normalize_name(pw["player_display_name"])

    # When multiple players share a normalized name+position, keep
    # the MOST RECENT. Without this, "Marvin Harrison" would map to
    # both Sr. (1996-2008) and Jr. (2024+) → duplicate Sleeper rows.
    pw_unique = (
        pw.sort_values("latest_season", ascending=False)
          .drop_duplicates(subset=["nname", "position"])
    )

    # Join on (normalized_name, position) — exactly one match per
    # (sleeper player, position) thanks to the dedupe above.
    merged = sleeper.merge(
        pw_unique[["player_id", "nname", "position"]],
        on=["nname", "position"], how="left",
    )

    matched = merged["player_id"].notna().sum()
    total = len(merged)
    print(f"  name-matched: {matched:,} / {total:,} "
          f"({matched/total:.0%})")

    # ── Position-specific rank derived from overall search_rank ─────
    has_rank = merged.dropna(subset=["search_rank"]).copy()
    has_rank["search_rank"] = has_rank["search_rank"].astype(float)
    has_rank["pos_rank"] = (
        has_rank.groupby("position")["search_rank"]
                .rank(method="min")
    )
    has_rank["overall_rank"] = (
        has_rank["search_rank"].rank(method="min")
    )

    print(f"  with rank: {len(has_rank):,}")

    # Output
    out = has_rank[[
        "player_id", "sleeper_id", "full_name", "position", "team",
        "years_exp", "depth_chart_position", "injury_status",
        "search_rank", "overall_rank", "pos_rank",
    ]].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"  ✓ wrote {OUT_PATH.relative_to(REPO)}")
    print()

    # Spot checks
    print("=== Top 30 overall ===")
    print(out.nsmallest(30, "search_rank")[
        ["full_name", "position", "team", "overall_rank",
         "pos_rank", "search_rank"]
    ].to_string(index=False))
    print()
    print("=== Top 12 TE ===")
    tes = out[out["position"] == "TE"].nsmallest(12, "search_rank")
    print(tes[["full_name", "team", "overall_rank", "pos_rank"]
              ].to_string(index=False))


if __name__ == "__main__":
    main()

"""
Target counts for college receivers — derived from CFBD playText.

CFBD's aggregate /stats/player/season endpoint exposes receptions but
NOT targets. To recover targets we have to:
  1. Pull /plays for every (season, week)
  2. Regex-extract the targeted receiver's "F.LastName" from the playText
  3. Disambiguate against the team's roster (full names) for that season
  4. Aggregate per (team, player_id, season)

Limitation: when a team has two players sharing the same first initial +
last name (e.g., Ohio State 2024 had Jayden Smith AND Jeremiah Smith,
both rendered "J.Smith" in playText), we record the target under the
"ambiguous" bucket rather than guessing. Affected players will have
slightly under-counted targets — flagged in the output column
`targets_ambiguous_skipped`.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import client

# ── Regexes for finding the targeted receiver in playText ─────────
#
# Standard CFBD playText (most games) uses FULL names:
#   Pass Reception: "Drew Allar pass complete to Harrison Wallace III for 12 yds to the..."
#   Passing TD:     "Drew Allar pass complete to Jaden Greathouse for 54 yds for a TD ..."
#   Pass Incompletion: "Drew Allar pass incomplete to Omari Evans" (sometimes + ", broken up by ...")
#
# A small minority (NFL-style writeups, e.g., 2024 Iowa games) instead use:
#   "C.McNamara pass complete. Catch made by R.Vander Zee for 11 yards. ..."
#
# We capture both styles. Receiver names can include suffixes ("III"),
# punctuation ("D'Andre"), hyphens ("Al-Jay"), and spaces ("Vander Zee",
# "St. John") — so we capture greedily up to the next " for " / "," /
# end-of-string. The captured name is then matched against rosters' full
# names with a normalize step.

# "pass complete to <name> for ..." — standard format, captures everything
# between "to " and " for"
PASS_COMPLETE_TO_RE = re.compile(
    r"pass complete to\s+(.+?)\s+for\b",
    re.IGNORECASE,
)
# "pass incomplete to <name>[,. ]" — standard format
PASS_INCOMPLETE_TO_RE = re.compile(
    r"pass incomplete to\s+(.+?)(?:[\,\.]|$|\s+broken up|\s+intended)",
    re.IGNORECASE,
)
# Legacy / NFL-style: "Catch made by <name> for ..."
PASS_COMPLETE_CATCH_RE = re.compile(
    r"Catch made by\s+(.+?)\s+for\b",
    re.IGNORECASE,
)
# Legacy / NFL-style: "Pass incomplete intended for <name>"
PASS_INCOMPLETE_INTENDED_RE = re.compile(
    r"Pass(?: incomplete)?\s+intended for\s+(.+?)(?:[\.\,\;]|$)",
    re.IGNORECASE,
)

PASS_PLAY_TYPES = {"Pass Reception", "Passing Touchdown"}
INCOMPLETE_PLAY_TYPES = {"Pass Incompletion"}


def _normalize_name(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. For roster matching."""
    s = s.lower().strip()
    s = re.sub(r"[\.\'\,]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _short_name(first_name: str, last_name: str) -> str:
    """Convert ('Will', 'Rogers') -> 'w.rogers' (normalized)."""
    if not first_name or not last_name:
        return ""
    return f"{first_name[0].lower()}.{_normalize_name(last_name)}"


def build_roster_lookup(year: int, verbose: bool = False) -> dict:
    """For a season, return TWO lookup dicts merged:
        - keyed by (team, normalized_full_name) — primary, for standard playText
        - keyed by (team, 'f.lastname') — fallback, for legacy "Catch made by F.Last" style
    Each value is a list of (player_id, display_full_name). Multiple entries = ambiguous.
    """
    roster = client.roster(year, verbose=verbose)
    out: dict = defaultdict(list)
    for r in roster:
        team = r.get("team")
        fn = r.get("firstName") or ""
        ln = r.get("lastName") or ""
        pid = str(r.get("id") or "")
        if not team or not ln or not pid:
            continue
        full = f"{fn} {ln}".strip()
        full_norm = _normalize_name(full)
        out[(team, full_norm)].append((pid, full))
        short = _short_name(fn, ln)
        if short:
            out[(team, short)].append((pid, full))
    if verbose:
        n_unique = sum(1 for v in out.values() if len(v) == 1)
        n_ambig = sum(1 for v in out.values() if len(v) > 1)
        print(f"  Roster lookup ({year}): {n_unique} unique, {n_ambig} ambiguous keys")
    return out


def parse_target(play: dict) -> tuple[str, bool] | None:
    """If this play is a pass attempt with an extractable intended
    receiver, return (receiver_text, was_caught). Otherwise None.
    The receiver_text is the raw capture (full name OR F.LastName);
    the caller normalizes + matches against roster.
    """
    pt = play.get("playType") or ""
    text = play.get("playText") or ""
    if not text:
        return None

    if pt in PASS_PLAY_TYPES:
        # Try standard format first, then legacy
        m = PASS_COMPLETE_TO_RE.search(text)
        if not m:
            m = PASS_COMPLETE_CATCH_RE.search(text)
        if m:
            return (m.group(1).strip(), True)
    if pt in INCOMPLETE_PLAY_TYPES:
        m = PASS_INCOMPLETE_TO_RE.search(text)
        if not m:
            m = PASS_INCOMPLETE_INTENDED_RE.search(text)
        if m:
            return (m.group(1).strip(), False)
    return None


def _match_receiver(team: str, receiver_text: str, lookup: dict) -> tuple[str, str] | None | str:
    """Match a captured receiver text to a roster entry.
    Returns (player_id, full_name) on unique match,
    'AMBIGUOUS' if multiple roster entries share the key,
    None if no match found.
    """
    norm = _normalize_name(receiver_text)
    candidates = lookup.get((team, norm), [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return "AMBIGUOUS"
    return None


def aggregate_targets_for_season(year: int, verbose: bool = True) -> pd.DataFrame:
    """Pull /plays for every week of the season, parse targets, and
    aggregate per (team, player_id) using the season roster as the
    disambiguation source."""
    if verbose:
        print(f"\n=== Targets {year} ===")
    roster_lookup = build_roster_lookup(year, verbose=verbose)

    # CFBD weeks: regular season = 1-15, postseason = 16+. We want all.
    # Empty weeks return [], so just iterate through 1..20 and bail when
    # consecutive weeks are empty.
    target_counts: dict = defaultdict(lambda: {"targets": 0, "receptions": 0})
    ambiguous_skips = defaultdict(int)
    unmatched = defaultdict(int)
    total_pass_plays = 0
    consecutive_empty = 0

    for week in range(1, 21):
        plays = client.plays(year, week, verbose=verbose)
        if not plays:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                break
            continue
        consecutive_empty = 0

        for play in plays:
            parsed = parse_target(play)
            if not parsed:
                continue
            receiver_text, was_caught = parsed
            total_pass_plays += 1
            offense = play.get("offense")
            if not offense:
                continue
            match = _match_receiver(offense, receiver_text, roster_lookup)
            if match is None:
                unmatched[(offense, receiver_text)] += 1
                continue
            if match == "AMBIGUOUS":
                ambiguous_skips[(offense, receiver_text)] += 1
                continue
            pid, full = match
            target_counts[(offense, pid, full)]["targets"] += 1
            if was_caught:
                target_counts[(offense, pid, full)]["receptions"] += 1

    if verbose:
        print(f"  Total parsed pass plays: {total_pass_plays}")
        print(f"  Unique receivers identified: {len(target_counts)}")
        print(f"  Ambiguous (skipped) keys: {len(ambiguous_skips)}, plays: {sum(ambiguous_skips.values())}")
        print(f"  Unmatched (no roster entry): {len(unmatched)} keys, {sum(unmatched.values())} plays")

    rows = []
    for (team, pid, full), stats in target_counts.items():
        rows.append({
            "season": year,
            "team": team,
            "player_id": str(pid),
            "player": full,
            "targets_pbp": stats["targets"],
            "receptions_pbp": stats["receptions"],
            "catch_rate_pbp": (stats["receptions"] / stats["targets"]) if stats["targets"] > 0 else None,
        })
    return pd.DataFrame(rows)


def merge_targets_into_advanced(year: int, verbose: bool = True) -> None:
    """Pull targets for `year`, then merge into each existing
    college_<pos>_cfbd_advanced.parquet under the matching season."""
    targets_df = aggregate_targets_for_season(year, verbose=verbose)
    if targets_df.empty:
        if verbose:
            print(f"  No targets data for {year}")
        return

    out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "college"
    for pos in ["wr", "te", "rb", "qb"]:
        path = out_dir / f"college_{pos}_cfbd_advanced.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        # Drop existing target cols for this season (so re-runs work cleanly)
        target_cols = ["targets_pbp", "receptions_pbp", "catch_rate_pbp"]
        for col in target_cols:
            if col in df.columns:
                df.loc[df["season"] == year, col] = pd.NA
        # Merge in
        merged = df.merge(
            targets_df[["season", "team", "player_id"] + target_cols],
            on=["season", "team", "player_id"],
            how="left",
            suffixes=("", "_new"),
        )
        # Where _new exists, prefer it (for the year we just pulled)
        for col in target_cols:
            new_col = col + "_new"
            if new_col in merged.columns:
                merged[col] = merged[new_col].fillna(merged[col])
                merged = merged.drop(columns=[new_col])
        merged.to_parquet(path, index=False)
        if verbose:
            n_with = merged.loc[merged["season"] == year, "targets_pbp"].notna().sum()
            print(f"    Updated {path.name}: {n_with} rows now have targets for {year}")


def backfill(years: list[int], verbose: bool = True) -> None:
    for y in years:
        merge_targets_into_advanced(y, verbose=verbose)

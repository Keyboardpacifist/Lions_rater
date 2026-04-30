#!/usr/bin/env python3
"""Scrape track & field marks + HS combine times for 2027 Draft prospects
via Serper.dev's Google search API.

Reads each prospect's name + state from our parquets, runs a handful
of targeted queries per prospect (100m, 200m, long jump, high jump,
triple jump, 40-yard dash), parses snippets for time/distance
patterns, and saves best-effort matches to data/draft_track_marks.parquet.
Raw snippet evidence stays in the output so the user can verify
ambiguous matches.

Setup:
  SERPER_API_KEY — from https://serper.dev dashboard

Set as env var OR pulled from .streamlit/secrets.toml.
Free tier: 2,500 queries on signup — covers our entire 2,400-query
run at $0.

Usage:
  python tools/scrape_track_marks.py             # all 400 prospects
  python tools/scrape_track_marks.py --top 50    # top 50 only
  python tools/scrape_track_marks.py --top 3 --verbose   # smoke test
  python tools/scrape_track_marks.py --resume    # skip prospects
                                                  # already scraped
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
CONSENSUS = REPO / "data" / "draft_2027_consensus.parquet"
RECRUITING = REPO / "data" / "college" / "college_recruiting.parquet"
OUT = REPO / "data" / "draft_track_marks.parquet"
SECRETS = REPO / ".streamlit" / "secrets.toml"

SERPER_ENDPOINT = "https://google.serper.dev/search"


def _load_credentials() -> str | None:
    api_key = os.environ.get("SERPER_API_KEY")
    if api_key:
        return api_key
    if SECRETS.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # py<3.11
        with open(SECRETS, "rb") as f:
            cfg = tomllib.load(f)
        api_key = cfg.get("SERPER_API_KEY")
    return api_key


# ── Pattern matchers ────────────────────────────────────────────
# Snippets are noisy — we have to require the event label adjacent
# to the number. The number BEFORE the label and the number AFTER
# the label are both common phrasings:
#   "ran 10.84 in the 100 meter"   — number before
#   "100 Meter Dash, 11.46"        — number after
#   "40 Yard Dash, 4.2 sec"        — number after
# Window: 30 chars on each side of the event label.

def _looks_doubled(s: str) -> bool:
    """Reject suspicious digit-doubling patterns like '11.11' or
    '22.22' that show up in tabular junk text from athletic.net."""
    digits = s.replace(".", "")
    return len(set(digits)) == 1


def _name_in_window(text: str, match_start: int, match_end: int,
                       name: str, window: int = 80) -> bool:
    """Check that the player's name (or last-name only) appears within
    `window` chars on either side of the captured match. Catches the
    'wrong athlete in same list' false-positive."""
    lo = max(0, match_start - window)
    hi = min(len(text), match_end + window)
    chunk = text[lo:hi].lower()
    n = name.lower()
    if n in chunk:
        return True
    last = name.split()[-1].lower()
    if len(last) >= 4 and last in chunk:
        return True
    return False


def _scan_event_time(text: str, name: str, label_re: str,
                        low: float, high: float,
                        digit_pat: str = r"\d{1,2}\.\d{1,2}") -> float | None:
    """Find a time near the event label AND adjacent to the player's
    name. Tries number-before-label first, then number-after-label."""
    after = re.compile(
        rf"(?:{label_re})\W{{0,12}}(?:dash|run)?\W{{0,8}}({digit_pat})",
        re.IGNORECASE,
    )
    before = re.compile(
        rf"({digit_pat})\W{{0,8}}(?:sec|s)?\W{{0,16}}(?:in\s+the\s+|at\s+the\s+)?(?:{label_re})",
        re.IGNORECASE,
    )
    for pat in (after, before):
        for m in pat.finditer(text):
            raw = m.group(1)
            if _looks_doubled(raw):
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            if not (low <= v <= high):
                continue
            if not _name_in_window(text, m.start(), m.end(), name):
                continue
            return v
    return None


_LABEL_100M = r"100\s*(?:m\b|meter(?:s)?|m\s*(?:dash|race)?\b)"
_LABEL_200M = r"200\s*(?:m\b|meter(?:s)?|m\s*(?:dash|race)?\b)"
_LABEL_40   = r"40[\s-]?(?:yard|yd|y\b)\s*(?:dash)?"

# Jumps: "23'5\"", "23-5", "23.5 ft" — all in proximity to event label.
_JUMP_FTIN = r"(\d{1,2})['’]\s*(\d{0,2})(?:[\"”]|\s*(?:in|inches))?"
_JUMP_DEC  = r"(\d{1,2}\.\d{1,2})\s*(?:ft|feet)"


def _scan_event_jump(text: str, name: str, label_re: str,
                        low_in: int, high_in: int) -> float | None:
    """Find a jump distance near the event label AND adjacent to the
    player's name. Returns inches."""
    pat_after = re.compile(
        rf"(?:{label_re})\W{{0,12}}(?:cleared|jump|of)?\W{{0,8}}"
        rf"(?:{_JUMP_FTIN}|{_JUMP_DEC})",
        re.IGNORECASE,
    )
    pat_before = re.compile(
        rf"(?:{_JUMP_FTIN}|{_JUMP_DEC})\W{{0,16}}(?:in\s+the\s+|at\s+the\s+)?"
        rf"(?:{label_re})",
        re.IGNORECASE,
    )
    for pat in (pat_after, pat_before):
        for m in pat.finditer(text):
            g = m.groups()
            inches = None
            if g[0] and g[0].strip().isdigit():
                ft = int(g[0])
                inch = int(g[1]) if g[1] and g[1].strip() else 0
                if 0 <= inch < 12 or g[1] == "":
                    inches = ft * 12 + (inch if inch < 12 else 0)
            elif g[2]:
                try:
                    inches = float(g[2]) * 12
                except ValueError:
                    pass
            if inches is None:
                continue
            if not (low_in <= inches <= high_in):
                continue
            if not _name_in_window(text, m.start(), m.end(), name):
                continue
            return float(inches)
    return None


_LABEL_LJ = r"long\s*jump|\bLJ\b"
_LABEL_HJ = r"high\s*jump|\bHJ\b"
_LABEL_TJ = r"triple\s*jump|\bTJ\b"
_LABEL_VERT = r"vertical(?:\s*jump)?"
_LABEL_BROAD = r"broad(?:\s*jump)?|standing\s*broad"


def _build_state_lookup() -> dict:
    """player_name → {state, school} from recruiting parquet for the
    most recent entry per name (handles transfer-portal case)."""
    if not RECRUITING.exists():
        return {}
    df = pd.read_parquet(RECRUITING)
    df = df.dropna(subset=["name"]).copy()
    if "recruit_year" in df.columns:
        df = df.sort_values("recruit_year", ascending=False)
    df = df.drop_duplicates(subset=["name"], keep="first")
    out = {}
    for _, r in df.iterrows():
        out[str(r["name"])] = {
            "state": r.get("state"),
            "city": r.get("city"),
            "hs_school": r.get("school"),
        }
    return out


def serper_search(query: str, api_key: str,
                       num: int = 5) -> list[dict]:
    """Hit Serper.dev API and return result items normalized to
    {title, snippet, link}. Returns empty list on errors."""
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": min(num, 10)}
    try:
        resp = requests.post(SERPER_ENDPOINT, headers=headers,
                                json=payload, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        # Serper returns 'organic' for the main results list
        return data.get("organic", []) or []
    except Exception:
        return []


def _is_likely_match(snippet_text: str, name: str,
                          state: str | None, hs_city: str | None,
                          hs_school: str | None,
                          college_school: str | None) -> bool:
    """Strong identity check: require an actual disambiguator beyond
    the player's name. State alone is too coarse (multiple players
    named 'Jeremiah Smith' from FL exist). Accept the snippet only
    if it references the player's HS school, HS city, or — most
    useful for current college prospects — their COLLEGE school."""
    s = snippet_text.lower()
    if name.lower() not in s:
        return False
    if hs_school and len(hs_school) >= 4 and hs_school.lower() in s:
        return True
    if hs_city and len(hs_city) >= 4 and hs_city.lower() in s:
        return True
    if college_school and len(college_school) >= 3:
        cs = college_school.lower()
        # Allow nicknames: Ole Miss = Mississippi, etc.
        if cs in s:
            return True
        if cs == "mississippi" and "ole miss" in s:
            return True
        if cs == "ole miss" and "mississippi" in s:
            return True
    return False


def scrape_prospect(name: str, state: str | None,
                       hs_city: str | None, hs_school: str | None,
                       college_school: str | None,
                       api_key: str) -> dict:
    """Run targeted queries for one prospect; parse snippets for marks
    only when the surrounding text confirms the right athlete (via
    HS city / HS school / college school match)."""
    state_str = f' "{state}"' if state else ""

    queries = {
        "time_100m":      f'"{name}"{state_str} "100 meter" track',
        "time_200m":      f'"{name}"{state_str} "200 meter" track',
        "long_jump_in":   f'"{name}"{state_str} "long jump"',
        "high_jump_in":   f'"{name}"{state_str} "high jump"',
        "triple_jump_in": f'"{name}"{state_str} "triple jump"',
        "forty_time":     f'"{name}"{state_str} "40 yard dash"',
        "vertical_in":    f'"{name}"{state_str} "vertical jump"',
        "broad_jump_in":  f'"{name}"{state_str} "broad jump"',
    }

    found = {}
    evidence = {}
    for field, q in queries.items():
        items = serper_search(q, api_key, num=5)
        if not items:
            continue
        # Verify each result individually + scan only the SNIPPET body
        # (not title) so we don't trust a match whose only Smith
        # reference is in the headline of an article that talks about
        # a different player by paragraph 2.
        v = None
        winning_item = None
        for it in items:
            full = f"{it.get('title', '')} {it.get('snippet', '')}"
            if not _is_likely_match(full, name, state, hs_city,
                                       hs_school, college_school):
                continue
            snippet = it.get("snippet", "") or ""
            if name.lower() not in snippet.lower():
                # Snippet doesn't even mention our player — skip
                # parsing values from it.
                continue
            if field == "time_100m":
                v = _scan_event_time(snippet, name, _LABEL_100M,
                                            10.0, 12.0)
            elif field == "time_200m":
                v = _scan_event_time(snippet, name, _LABEL_200M,
                                            20.0, 24.0)
            elif field == "forty_time":
                v = _scan_event_time(snippet, name, _LABEL_40, 4.2, 5.4,
                                            digit_pat=r"\d\.\d{1,2}")
            elif field == "long_jump_in":
                v = _scan_event_jump(snippet, name, _LABEL_LJ, 216, 320)
            elif field == "high_jump_in":
                v = _scan_event_jump(snippet, name, _LABEL_HJ, 60, 90)
            elif field == "triple_jump_in":
                v = _scan_event_jump(snippet, name, _LABEL_TJ, 420, 600)
            elif field == "vertical_in":
                v = _scan_event_jump(snippet, name, _LABEL_VERT, 24, 50)
            elif field == "broad_jump_in":
                v = _scan_event_jump(snippet, name, _LABEL_BROAD, 96, 160)
            if v is not None:
                winning_item = it
                break

        if v is not None:
            found[field] = v
            evidence[field] = [{
                "title": winning_item.get("title", ""),
                "snippet": (winning_item.get("snippet", "") or "")[:240],
                "url": winning_item.get("link", ""),
            }]
    return {"marks": found, "evidence": evidence}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=None,
                          help="Only scrape top N prospects")
    parser.add_argument("--resume", action="store_true",
                          help="Skip prospects already in output parquet")
    parser.add_argument("--sleep", type=float, default=0.3,
                          help="Seconds between API calls")
    args = parser.parse_args()

    api_key = _load_credentials()
    if not api_key:
        print("ERROR: SERPER_API_KEY must be set (env var or "
              ".streamlit/secrets.toml).")
        sys.exit(1)

    if not CONSENSUS.exists():
        print(f"ERROR: missing {CONSENSUS} — run "
              "tools/seed_draft_2027_consensus.py first.")
        sys.exit(1)

    consensus = pd.read_parquet(CONSENSUS).sort_values("expert_rank")
    if args.top:
        consensus = consensus.head(args.top)

    state_lookup = _build_state_lookup()

    existing = pd.DataFrame()
    if args.resume and OUT.exists():
        existing = pd.read_parquet(OUT)
        print(f"Resume: {len(existing)} already scraped, skipping those.")

    rows = []
    for i, p in enumerate(consensus.itertuples(), start=1):
        name = p.player
        if not existing.empty and name in existing["player"].values:
            continue
        info = state_lookup.get(name, {})
        state = info.get("state")
        hs_city = info.get("city")
        hs_school = info.get("hs_school")
        college_school = getattr(p, "school", None)  # consensus row
        print(f"[{i}/{len(consensus)}] {name} ({state or '?'}, "
              f"{college_school or '?'}) ...",
              end=" ", flush=True)
        result = scrape_prospect(name, state, hs_city, hs_school,
                                       college_school, api_key)
        marks = result["marks"]
        evidence = result["evidence"]
        rows.append({
            "player": name,
            "state": state,
            "time_100m":      marks.get("time_100m"),
            "time_200m":      marks.get("time_200m"),
            "long_jump_in":   marks.get("long_jump_in"),
            "high_jump_in":   marks.get("high_jump_in"),
            "triple_jump_in": marks.get("triple_jump_in"),
            "forty_time":     marks.get("forty_time"),
            "vertical_in":    marks.get("vertical_in"),
            "broad_jump_in":  marks.get("broad_jump_in"),
            "evidence_json":  json.dumps(evidence),
            "scraped_at":     pd.Timestamp.utcnow().isoformat(),
        })
        if marks:
            print(f"  found: {list(marks.keys())}")
        else:
            print("  no hits")
        time.sleep(args.sleep)

    if not rows:
        print("Nothing new to scrape.")
        return

    new_df = pd.DataFrame(rows)
    if not existing.empty:
        new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df = new_df.drop_duplicates(subset=["player"], keep="last")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(OUT, index=False)

    # Also dump a flat CSV for human review
    csv_path = OUT.with_suffix(".csv")
    review_cols = [c for c in new_df.columns if c != "evidence_json"]
    new_df[review_cols].to_csv(csv_path, index=False)

    n_with_marks = sum(
        any(pd.notna(new_df.loc[i, c])
            for c in ("time_100m", "time_200m", "long_jump_in",
                       "high_jump_in", "triple_jump_in", "forty_time",
                       "vertical_in", "broad_jump_in"))
        for i in new_df.index
    )
    print(f"\n✓ wrote {OUT.relative_to(REPO)}")
    print(f"  {len(new_df)} prospects, {n_with_marks} with at least one "
          "track/combine mark")
    print(f"  review CSV: {csv_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()

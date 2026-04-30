#!/usr/bin/env python3
"""Scrape track & field marks + HS combine times for 2027 Draft prospects
via Google Custom Search API.

Reads each prospect's name + state from our parquets, runs a handful
of targeted queries per prospect (100m, 200m, long jump, high jump,
triple jump, 40-yard dash), parses snippets for time/distance
patterns, and saves best-effort matches to data/draft_track_marks.parquet.
Raw snippet evidence stays in the output so the user can verify
ambiguous matches.

Setup:
  GOOGLE_CSE_API_KEY   — Custom Search API key
  GOOGLE_CSE_ID        — Programmable Search Engine ID (cx)

Both can be set as env vars OR pulled from .streamlit/secrets.toml.
Free tier covers 100 queries/day; beyond that, ~$5/1000 queries.
At 5 queries per prospect × 400 prospects = 2,000 queries ≈ $5.

Usage:
  python tools/scrape_track_marks.py             # all 400 prospects
  python tools/scrape_track_marks.py --top 50    # top 50 only
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

CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


def _load_credentials() -> tuple[str | None, str | None]:
    api_key = os.environ.get("GOOGLE_CSE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")
    if api_key and cse_id:
        return api_key, cse_id
    if SECRETS.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # py<3.11
        with open(SECRETS, "rb") as f:
            cfg = tomllib.load(f)
        api_key = api_key or cfg.get("GOOGLE_CSE_API_KEY")
        cse_id = cse_id or cfg.get("GOOGLE_CSE_ID")
    return api_key, cse_id


# ── Pattern matchers ────────────────────────────────────────────
# Search snippets are messy. Each pattern targets a specific event
# and rejects values outside reasonable HS-elite ranges.

# 100m: 10.0–11.99 sec (anything faster is suspect; slower not relevant)
RE_100M = re.compile(r"\b(1[01]\.\d{1,2})\b(?=\s*(?:sec|s\b|in\s+the\s+100|100m|100 meter))",
                       re.IGNORECASE)
RE_100M_LOOSE = re.compile(r"\b(1[01]\.\d{1,2})\b")

# 200m: 20.0–23.99 sec
RE_200M = re.compile(r"\b(2[0-3]\.\d{1,2})\b(?=\s*(?:sec|s\b|in\s+the\s+200|200m|200 meter))",
                       re.IGNORECASE)
RE_200M_LOOSE = re.compile(r"\b(2[0-3]\.\d{1,2})\b")

# 40-yard dash: 4.2–5.4 sec
RE_40YD = re.compile(r"\b(4\.\d{1,2}|5\.[0-3]\d?)\b(?=\s*(?:40|forty))",
                       re.IGNORECASE)
RE_40YD_LOOSE = re.compile(r"\b(4\.[2-9]\d?|5\.[0-3]\d?)\b")

# Long jump: 18'00" to 26'11" — match feet'inches" or feet-inches or
# decimal feet "23.5"
RE_JUMP_FTIN = re.compile(
    r"(\b\d{1,2})['’\-]\s*(\d{1,2})(?:[\"”]|\s*(?:in|inches))?",
)
RE_JUMP_DEC = re.compile(r"(\b\d{1,2}\.\d{1,2})\s*(?:ft|feet)\b",
                            re.IGNORECASE)


def _parse_jump_inches(text: str, low_in: int, high_in: int) -> float | None:
    """Try ft'in" then decimal-feet. Return inches if within range."""
    for m in RE_JUMP_FTIN.finditer(text):
        ft = int(m.group(1))
        inches = int(m.group(2) or 0)
        total = ft * 12 + inches
        if low_in <= total <= high_in:
            return float(total)
    for m in RE_JUMP_DEC.finditer(text):
        ft = float(m.group(1))
        total = ft * 12
        if low_in <= total <= high_in:
            return float(total)
    return None


def _scan_time(text: str, strict_re, loose_re,
                  low: float, high: float) -> float | None:
    """Strict pattern (with 'sec/100m' nearby) takes precedence; loose
    fallback only if strict missed. Time must fall within [low, high]."""
    for m in strict_re.finditer(text):
        try:
            v = float(m.group(1))
            if low <= v <= high:
                return v
        except ValueError:
            pass
    for m in loose_re.finditer(text):
        try:
            v = float(m.group(1))
            if low <= v <= high:
                return v
        except ValueError:
            pass
    return None


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


def cse_search(query: str, api_key: str, cse_id: str,
                  num: int = 5) -> list[dict]:
    """Hit Google Custom Search API and return result items."""
    params = {
        "key": api_key, "cx": cse_id, "q": query,
        "num": min(num, 10),
    }
    try:
        resp = requests.get(CSE_ENDPOINT, params=params, timeout=15)
        if resp.status_code != 200:
            return []
        return resp.json().get("items", []) or []
    except Exception:
        return []


def scrape_prospect(name: str, state: str | None,
                       api_key: str, cse_id: str) -> dict:
    """Run targeted queries for one prospect; parse snippets for marks."""
    state_str = f' "{state}"' if state else ""

    # Per-event queries. We add the state filter to disambiguate names.
    queries = {
        "time_100m":      f'"{name}"{state_str} 100m track',
        "time_200m":      f'"{name}"{state_str} 200m track',
        "long_jump_in":   f'"{name}"{state_str} "long jump"',
        "high_jump_in":   f'"{name}"{state_str} "high jump"',
        "triple_jump_in": f'"{name}"{state_str} "triple jump"',
        "forty_time":     f'"{name}"{state_str} "40-yard dash"',
    }

    found = {}
    evidence = {}
    for field, q in queries.items():
        items = cse_search(q, api_key, cse_id, num=5)
        # Concatenate snippet+title from top results — increases hit rate
        text = " ".join(
            (it.get("title", "") + " " + it.get("snippet", ""))
            for it in items
        )
        if field == "time_100m":
            v = _scan_time(text, RE_100M, RE_100M_LOOSE, 10.0, 12.0)
        elif field == "time_200m":
            v = _scan_time(text, RE_200M, RE_200M_LOOSE, 20.0, 24.0)
        elif field == "forty_time":
            v = _scan_time(text, RE_40YD, RE_40YD_LOOSE, 4.2, 5.4)
        elif field == "long_jump_in":
            v = _parse_jump_inches(text, 216, 320)  # 18'-26'8"
        elif field == "high_jump_in":
            v = _parse_jump_inches(text, 60, 90)    # 5'-7'6"
        elif field == "triple_jump_in":
            v = _parse_jump_inches(text, 420, 600)  # 35'-50'
        else:
            v = None
        if v is not None:
            found[field] = v
            # Save the first ~3 snippets that contained the value's
            # vicinity so the user can verify
            evidence[field] = [
                {"title": it.get("title", ""),
                 "snippet": it.get("snippet", "")[:200],
                 "url": it.get("link", "")}
                for it in items[:3]
            ]
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

    api_key, cse_id = _load_credentials()
    if not api_key or not cse_id:
        print("ERROR: GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID must be set "
              "(env vars or .streamlit/secrets.toml).")
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
        print(f"[{i}/{len(consensus)}] {name} ({state or '?'}) ...",
              end=" ", flush=True)
        result = scrape_prospect(name, state, api_key, cse_id)
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
                       "high_jump_in", "triple_jump_in", "forty_time"))
        for i in new_df.index
    )
    print(f"\n✓ wrote {OUT.relative_to(REPO)}")
    print(f"  {len(new_df)} prospects, {n_with_marks} with at least one "
          "track/combine mark")
    print(f"  review CSV: {csv_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()

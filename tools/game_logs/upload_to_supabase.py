#!/usr/bin/env python3
"""
Upload runtime parquets to Supabase Storage so the live site can read
them. Run after `make game-logs`, `make defense-scheme`, or any time
you regenerate the underlying data.

Skips the raw PBP / participation parquets — those are intermediates,
the app never reads them at runtime.

Auth:
  Reads SUPABASE_URL + SUPABASE_SERVICE_KEY from .streamlit/secrets.toml.
  The service_role key is required because anon (publishable) keys
  can't write to private buckets. Get it from your Supabase dashboard
  under Settings → API → 'service_role' key.

Usage:
    python tools/game_logs/upload_to_supabase.py
    python tools/game_logs/upload_to_supabase.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SECRETS = REPO_ROOT / ".streamlit" / "secrets.toml"
DATA_DIR = REPO_ROOT / "data" / "games"
BUCKET = "lions-rater-data"

# The parquets the production app reads at runtime. PBP +
# participation are intermediates only — keep them local.
RUNTIME_FILES = [
    "nfl_weekly_adjusted.parquet",
    "nfl_defense_baselines.parquet",
    "nfl_defensive_player_adjusted.parquet",
    "nfl_offense_baselines.parquet",
    "nfl_schedules.parquet",
    "nfl_defense_game_scheme.parquet",
    "nfl_offense_game_scheme.parquet",
    "nfl_explosive_player_games.parquet",
    "nfl_explosive_def_baselines.parquet",
    "nfl_advanced_player_games.parquet",
    "nfl_advanced_def_baselines.parquet",
    "nfl_route_distribution_player_games.parquet",
    "nfl_targeted_plays.parquet",
    "nfl_rusher_plays.parquet",
]


def _load_secrets() -> tuple[str, str]:
    if not SECRETS.exists():
        raise SystemExit(f"Missing {SECRETS}.")
    with open(SECRETS, "rb") as f:
        s = tomllib.load(f)
    url = s.get("SUPABASE_URL")
    key = s.get("SUPABASE_SERVICE_KEY") or s.get("SUPABASE_KEY")
    if not url:
        raise SystemExit("SUPABASE_URL missing from secrets.toml.")
    if not key:
        raise SystemExit("SUPABASE_KEY / SUPABASE_SERVICE_KEY missing.")
    return url, key


def main(dry_run: bool) -> None:
    url, key = _load_secrets()
    # Recognize both Supabase key formats:
    #   • "sb_secret_..." (new format) or "eyJ..." (legacy JWT) → admin-tier
    #   • "sb_publishable_..." → anon, won't be able to write
    if key.startswith("sb_publishable_"):
        print("⚠️  Using publishable/anon key — uploads will likely fail. "
              "Add SUPABASE_SERVICE_KEY (the secret key) to "
              ".streamlit/secrets.toml.")

    files = [(f, DATA_DIR / f) for f in RUNTIME_FILES]
    missing = [f for f, p in files if not p.exists()]
    if missing:
        print(f"⚠️  {len(missing)} file(s) missing locally — they'll be skipped:")
        for f in missing:
            print(f"   {f}")
        files = [(f, p) for f, p in files if p.exists()]

    total = sum(p.stat().st_size for _, p in files)
    print(f"\nWill upload {len(files)} files ({total/1024/1024:.1f} MB) "
          f"to Supabase bucket '{BUCKET}'.")
    if dry_run:
        for f, p in files:
            print(f"  • {f}  ({p.stat().st_size/1024:.0f}K)")
        print("\n(dry run — no upload)")
        return

    # Use the Storage REST API directly — bypasses supabase-py which
    # currently chokes on the new sb_secret_... key format.
    import requests
    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
    }

    # Ensure the bucket exists (idempotent).
    bucket_url = f"{url}/storage/v1/bucket"
    r = requests.post(bucket_url, headers={**headers, "Content-Type": "application/json"},
                       json={"id": BUCKET, "name": BUCKET, "public": True},
                       timeout=30)
    if r.status_code in (200, 201):
        print(f"✓ created bucket '{BUCKET}' (public)")
    elif r.status_code == 409 or "already exists" in r.text.lower():
        print(f"✓ bucket '{BUCKET}' already exists")
    else:
        print(f"⚠️  bucket create: HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code == 401 or r.status_code == 403:
            print("    Auth failed. Check SUPABASE_SERVICE_KEY in secrets.toml.")
            sys.exit(1)

    print()
    for f, p in files:
        with open(p, "rb") as fp:
            data = fp.read()
        # upsert via PUT (POST creates only; PUT replaces / inserts)
        upload_url = f"{url}/storage/v1/object/{BUCKET}/{f}"
        h = {**headers,
             "Content-Type": "application/octet-stream",
             "x-upsert": "true"}
        r = requests.put(upload_url, headers=h, data=data, timeout=300)
        if r.status_code in (200, 201):
            print(f"  ✓ {f}  ({len(data)/1024:.0f}K)")
        else:
            print(f"  ✗ {f}: HTTP {r.status_code}: {r.text[:200]}")
            sys.exit(1)

    print(f"\n✅ uploaded {len(files)} files to Supabase '{BUCKET}'.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be uploaded without sending it.")
    args = p.parse_args()
    try:
        main(args.dry_run)
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

#!/usr/bin/env python3
"""
Per-game and per-season OFFENSIVE scheme summaries — the mirror of
build_defense_scheme. Plus per-receiver route distribution.

Joins play-by-play with participation (formation + personnel labels +
route + time-to-throw). For each offense-game we compute:

  • shotgun_rate, under_center_rate, empty_rate, pistol_rate
  • pct_11_personnel, pct_12_personnel, pct_21_personnel,
    pct_10_personnel, pct_13_personnel
  • play_action_rate, no_huddle_rate
  • avg_air_yards (on pass attempts), avg_time_to_throw
  • deep_attempt_rate (≥20 air yards), pass_rate

For each receiver-game we also compute targeted-route distribution —
how many slants / hitches / gos / outs / etc. that receiver was
targeted on. (Targeted only — non-targeted routes aren't labeled in
the public NGS feed; that's PFF territory.)

Outputs:
  data/games/nfl_offense_game_scheme.parquet
    one row per (offense_team, season, week)
  data/games/nfl_offense_season_scheme.parquet
    one row per (offense_team, season)
  data/games/nfl_route_distribution_player_games.parquet
    one row per (player_id, season, week, team) with route counts
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PBP_PATH = REPO_ROOT / "data" / "games" / "nfl_pbp.parquet"
PART_PATH = REPO_ROOT / "data" / "games" / "nfl_participation.parquet"
GAME_OUT = REPO_ROOT / "data" / "games" / "nfl_offense_game_scheme.parquet"
SEASON_OUT = REPO_ROOT / "data" / "games" / "nfl_offense_season_scheme.parquet"
ROUTE_OUT = REPO_ROOT / "data" / "games" / "nfl_route_distribution_player_games.parquet"

# Top routes to surface as their own columns. "Other" lumps the rest.
ROUTE_COLUMNS = [
    "GO", "HITCH", "FLAT", "SCREEN", "OUT", "CROSS", "SLANT",
    "QUICK OUT", "HITCH/CURL", "POST", "CORNER", "IN", "IN/DIG",
    "ANGLE", "WHEEL", "SHORT CROSS", "RUN FAKE",
]


def _classify_personnel(d: str | None) -> str | None:
    """Bucket offense_personnel like '1 RB, 1 TE, 3 WR' → '11', '12',
    '21', '10', '13', '20', '22', '02', etc."""
    if d is None or not d:
        return None
    rb = te = 0
    for chunk in str(d).split(","):
        chunk = chunk.strip()
        parts = chunk.split(" ", 1)
        if len(parts) != 2:
            continue
        try:
            n = int(parts[0])
        except ValueError:
            continue
        pos = parts[1].strip()
        if pos == "RB":
            rb = n
        elif pos == "TE":
            te = n
    if 0 <= rb <= 9 and 0 <= te <= 9:
        return f"{rb}{te}"
    return None


def main():
    if not PBP_PATH.exists():
        raise SystemExit(f"Missing {PBP_PATH}.")
    if not PART_PATH.exists():
        raise SystemExit(f"Missing {PART_PATH}.")

    print("Loading inputs…")
    pbp = pl.read_parquet(PBP_PATH)
    part = pl.read_parquet(PART_PATH)
    print(f"  pbp:  {pbp.shape[0]:,} plays")
    print(f"  part: {part.shape[0]:,} plays")

    # Join PBP context with participation labels
    # Note: nflverse PBP for our seasons doesn't include `play_action`
    # — we lose that as a free signal. Everything else is here.
    pbp_slim = pbp.select([
        "game_id", "play_id", "season", "week", "season_type",
        "posteam", "defteam", "play_type", "qb_dropback",
        "shotgun", "no_huddle", "air_yards",
        "complete_pass", "yards_gained",
        "passer_player_id", "receiver_player_id",
    ])
    part_slim = part.select([
        pl.col("nflverse_game_id").alias("game_id"),
        "play_id", "offense_formation", "offense_personnel",
        "time_to_throw", "route",
    ])
    plays = pbp_slim.join(part_slim, on=["game_id", "play_id"], how="inner")
    plays = plays.filter(
        pl.col("play_type").is_in(["run", "pass"])
        & pl.col("posteam").is_not_null()
    )
    plays = plays.with_columns(
        pl.col("offense_personnel").map_elements(
            _classify_personnel, return_dtype=pl.String
        ).alias("off_pers_bucket")
    )
    print(f"  joined scrimmage plays: {plays.shape[0]:,}\n")

    # ── Per-game / per-season offense scheme ──
    def _build(group_keys, scope: str):
        pass_plays = plays.filter(pl.col("play_type") == "pass")

        # Total + per-play-type rates
        total = plays.group_by(group_keys).agg([
            pl.len().alias("n_plays"),
            (pl.col("play_type") == "run").sum().alias("n_run"),
            (pl.col("play_type") == "pass").sum().alias("n_pass"),
            pl.col("shotgun").cast(pl.Float64).mean().alias("shotgun_rate"),
            pl.col("no_huddle").cast(pl.Float64).mean().alias("no_huddle_rate"),
        ])

        # Pass-side rates
        pass_agg = pass_plays.group_by(group_keys).agg([
            pl.col("air_yards").mean().alias("avg_air_yards"),
            (pl.col("air_yards") >= 20).cast(pl.Float64).mean().alias("deep_attempt_rate"),
            pl.col("time_to_throw").mean().alias("avg_time_to_throw"),
        ])

        # Formation rates (only on plays where formation is labeled)
        form_pool = plays.filter(pl.col("offense_formation").is_not_null())
        form_agg = (
            form_pool.group_by(group_keys + ["offense_formation"])
                     .agg(pl.len().alias("n"))
        )
        form_tot = form_agg.group_by(group_keys).agg(pl.col("n").sum().alias("n_form_total"))
        form_wide = (
            form_agg.join(form_tot, on=group_keys)
                    .with_columns((pl.col("n") / pl.col("n_form_total")).alias("pct"))
                    .pivot(values="pct", index=group_keys,
                            on="offense_formation",
                            aggregate_function="first")
        )

        # Personnel split (11/12/21 etc.)
        pers_pool = plays.filter(pl.col("off_pers_bucket").is_not_null())
        pers_agg = pers_pool.group_by(group_keys + ["off_pers_bucket"]).agg(pl.len().alias("n"))
        pers_tot = pers_agg.group_by(group_keys).agg(pl.col("n").sum().alias("n_pers_total"))
        pers_wide = (
            pers_agg.join(pers_tot, on=group_keys)
                    .with_columns((pl.col("n") / pl.col("n_pers_total")).alias("pct"))
                    .pivot(values="pct", index=group_keys,
                            on="off_pers_bucket",
                            aggregate_function="first")
        )

        # Stitch
        out = total
        for piece in (pass_agg, form_wide, pers_wide):
            out = out.join(piece, on=group_keys, how="left")

        out = out.with_columns(
            (pl.col("n_pass") / pl.col("n_plays")).alias("pass_rate")
        )

        # Friendly column names
        rename_form = {
            "SHOTGUN": "form_shotgun",
            "SINGLEBACK": "form_singleback",
            "UNDER CENTER": "form_under_center",
            "I_FORM": "form_i",
            "EMPTY": "form_empty",
            "PISTOL": "form_pistol",
            "JUMBO": "form_jumbo",
            "WILDCAT": "form_wildcat",
        }
        out = out.rename({k: v for k, v in rename_form.items() if k in out.columns})
        # Personnel keys are like "11", "12", "21" — prefix to be safe
        rename_pers = {p: f"pct_{p}_personnel"
                        for p in out.columns if p.isdigit() and len(p) == 2}
        out = out.rename(rename_pers)
        out = out.rename({"posteam": "offense_team"})
        return out

    print("Building per-game scheme…")
    t0 = time.time()
    game_df = _build(["posteam", "season", "week"], "game")
    print(f"  per-game rows:   {game_df.shape[0]:,}")
    season_df = _build(["posteam", "season"], "season")
    print(f"  per-season rows: {season_df.shape[0]:,}  ({(time.time()-t0):.2f}s)\n")

    GAME_OUT.parent.mkdir(parents=True, exist_ok=True)
    game_df.write_parquet(GAME_OUT)
    season_df.write_parquet(SEASON_OUT)
    print(f"  ✓ {GAME_OUT.relative_to(REPO_ROOT)}")
    print(f"  ✓ {SEASON_OUT.relative_to(REPO_ROOT)}\n")

    # ── Per-receiver route distribution ──
    print("Building route distribution per (receiver, game)…")
    t0 = time.time()
    rec_pool = plays.filter(
        pl.col("receiver_player_id").is_not_null()
        & pl.col("route").is_not_null()
        & (pl.col("route") != "")
    )

    def _bucket_route(r):
        if r is None or r == "":
            return None
        if r in ROUTE_COLUMNS:
            return r
        return "OTHER"

    rec_pool = rec_pool.with_columns(
        pl.col("route").map_elements(_bucket_route, return_dtype=pl.String)
                       .alias("route_bucket")
    )
    rec_pool = rec_pool.with_columns(pl.col("posteam").alias("team"))

    route_agg = (
        rec_pool.group_by(["season", "week", "team", "receiver_player_id",
                            "route_bucket"])
                .agg(pl.len().alias("n"))
                .rename({"receiver_player_id": "player_id"})
    )
    route_wide = route_agg.pivot(
        values="n", index=["season", "week", "team", "player_id"],
        on="route_bucket", aggregate_function="first",
    ).fill_null(0)

    # Total targeted-routes per row
    route_cols_present = [c for c in ROUTE_COLUMNS + ["OTHER"]
                          if c in route_wide.columns]
    route_wide = route_wide.with_columns(
        pl.sum_horizontal([pl.col(c) for c in route_cols_present])
          .alias("total_routes")
    )
    # Friendly column names: "GO" → "rt_go", "QUICK OUT" → "rt_quick_out"
    rename = {c: f"rt_{c.lower().replace(' ', '_').replace('/', '_')}"
              for c in route_cols_present}
    route_wide = route_wide.rename(rename)

    print(f"  rows: {route_wide.shape[0]:,}  ({(time.time()-t0):.2f}s)")
    route_wide.write_parquet(ROUTE_OUT)
    print(f"  ✓ {ROUTE_OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  interrupted")
        sys.exit(1)

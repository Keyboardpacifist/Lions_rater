"""Team tendency explorer — answers "when this team is in X situation,
what do they do?" for offense, and "when opponents face this team in
X situation, what do they do?" for defense.

Backed by the existing per-play parquets:
  nfl_rusher_plays.parquet (per run play, with formation/personnel/box)
  data/qb_dropbacks.parquet (per dropback, with full participation
                              context — coverage, rushers, personnel)
  nfl_targeted_plays.parquet (per pass attempt, with route/coverage)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

_DATA = Path(__file__).resolve().parent / "data"


@st.cache_data(show_spinner=False)
def _player_name_index() -> dict:
    """Build a {player_id → display_name} index from the position
    parquets we already have. Used to resolve receiver_player_id
    in the tendency output to readable names."""
    out: dict[str, str] = {}
    for fname in (
        "league_wr_all_seasons.parquet",
        "league_te_all_seasons.parquet",
        "league_rb_all_seasons.parquet",
        "league_qb_all_seasons.parquet",
    ):
        path = _DATA / fname
        if not path.exists():
            continue
        try:
            df = pl.read_parquet(path).to_pandas()
        except Exception:
            continue
        if "player_id" not in df.columns:
            continue
        for name_col in ("player_display_name", "player_name", "full_name"):
            if name_col in df.columns:
                pairs = df[["player_id", name_col]].dropna().drop_duplicates(
                    subset=["player_id"]
                )
                for pid, name in zip(pairs["player_id"], pairs[name_col]):
                    if pid and pid not in out:
                        out[str(pid)] = str(name)
                break
    return out


@st.cache_data(show_spinner=False)
def _load_unified_plays() -> pd.DataFrame:
    """Return a unified per-play view: every offensive play (run or
    dropback) for every team-season we have, with formation / personnel
    / coverage context where available.

    Schema:
      game_id, play_id, season, week, posteam, defteam, play_type
        (run/pass/sack/scramble), down, ydstogo, distance_bucket,
        yardline_100, formation, personnel_group, defense_coverage,
        defense_man_zone, n_pass_rushers, defenders_in_box,
        epa, success, yards_gained,
        result fields:
          run-only: run_location, run_gap
          pass-only: target_player_name, route, complete_pass,
                      interception, air_yards, qb_hit, sack
    """
    drops_path = _DATA / "qb_dropbacks.parquet"

    # Runs — use the existing _load_rusher_plays() loader from lib_splits
    # which handles local + remote (Supabase) fallback.
    try:
        from lib_splits import _load_rusher_plays
        runs = _load_rusher_plays()
        if runs is None or runs.empty:
            runs = pd.DataFrame()
        else:
            runs = runs.copy()
            runs["play_type_unified"] = "run"
    except Exception:
        runs = pd.DataFrame()

    # Dropbacks (passes + sacks + scrambles)
    if not drops_path.exists():
        drops = pd.DataFrame()
    else:
        try:
            drops = pl.read_parquet(drops_path).to_pandas()
            # play_type col already in dropbacks ('pass', 'qb_kneel', etc.)
            # Normalize: pass attempts → 'pass'; sacks → 'sack'; scrambles → 'scramble'
            def _classify(row):
                if row.get("sack") == 1:
                    return "sack"
                if row.get("qb_scramble") == 1:
                    return "scramble"
                return "pass"
            drops["play_type_unified"] = drops.apply(_classify, axis=1)
        except Exception:
            drops = pd.DataFrame()

    # Common subset of columns to align
    common_cols = [
        "game_id", "play_id", "season", "week", "posteam", "defteam",
        "down", "ydstogo", "yardline_100", "play_type_unified",
        "epa", "success",
    ]

    # Add available context columns
    context_cols = [
        ("offense_formation", "formation"),
        ("offense_personnel", "personnel_full"),
        ("personnel_group", "personnel_group"),
        ("number_of_pass_rushers", "n_pass_rushers"),
        ("defenders_in_box", "defenders_in_box"),
        ("defense_coverage_type", "defense_coverage"),
        ("defense_man_zone_type", "defense_man_zone"),
    ]

    def _trim(df, run_specific=False, pass_specific=False):
        if df.empty:
            return df
        # Map team col → posteam if needed
        if "posteam" not in df.columns and "team" in df.columns:
            df = df.rename(columns={"team": "posteam"})
        if "defteam" not in df.columns and "opponent_team" in df.columns:
            df = df.rename(columns={"opponent_team": "defteam"})
        cols_keep = [c for c in common_cols if c in df.columns]
        for src, dst in context_cols:
            if src in df.columns:
                df = df.rename(columns={src: dst})
                cols_keep.append(dst)
        if run_specific:
            for c in ("run_location", "run_gap", "yards_gained",
                       "rusher_player_name", "player_id"):
                if c in df.columns:
                    cols_keep.append(c)
        if pass_specific:
            # Pass-side fields if present
            for c in ("complete_pass", "interception", "air_yards",
                       "passing_yards", "yards_after_catch", "qb_hit",
                       "pass_location", "pass_length", "route",
                       "receiver_player_id"):
                if c in df.columns:
                    cols_keep.append(c)
        cols_keep = list(dict.fromkeys(cols_keep))  # dedupe, preserve order
        return df[cols_keep].copy()

    runs_t = _trim(runs, run_specific=True)
    drops_t = _trim(drops, pass_specific=True)
    if runs_t.empty and drops_t.empty:
        return pd.DataFrame()
    out = pd.concat([runs_t, drops_t], ignore_index=True, sort=False)

    # Distance bucket
    def _dist_bucket(d):
        if pd.isna(d):
            return None
        if d <= 3:
            return "Short"
        if d <= 7:
            return "Medium"
        return "Long"
    out["distance_bucket"] = out["ydstogo"].apply(_dist_bucket)

    # Personnel group (for runs that only have full personnel string)
    def _personnel_group(row):
        existing = row.get("personnel_group")
        if isinstance(existing, str) and existing:
            return existing
        full = row.get("personnel_full")
        if not isinstance(full, str) or not full:
            return None
        rb = te = wr = 0
        for tok in full.split(","):
            tok = tok.strip()
            head = tok.split(" ")[0]
            if not head.isdigit():
                continue
            if " RB" in tok:
                rb = int(head)
            elif " TE" in tok:
                te = int(head)
            elif " WR" in tok:
                wr = int(head)
        if rb == 0:
            return "Empty"
        if rb == 1 and te == 1 and wr == 3:
            return "11"
        if rb == 1 and te == 2 and wr == 2:
            return "12"
        if rb == 2 and te == 1 and wr == 2:
            return "21"
        if te >= 3 or rb >= 2:
            return "Heavy"
        return None
    out["personnel_group_norm"] = out.apply(_personnel_group, axis=1)

    return out


def _apply_filters(df: pd.DataFrame, *,
                     downs: list[int] | None,
                     distance_buckets: list[str] | None,
                     formation: str | None,
                     personnel: list[str] | None,
                     coverage: list[str] | None,
                     manzone: str | None,
                     rushers: str | None) -> pd.DataFrame:
    out = df
    if downs:
        out = out[out["down"].isin(downs)]
    if distance_buckets:
        out = out[out["distance_bucket"].isin(distance_buckets)]
    if formation and formation != "All":
        out = out[out.get("formation") == formation]
    if personnel:
        out = out[out["personnel_group_norm"].isin(personnel)]
    if coverage:
        out = out[out["defense_coverage"].isin(coverage)]
    if manzone == "Man":
        out = out[out["defense_man_zone"] == "MAN_COVERAGE"]
    elif manzone == "Zone":
        out = out[out["defense_man_zone"] == "ZONE_COVERAGE"]
    if rushers == "3":
        out = out[out["n_pass_rushers"] == 3]
    elif rushers == "4":
        out = out[out["n_pass_rushers"] == 4]
    elif rushers == "5+":
        out = out[out["n_pass_rushers"] >= 5]
    return out


def get_team_tendencies(team: str, season: int, *, side: str = "offense",
                          **filters) -> dict:
    """Returns aggregate breakdown for the filtered slice.

    side='offense'  → plays where this team had the ball
    side='defense'  → plays where this team was defending

    Aggregates returned:
      - n_plays, n_runs, n_passes
      - run_pass_split (pct run, pct pass)
      - run_dir_breakdown ({left, middle, right} pcts)
      - top_targets ([{name, n_targets, catch_pct, epa_per_target}, …])
    """
    df = _load_unified_plays()
    if df.empty:
        return {}
    if side == "offense":
        df = df[df["posteam"] == team]
    else:
        df = df[df["defteam"] == team]
    df = df[df["season"] == season]
    if df.empty:
        return {}

    df = _apply_filters(df, **filters)
    if df.empty:
        return {"n_plays": 0}

    n = len(df)
    runs = df[df["play_type_unified"] == "run"]
    passes = df[df["play_type_unified"].isin(["pass", "scramble"])]
    sacks = df[df["play_type_unified"] == "sack"]

    run_dir = (
        runs["run_location"].value_counts(normalize=True).to_dict()
        if not runs.empty and "run_location" in runs.columns
        else {}
    )

    # Top receivers (offense view) — resolve IDs to names
    top_targets = []
    if side == "offense" and not passes.empty and "receiver_player_id" in passes.columns:
        name_idx = _player_name_index()
        target_agg = (
            passes.dropna(subset=["receiver_player_id"])
            .groupby("receiver_player_id")
            .agg(
                n=("epa", "size"),
                catch_pct=("complete_pass", "mean"),
                epa=("epa", "mean"),
            )
            .reset_index()
            .sort_values("n", ascending=False)
            .head(8)
        )
        target_agg["name"] = target_agg["receiver_player_id"].apply(
            lambda pid: name_idx.get(str(pid), str(pid))
        )
        top_targets = target_agg.to_dict("records")

    # Top runners (per-back distribution on run plays)
    top_runners = []
    if not runs.empty and "rusher_player_name" in runs.columns:
        runner_agg = (
            runs.dropna(subset=["rusher_player_name"])
            .groupby("rusher_player_name")
            .agg(
                n=("epa", "size"),
                ypc=("yards_gained", "mean"),
                epa=("epa", "mean"),
                success_rate=("success", "mean"),
            )
            .reset_index()
            .sort_values("n", ascending=False)
            .head(6)
        )
        top_runners = runner_agg.to_dict("records")

    return {
        "n_plays": n,
        "n_runs": len(runs),
        "n_passes": len(passes) + len(sacks),  # dropbacks
        "run_pct": len(runs) / n if n else 0,
        "pass_pct": (len(passes) + len(sacks)) / n if n else 0,
        "run_epa": float(runs["epa"].mean()) if not runs.empty else None,
        "pass_epa": float(passes["epa"].mean()) if not passes.empty else None,
        "sack_pct": len(sacks) / max(len(passes) + len(sacks), 1),
        "run_direction": run_dir,
        "top_targets": top_targets,
        "top_runners": top_runners,
    }


@st.cache_data(show_spinner=False)
def get_filter_options(team: str, season: int, side: str = "offense") -> dict:
    """Available filter values for the given team-season — used to
    populate dropdowns with only options that actually have data."""
    df = _load_unified_plays()
    if df.empty:
        return {}
    if side == "offense":
        df = df[df["posteam"] == team]
    else:
        df = df[df["defteam"] == team]
    df = df[df["season"] == season]
    if df.empty:
        return {}

    return {
        "formations": sorted(
            x for x in df.get("formation", pd.Series()).dropna().unique()
        ),
        "personnel": sorted(
            x for x in df.get("personnel_group_norm", pd.Series()).dropna().unique()
        ),
        "coverages": sorted(
            x for x in df.get("defense_coverage", pd.Series()).dropna().unique()
        ),
    }

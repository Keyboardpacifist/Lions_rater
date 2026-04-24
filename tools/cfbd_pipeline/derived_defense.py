"""
Augment the existing college defensive parquet with derived nerd stats.

All math is pure arithmetic on columns already in the file — no API
calls, no new sources. Writes back to data/college/college_def_all_seasons.parquet
in place. Z-scores are computed within (season, pos_group) so positions
are compared to their peers, not all defenders together.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "college" / "college_def_all_seasons.parquet"


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the five derived nerd stats to a defensive DataFrame."""
    df = df.copy()

    # Safe denominators (replace 0 with NaN so divisions yield NaN, not inf)
    games_safe = df["games"].replace(0, np.nan) if "games" in df.columns else np.nan
    tackles_safe = df["tackles_total"].replace(0, np.nan) if "tackles_total" in df.columns else np.nan

    # 1. Splash plays per game — single composite "disruption" rate
    splash = (
        df.get("sacks", 0).fillna(0)
        + df.get("tfl", 0).fillna(0)
        + df.get("interceptions", 0).fillna(0)
        + df.get("passes_deflected", 0).fillna(0)
    )
    df["splash_plays_per_game"] = splash / games_safe

    # 2. TFL share — penetration tackler vs. clean-up tackler
    if "tfl" in df.columns:
        df["tfl_share"] = df["tfl"] / tackles_safe
    else:
        df["tfl_share"] = np.nan

    # 3. Ball production per game — coverage / pass-disruption proxy
    ball = (
        df.get("interceptions", 0).fillna(0)
        + df.get("passes_deflected", 0).fillna(0)
    )
    df["ball_production_per_game"] = ball / games_safe

    # 4. Pressure conversion rate — sacks as a % of all pressures
    sacks_plus_hurries = df.get("sacks", 0).fillna(0) + df.get("qb_hurries", 0).fillna(0)
    df["pressure_conversion_rate"] = (
        df.get("sacks", 0).fillna(0) / sacks_plus_hurries.replace(0, np.nan)
    )

    # 5. INT-per-PD ratio — ball-hawking instinct
    pd_plus_int = df.get("interceptions", 0).fillna(0) + df.get("passes_deflected", 0).fillna(0)
    df["int_per_pd_ratio"] = (
        df.get("interceptions", 0).fillna(0) / pd_plus_int.replace(0, np.nan)
    )

    return df


def zscore_within_season_pos(df: pd.DataFrame, cols: list[str], min_games: int = 4) -> pd.DataFrame:
    """Z-score each col within (season, pos_group). Only include rows
    above min_games in the reference pool so noise doesn't pollute the mean."""
    df = df.copy()
    if "season" not in df.columns or "pos_group" not in df.columns:
        return df

    # Reference pool excludes low-volume players
    if "games" in df.columns:
        elig = df["games"].fillna(0) >= min_games
    else:
        elig = pd.Series(True, index=df.index)

    for col in cols:
        if col not in df.columns:
            df[col + "_z"] = np.nan
            continue
        z = pd.Series(np.nan, index=df.index)
        for (season, pos), grp in df[elig].groupby(["season", "pos_group"]):
            vals = grp[col].dropna()
            if len(vals) < 5:
                continue
            mean = vals.mean()
            std = vals.std()
            if not std or std == 0:
                continue
            mask = (df["season"] == season) & (df["pos_group"] == pos)
            z.loc[mask] = (df.loc[mask, col] - mean) / std
        df[col + "_z"] = z
    return df


def run(verbose: bool = True) -> None:
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Defensive parquet missing: {OUTPUT_PATH}")
    df = pd.read_parquet(OUTPUT_PATH)
    if verbose:
        print(f"Loaded {OUTPUT_PATH.name}: {len(df)} rows, {len(df.columns)} cols")

    df = add_derived_columns(df)
    z_cols = [
        "splash_plays_per_game",
        "tfl_share",
        "ball_production_per_game",
        "pressure_conversion_rate",
        "int_per_pd_ratio",
    ]
    df = zscore_within_season_pos(df, z_cols)
    if verbose:
        print(f"Added {len(z_cols)} derived stats + their z-scores")

    df.to_parquet(OUTPUT_PATH, index=False)
    if verbose:
        print(f"Wrote {OUTPUT_PATH.name}: {len(df)} rows, {len(df.columns)} cols")


if __name__ == "__main__":
    run(verbose=True)

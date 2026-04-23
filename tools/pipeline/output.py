"""
Output validation and parquet/metadata writer.

Validates that output DataFrames match what the Streamlit pages expect
before writing. Catches schema drift early.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import PositionConfig

# Columns every output parquet must have (the page contract).
REQUIRED_COLUMNS = [
    "player_id",
    "player_display_name",
    "position",
    "recent_team",
    "season_year",
    "off_snaps",
]


class ValidationError(Exception):
    """Raised when output validation fails."""


def validate(df: pd.DataFrame, config: PositionConfig) -> list[str]:
    """Validate a DataFrame against the page contract.

    Returns a list of warnings (non-fatal). Raises ValidationError for
    fatal issues.
    """
    warnings = []

    # Check required columns
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValidationError(
            f"Missing required columns: {missing_required}"
        )

    # Check z-score columns exist
    expected_z = [f"{s}_z" for s in config.stats_to_zscore]
    missing_z = [c for c in expected_z if c not in df.columns]
    if missing_z:
        warnings.append(f"Missing z-score columns: {missing_z}")

    # Check for all-NaN z-score columns
    for col in expected_z:
        if col in df.columns and df[col].isna().all():
            warnings.append(f"All-NaN z-score column: {col}")

    # Check z-score distributions (sanity: mean near 0, std near 1)
    for col in expected_z:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 3:
            continue
        mean = vals.mean()
        std = vals.std()
        if abs(mean) > 0.5:
            warnings.append(
                f"Z-score {col} has mean={mean:.2f} (expected ~0)"
            )
        if std < 0.3 or std > 3.0:
            warnings.append(
                f"Z-score {col} has std={std:.2f} (expected ~1)"
            )

    # Check row count
    if len(df) == 0:
        raise ValidationError("Output DataFrame is empty")

    return warnings


def write_parquet(
    df: pd.DataFrame,
    config: PositionConfig,
    output_dir: Path,
    verbose: bool = True,
) -> list[Path]:
    """Validate and write output parquet file(s).

    Args:
        df: The full population DataFrame with z-scores.
        config: Position config (defines output filenames and columns).
        output_dir: Directory to write to (typically data/).
        verbose: Print progress.

    Returns:
        List of paths written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate
    warnings = validate(df, config)
    for w in warnings:
        if verbose:
            print(f"  WARNING: {w}")

    # Select output columns (only those that exist)
    if config.output_columns:
        out_cols = [c for c in config.output_columns if c in df.columns]
        out = df[out_cols].copy()
    else:
        out = df.copy()

    # Write one parquet per output filename
    paths = []
    for filename in config.output_filenames:
        path = output_dir / filename
        out.to_parquet(path, index=False)
        paths.append(path)
        if verbose:
            print(f"  Wrote {path} ({len(out)} rows, {len(out.columns)} cols)")

    return paths


def write_metadata(
    df: pd.DataFrame,
    config: PositionConfig,
    output_dir: Path,
    season: int | str = "multi",
    verbose: bool = True,
) -> Path:
    """Write stat metadata JSON.

    Args:
        df: The output DataFrame (for row count).
        config: Position config.
        output_dir: Directory to write to.
        season: Season label for metadata.
        verbose: Print progress.

    Returns:
        Path to the metadata file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.snap_floor:
        population_desc = (
            "Players with >= "
            + ", ".join(f"{n} {p}" for p, n in config.snap_floor.items())
            + f" total snaps, min {config.min_games} games"
        )
    elif config.top_n:
        population_desc = (
            f"Top {sum(config.top_n.values())} "
            + "+".join(f"{n} {p}" for p, n in config.top_n.items())
            + f" by total offensive snaps, min {config.min_games} games"
        )
    else:
        population_desc = f"Min {config.min_games} games"

    metadata = {
        "position_group": config.key,
        "season": season,
        "population": population_desc,
        "n_players": int(len(df)),
        "stat_tiers": config.stat_tiers,
        "stat_labels": config.stat_labels,
        "stat_methodology": config.stat_methodology,
        "invert_stats": sorted(list(config.invert_stats)),
    }

    path = output_dir / config.metadata_filename
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"  Wrote {path}")

    return path

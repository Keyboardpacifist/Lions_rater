"""
Z-score engine — shared across all position pipelines.

Extracted from the identical loop in wr_data_pull.py:222-243 and
rb_data_pull.py:341-361. One function, tested, no duplication.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_stats(
    df: pd.DataFrame,
    stats: list[str],
    invert: set[str] | None = None,
    min_non_null: int = 3,
    suffix: str = "_z",
) -> pd.DataFrame:
    """Add z-score columns for each stat in `stats`.

    For each stat:
      - Skip if column is missing (z-col = 0.0)
      - Skip if fewer than `min_non_null` non-null values (z-col = NaN)
      - Otherwise: z = (x - mean) / std, inverted if stat is in `invert`

    Args:
        df: DataFrame with raw stat columns.
        stats: List of column names to z-score.
        invert: Set of stat names where lower is better (z = -z).
        min_non_null: Minimum non-null values required to compute z-scores.
        suffix: Suffix for z-score column names (default: "_z").

    Returns:
        df with new z-score columns added (mutates in place AND returns).
    """
    invert = invert or set()

    for stat in stats:
        z_col = f"{stat}{suffix}"

        if stat not in df.columns:
            df[z_col] = 0.0
            continue

        vals = df[stat].astype(float)
        clean = vals.dropna()

        if len(clean) < min_non_null:
            df[z_col] = np.nan
            continue

        mean = clean.mean()
        std = clean.std()

        if std and std > 0:
            z = (vals - mean) / std
            if stat in invert:
                z = -z
            df[z_col] = z
        else:
            df[z_col] = 0.0

    return df

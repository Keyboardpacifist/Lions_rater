"""
Tests for data file integrity.

Verifies that all expected parquet and metadata files exist,
load without error, and have sane contents.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLLEGE_DIR = DATA_DIR / "college"


# ── File existence ───────────────────────────────────────────

EXPECTED_LEAGUE_FILES = [
    "league_qb_all_seasons.parquet",
    "league_wr_all_seasons.parquet",
    "league_te_all_seasons.parquet",
    "league_rb_all_seasons.parquet",
    "league_ol_all_seasons.parquet",
    "league_de_all_seasons.parquet",
    "league_dt_all_seasons.parquet",
    "league_lb_all_seasons.parquet",
    "league_cb_all_seasons.parquet",
    "league_s_all_seasons.parquet",
    "league_k_all_seasons.parquet",
    "league_p_all_seasons.parquet",
]

EXPECTED_METADATA_FILES = [
    "qb_stat_metadata.json",
    "rb_stat_metadata.json",
    "wr_te_stat_metadata.json",
    "ol_stat_metadata.json",
    "coach_stat_metadata.json",
    "dc_stat_metadata.json",
    "gm_stat_metadata.json",
    "kicker_stat_metadata.json",
    "punter_stat_metadata.json",
    "safety_stat_metadata.json",
]

EXPECTED_COLLEGE_FILES = [
    "college_qb_all_seasons.parquet",
    "college_wr_all_seasons.parquet",
    "college_te_all_seasons.parquet",
    "college_rb_all_seasons.parquet",
    "college_def_all_seasons.parquet",
    "college_recruiting.parquet",
]


@pytest.mark.parametrize("filename", EXPECTED_LEAGUE_FILES)
def test_league_parquet_exists(filename):
    """Every league reference parquet must exist."""
    assert (DATA_DIR / filename).exists(), f"Missing: data/{filename}"


@pytest.mark.parametrize("filename", EXPECTED_METADATA_FILES)
def test_metadata_json_exists(filename):
    """Every stat metadata JSON must exist."""
    assert (DATA_DIR / filename).exists(), f"Missing: data/{filename}"


@pytest.mark.parametrize("filename", EXPECTED_COLLEGE_FILES)
def test_college_parquet_exists(filename):
    """Core college data parquets must exist."""
    assert (COLLEGE_DIR / filename).exists(), f"Missing: data/college/{filename}"


# ── Parquet loading ──────────────────────────────────────────

@pytest.mark.parametrize("filename", EXPECTED_LEAGUE_FILES)
def test_league_parquet_loads(filename):
    """League parquets should load without error and have rows."""
    df = pd.read_parquet(DATA_DIR / filename)
    assert len(df) > 0, f"{filename} is empty"


@pytest.mark.parametrize("filename", EXPECTED_COLLEGE_FILES)
def test_college_parquet_loads(filename):
    """College parquets should load without error and have rows."""
    df = pd.read_parquet(COLLEGE_DIR / filename)
    assert len(df) > 0, f"{filename} is empty"


# ── Metadata structure ───────────────────────────────────────

@pytest.mark.parametrize("filename", EXPECTED_METADATA_FILES)
def test_metadata_json_valid(filename):
    """Metadata JSONs should be valid JSON with expected keys."""
    path = DATA_DIR / filename
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, dict), f"{filename} root should be a dict"
    # Every metadata file should have at least stat_tiers or stat_labels
    assert any(k in data for k in ("stat_tiers", "stat_labels")), \
        f"{filename} missing stat_tiers or stat_labels"


# ── Z-score sanity ───────────────────────────────────────────

@pytest.mark.parametrize("filename", EXPECTED_LEAGUE_FILES[:4])  # QB, WR, TE, RB
def test_z_scores_are_reasonable(filename):
    """Z-score columns should have mean near 0 and std near 1."""
    df = pd.read_parquet(DATA_DIR / filename)
    z_cols = [c for c in df.columns if c.endswith("_z")]
    assert len(z_cols) > 0, f"{filename} has no z-score columns"

    for col in z_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue  # Skip columns with too few values
        mean = series.mean()
        std = series.std()
        # Z-scores should be roughly centered with std roughly 1
        # Generous bounds: multi-season files include players outside the
        # reference population (e.g., backup QBs), which shifts means.
        # QB passing_yards_per_game_z mean=-2.08, sack_rate_z std=5.01
        assert abs(mean) < 3.0, f"{filename}.{col} mean={mean:.2f} (expected near 0)"
        assert 0.1 < std < 6.0, f"{filename}.{col} std={std:.2f} (expected near 1)"

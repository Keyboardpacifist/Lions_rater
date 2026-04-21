"""
Tests for the data pipeline (tools/pipeline/).

Tests the z-score engine, output validation, and position config
structure without requiring network access or nflverse data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add tools/ to path so pipeline imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from pipeline.zscore import zscore_stats
from pipeline.output import validate, ValidationError, REQUIRED_COLUMNS
from pipeline.base import PositionConfig


# ── zscore_stats tests ───────────────────────────────────────────────────────


class TestZscoreStats:
    def test_basic_zscore(self):
        """Z-scores should have mean ~0 and std ~1."""
        df = pd.DataFrame({"stat_a": [10, 20, 30, 40, 50]})
        result = zscore_stats(df, ["stat_a"])
        assert "stat_a_z" in result.columns
        z = result["stat_a_z"]
        assert abs(z.mean()) < 1e-10
        assert abs(z.std() - 1.0) < 0.1  # ddof=1 gives slightly different std

    def test_invert_stat(self):
        """Inverted stats should flip the z-score sign."""
        df = pd.DataFrame({"sack_rate": [1, 2, 3, 4, 5]})
        result = zscore_stats(df, ["sack_rate"], invert={"sack_rate"})
        # Highest sack_rate should have most negative z-score
        assert result["sack_rate_z"].iloc[4] < result["sack_rate_z"].iloc[0]

    def test_missing_column(self):
        """Missing columns get z=0.0."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = zscore_stats(df, ["nonexistent"])
        assert "nonexistent_z" in result.columns
        assert (result["nonexistent_z"] == 0.0).all()

    def test_too_few_values(self):
        """Fewer than min_non_null values get z=NaN."""
        df = pd.DataFrame({"stat_a": [1.0, np.nan, np.nan]})
        result = zscore_stats(df, ["stat_a"], min_non_null=3)
        assert result["stat_a_z"].isna().all()

    def test_zero_std(self):
        """Constant values (std=0) get z=0.0."""
        df = pd.DataFrame({"stat_a": [5.0, 5.0, 5.0, 5.0]})
        result = zscore_stats(df, ["stat_a"])
        assert (result["stat_a_z"] == 0.0).all()

    def test_nan_preserved(self):
        """NaN inputs should produce NaN z-scores."""
        df = pd.DataFrame({"stat_a": [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = zscore_stats(df, ["stat_a"])
        assert result["stat_a_z"].iloc[2] != result["stat_a_z"].iloc[2]  # NaN != NaN

    def test_custom_suffix(self):
        """Custom suffix should work."""
        df = pd.DataFrame({"stat_a": [1, 2, 3, 4, 5]})
        result = zscore_stats(df, ["stat_a"], suffix="_zscore")
        assert "stat_a_zscore" in result.columns

    def test_multiple_stats(self):
        """Multiple stats should all get z-scored."""
        df = pd.DataFrame({
            "stat_a": [10, 20, 30, 40, 50],
            "stat_b": [1, 2, 3, 4, 5],
        })
        result = zscore_stats(df, ["stat_a", "stat_b"])
        assert "stat_a_z" in result.columns
        assert "stat_b_z" in result.columns


# ── Output validation tests ──────────────────────────────────────────────────


def _make_minimal_config(**overrides) -> PositionConfig:
    """Create a minimal PositionConfig for testing."""
    defaults = dict(
        key="test",
        output_filenames=["test.parquet"],
        metadata_filename="test_metadata.json",
        snap_positions=["WR"],
        top_n={"WR": 32},
        min_games=6,
        pbp_play_types=["pass"],
        ngs_stat_type=None,
        pfr_stat_type=None,
        aggregate_stats=[],
        stats_to_zscore=["rec_yards"],
    )
    defaults.update(overrides)
    return PositionConfig(**defaults)


class TestValidation:
    def test_valid_df(self):
        """A valid DataFrame should pass with no fatal errors."""
        df = pd.DataFrame({
            "player_id": ["A", "B", "C"],
            "player_display_name": ["A", "B", "C"],
            "position": ["WR", "WR", "WR"],
            "recent_team": ["DET", "DET", "DET"],
            "season_year": [2024, 2024, 2024],
            "off_snaps": [500, 400, 300],
            "rec_yards": [1000, 800, 600],
            "rec_yards_z": [1.0, 0.0, -1.0],
        })
        config = _make_minimal_config()
        warnings = validate(df, config)
        # Should not raise

    def test_missing_required_columns(self):
        """Missing required columns should raise ValidationError."""
        df = pd.DataFrame({"player_id": ["A"]})
        config = _make_minimal_config()
        with pytest.raises(ValidationError, match="Missing required columns"):
            validate(df, config)

    def test_empty_df(self):
        """Empty DataFrame should raise ValidationError."""
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        config = _make_minimal_config()
        with pytest.raises(ValidationError, match="empty"):
            validate(df, config)

    def test_all_nan_zscore_warning(self):
        """All-NaN z-score columns should produce a warning."""
        df = pd.DataFrame({
            "player_id": ["A", "B", "C"],
            "player_display_name": ["A", "B", "C"],
            "position": ["WR", "WR", "WR"],
            "recent_team": ["DET", "DET", "DET"],
            "season_year": [2024, 2024, 2024],
            "off_snaps": [500, 400, 300],
            "rec_yards_z": [np.nan, np.nan, np.nan],
        })
        config = _make_minimal_config()
        warnings = validate(df, config)
        assert any("All-NaN" in w for w in warnings)


# ── Position config structure tests ───��──────────────────────────────────────


class TestPositionConfigs:
    def test_wr_config_loads(self):
        """WR config should load without errors."""
        from pipeline.positions.wr import WR_CONFIG
        assert WR_CONFIG.key == "wr"
        assert len(WR_CONFIG.output_filenames) == 2
        assert len(WR_CONFIG.stats_to_zscore) > 0
        assert len(WR_CONFIG.stat_tiers) > 0
        assert len(WR_CONFIG.stat_labels) > 0

    def test_wr_tiers_match_zscores(self):
        """Every z-score stat should have a tier assignment."""
        from pipeline.positions.wr import WR_CONFIG
        expected_z = {f"{s}_z" for s in WR_CONFIG.stats_to_zscore}
        tier_keys = set(WR_CONFIG.stat_tiers.keys())
        assert expected_z == tier_keys, f"Mismatch: {expected_z.symmetric_difference(tier_keys)}"

    def test_wr_labels_match_tiers(self):
        """Every tier entry should have a label."""
        from pipeline.positions.wr import WR_CONFIG
        assert set(WR_CONFIG.stat_tiers.keys()) == set(WR_CONFIG.stat_labels.keys())

    def test_wr_output_columns_include_zscores(self):
        """Output columns should include all z-score columns."""
        from pipeline.positions.wr import WR_CONFIG
        expected_z = {f"{s}_z" for s in WR_CONFIG.stats_to_zscore}
        output_set = set(WR_CONFIG.output_columns)
        missing = expected_z - output_set
        assert not missing, f"Z-score columns missing from output: {missing}"

    def test_rb_config_loads(self):
        """RB config should load without errors."""
        from pipeline.positions.rb import RB_CONFIG
        assert RB_CONFIG.key == "rb"
        assert len(RB_CONFIG.stats_to_zscore) > 0

    def test_registry_has_wr(self):
        """The position registry should contain WR."""
        from pipeline.positions import POSITIONS
        assert "wr" in POSITIONS

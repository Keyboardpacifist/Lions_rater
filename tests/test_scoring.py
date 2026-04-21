"""
Tests for the core scoring engine in lib_shared.py.

These are the most critical tests — if scoring is wrong,
every ranking on every page is wrong.
"""
import pandas as pd
import pytest


def test_score_players_basic(sample_player_data):
    """Weighted average of z-scores should produce correct composite scores."""
    from lib_shared import score_players

    weights = {"stat_a_z": 50, "stat_b_z": 50}
    result = score_players(sample_player_data, weights)

    assert "score" in result.columns
    # Player A: (1.0 * 0.5) + (0.5 * 0.5) = 0.75
    assert abs(result.iloc[0]["score"] - 0.75) < 1e-9
    # Player B: (0.0 * 0.5) + (1.5 * 0.5) = 0.75
    assert abs(result.iloc[1]["score"] - 0.75) < 1e-9
    # Player C: (-1.0 * 0.5) + (-0.5 * 0.5) = -0.75
    assert abs(result.iloc[2]["score"] - (-0.75)) < 1e-9


def test_score_players_single_stat(sample_player_data):
    """When only one stat has weight, score equals that stat's z-score."""
    from lib_shared import score_players

    weights = {"stat_a_z": 100}
    result = score_players(sample_player_data, weights)

    assert abs(result.iloc[0]["score"] - 1.0) < 1e-9
    assert abs(result.iloc[1]["score"] - 0.0) < 1e-9
    assert abs(result.iloc[2]["score"] - (-1.0)) < 1e-9


def test_score_players_all_zero_weights(sample_player_data):
    """All-zero weights should produce all-zero scores, not crash."""
    from lib_shared import score_players

    weights = {"stat_a_z": 0, "stat_b_z": 0}
    result = score_players(sample_player_data, weights)

    assert all(result["score"] == 0.0)


def test_score_players_empty_weights(sample_player_data):
    """Empty weight dict should produce all-zero scores."""
    from lib_shared import score_players

    result = score_players(sample_player_data, {})
    assert all(result["score"] == 0.0)


def test_score_players_missing_column(sample_player_data):
    """Weights referencing nonexistent columns should be silently skipped."""
    from lib_shared import score_players

    weights = {"stat_a_z": 50, "nonexistent_z": 50}
    result = score_players(sample_player_data, weights)

    # Only stat_a_z contributes; nonexistent is skipped but still in denominator
    # score = stat_a_z * (50/100) = stat_a_z * 0.5
    assert abs(result.iloc[0]["score"] - 0.5) < 1e-9


def test_score_players_does_not_mutate_input(sample_player_data):
    """score_players should return a copy, not modify the input."""
    from lib_shared import score_players

    original_cols = set(sample_player_data.columns)
    _ = score_players(sample_player_data, {"stat_a_z": 50})
    assert "score" not in sample_player_data.columns
    assert set(sample_player_data.columns) == original_cols


def test_compute_effective_weights():
    """Bundle weights should correctly combine into per-stat weights."""
    from lib_shared import compute_effective_weights

    bundles = {
        "reliability": {
            "stats": {"catch_rate_z": 0.5, "success_rate_z": 0.5},
        },
        "explosive": {
            "stats": {"yards_per_target_z": 0.6, "catch_rate_z": 0.4},
        },
    }
    bundle_weights = {"reliability": 80, "explosive": 40}

    result = compute_effective_weights(bundles, bundle_weights)

    # catch_rate_z: 80 * 0.5 + 40 * 0.4 = 40 + 16 = 56
    assert abs(result["catch_rate_z"] - 56.0) < 1e-9
    # success_rate_z: 80 * 0.5 = 40
    assert abs(result["success_rate_z"] - 40.0) < 1e-9
    # yards_per_target_z: 40 * 0.6 = 24
    assert abs(result["yards_per_target_z"] - 24.0) < 1e-9


def test_compute_effective_weights_zero_bundle():
    """Bundles with weight 0 should be excluded entirely."""
    from lib_shared import compute_effective_weights

    bundles = {
        "a": {"stats": {"stat_1_z": 1.0}},
        "b": {"stats": {"stat_2_z": 1.0}},
    }
    result = compute_effective_weights(bundles, {"a": 50, "b": 0})

    assert "stat_1_z" in result
    assert "stat_2_z" not in result


def test_score_players_handles_nan():
    """NaN z-scores should be treated as 0 (not propagate NaN)."""
    from lib_shared import score_players

    df = pd.DataFrame({
        "player": ["A"],
        "stat_z": [float("nan")],
    })
    result = score_players(df, {"stat_z": 100})
    assert result.iloc[0]["score"] == 0.0

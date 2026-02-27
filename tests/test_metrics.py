"""Tests for src/lazyforecast/metrics.py."""

import numpy as np
import pytest

from lazyforecast.metrics import (
    build_eval_table,
    forecast_accuracy,
    mean_directional_accuracy,
)


def test_mda_perfect_forecast():
    """When predicted == actual, every direction is correct → MDA = 1.0."""
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = actual.copy()
    assert mean_directional_accuracy(actual, predicted) == pytest.approx(1.0)


def test_mda_opposite_forecast():
    """When predicted always moves in the opposite direction → MDA = 0.0."""
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])          # always up
    predicted = np.array([5.0, 4.0, 3.0, 2.0, 1.0])        # always down
    assert mean_directional_accuracy(actual, predicted) == pytest.approx(0.0)


def test_mda_halfway():
    """Half correct, half wrong → MDA = 0.5."""
    # 4 transitions: [+, +, -, -]
    actual = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    # 4 transitions: [+, +, +, +]  — matches first 2, misses last 2
    predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean_directional_accuracy(actual, predicted) == pytest.approx(0.5)


def test_forecast_accuracy_shapes():
    """forecast_accuracy must return a dict with all 6 metric keys."""
    actual = np.arange(1.0, 11.0)
    predicted = actual + 0.1
    result = forecast_accuracy(actual, predicted, "TestModel")

    assert isinstance(result, dict)
    for key in ("model", "mda", "rmse", "mape", "R2", "mae", "corr"):
        assert key in result, f"Missing key: {key}"


def test_forecast_accuracy_perfect():
    """Perfect predictions should yield rmse=0, mape≈0, R2=1, mae=0."""
    actual = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    predicted = actual.copy()
    result = forecast_accuracy(actual, predicted, "Perfect")

    assert result["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert result["mae"] == pytest.approx(0.0, abs=1e-6)
    assert result["R2"] == pytest.approx(1.0)


def test_build_eval_table_sorting():
    """Lower RMSE should be ranked above higher RMSE (when MDA is equal)."""
    eval_data = [
        {"model": "Bad",  "mda": 0.5, "rmse": 10.0, "mape": 0.2, "R2": 0.5, "mae": 5.0, "corr": 0.8},
        {"model": "Good", "mda": 0.5, "rmse":  2.0, "mape": 0.1, "R2": 0.9, "mae": 1.0, "corr": 0.95},
    ]
    table = build_eval_table(eval_data)
    assert table.index[0] == "Good"
    assert table.index[1] == "Bad"


def test_build_eval_table_columns():
    """Eval table must contain all metric columns."""
    eval_data = [
        {"model": "M1", "mda": 0.7, "rmse": 1.0, "mape": 0.05, "R2": 0.9, "mae": 0.5, "corr": 0.97},
    ]
    table = build_eval_table(eval_data)
    for col in ("mda", "rmse", "mape", "R2", "mae", "corr"):
        assert col in table.columns

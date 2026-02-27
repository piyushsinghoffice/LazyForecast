"""Unit tests for src/lazyforecast/baselines.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazyforecast.baselines import drift_forecast, naive_forecast, seasonal_naive_forecast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_PERIODS = 5


def _series(n: int = 40, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0, 1, n)))


# ---------------------------------------------------------------------------
# naive_forecast
# ---------------------------------------------------------------------------


def test_naive_output_shapes():
    s = _series()
    fc, confint = naive_forecast(s, N_PERIODS)
    assert fc.shape == (N_PERIODS,)
    assert confint.shape == (N_PERIODS, 2)


def test_naive_flat_forecast():
    """Every forecast period must equal the last training observation."""
    s = _series()
    last_train = float(s.iloc[-(N_PERIODS + 1)])
    fc, _ = naive_forecast(s, N_PERIODS)
    np.testing.assert_allclose(fc, last_train, rtol=1e-5)


def test_naive_confint_ordered():
    """Lower bound must be ≤ upper bound everywhere."""
    s = _series()
    _, confint = naive_forecast(s, N_PERIODS)
    assert (confint[:, 0] <= confint[:, 1]).all()


def test_naive_interval_widens_with_horizon():
    """Prediction interval should be non-decreasing with horizon."""
    s = _series()
    _, confint = naive_forecast(s, 10)
    widths = confint[:, 1] - confint[:, 0]
    assert (np.diff(widths) >= -1e-6).all(), "Intervals should not narrow with horizon"


# ---------------------------------------------------------------------------
# seasonal_naive_forecast
# ---------------------------------------------------------------------------


def test_seasonal_naive_output_shapes():
    s = _series()
    fc, confint = seasonal_naive_forecast(s, N_PERIODS, season_length=4)
    assert fc.shape == (N_PERIODS,)
    assert confint.shape == (N_PERIODS, 2)


def test_seasonal_naive_formula():
    """fc[h-1] must equal train[train_len - season_length + (h-1) % season_length]."""
    m = 4
    s = pd.Series(np.arange(20, dtype=float))
    n_periods = 4
    train = s.iloc[:-n_periods].values
    train_len = len(train)

    fc, _ = seasonal_naive_forecast(s, n_periods, season_length=m)
    for h in range(1, n_periods + 1):
        idx = train_len - m + (h - 1) % m
        idx = max(0, min(idx, train_len - 1))
        assert fc[h - 1] == pytest.approx(train[idx], rel=1e-5)


def test_seasonal_naive_season_length_1_equals_naive():
    """With season_length=1, seasonal naive degenerates to naive."""
    s = _series(seed=7)
    fc_naive, _ = naive_forecast(s, N_PERIODS)
    fc_sn, _ = seasonal_naive_forecast(s, N_PERIODS, season_length=1)
    np.testing.assert_allclose(fc_naive, fc_sn, rtol=1e-5)


def test_seasonal_naive_confint_ordered():
    s = _series()
    _, confint = seasonal_naive_forecast(s, N_PERIODS, season_length=4)
    assert (confint[:, 0] <= confint[:, 1]).all()


# ---------------------------------------------------------------------------
# drift_forecast
# ---------------------------------------------------------------------------


def test_drift_output_shapes():
    s = _series()
    fc, confint = drift_forecast(s, N_PERIODS)
    assert fc.shape == (N_PERIODS,)
    assert confint.shape == (N_PERIODS, 2)


def test_drift_formula():
    """fc[h-1] = y_last + h * (y_last - y_first) / (T-1)."""
    s = pd.Series(np.arange(20, dtype=float))
    n_periods = 4
    train = s.iloc[:-n_periods].values
    y1, yt, T = train[0], train[-1], len(train)
    slope = (yt - y1) / (T - 1)

    fc, _ = drift_forecast(s, n_periods)
    for h in range(1, n_periods + 1):
        expected = yt + h * slope
        assert fc[h - 1] == pytest.approx(expected, rel=1e-5)


def test_drift_linear_series():
    """On a perfectly linear series the drift forecast must be exact."""
    s = pd.Series(np.arange(30, dtype=float))  # 0, 1, …, 29
    n_periods = 5
    fc, _ = drift_forecast(s, n_periods)
    expected = np.arange(25, 30, dtype=float)  # 25, 26, 27, 28, 29
    np.testing.assert_allclose(fc, expected, rtol=1e-5)


def test_drift_confint_ordered():
    s = _series()
    _, confint = drift_forecast(s, N_PERIODS)
    assert (confint[:, 0] <= confint[:, 1]).all()

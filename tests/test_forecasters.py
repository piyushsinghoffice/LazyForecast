"""Smoke tests for src/lazyforecast/forecasters.py."""

import numpy as np
import pandas as pd
import pytest
import torch

from lazyforecast.forecasters import arima_forecast, deep_forecast
from lazyforecast.models import TorchMLP

N_PERIODS = 5
N_STEPS = 5


@pytest.fixture
def series() -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(np.cumsum(rng.normal(0, 1, 80)))


def test_arima_forecast_shapes(series):
    fc, confint = arima_forecast(series, n_periods=N_PERIODS)
    assert fc.shape == (N_PERIODS,)
    assert confint.shape == (N_PERIODS, 2)


def test_arima_forecast_confint_ordered(series):
    """Lower bound must be <= upper bound for every period."""
    _, confint = arima_forecast(series, n_periods=N_PERIODS)
    assert (confint[:, 0] <= confint[:, 1]).all()


def test_deep_forecast_mlp_shapes(series):
    device = torch.device("cpu")

    def factory():
        return TorchMLP(n_steps=N_STEPS, n_features=1, hidden=16)

    fc, confint = deep_forecast(
        model_factory=factory,
        data=series,
        n_periods=N_PERIODS,
        n_steps=N_STEPS,
        n_members=2,
        device=device,
        epochs=5,
        batch_size=16,
        lr=1e-3,
        verbose=False,
        random_state=42,
    )
    assert fc.shape == (N_PERIODS,)
    assert confint.shape == (N_PERIODS, 2)


def test_deep_forecast_reproducible(series):
    """Same random_state must produce identical forecasts."""
    device = torch.device("cpu")

    def factory():
        return TorchMLP(n_steps=N_STEPS, n_features=1, hidden=16)

    kwargs = dict(
        model_factory=factory,
        data=series,
        n_periods=N_PERIODS,
        n_steps=N_STEPS,
        n_members=2,
        device=device,
        epochs=5,
        batch_size=16,
        lr=1e-3,
        verbose=False,
        random_state=99,
    )
    fc1, _ = deep_forecast(**kwargs)
    fc2, _ = deep_forecast(**kwargs)
    np.testing.assert_array_almost_equal(fc1, fc2, decimal=4)

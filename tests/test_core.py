"""Tests for src/lazyforecast/core.py."""

import numpy as np
import pandas as pd
import pytest

from lazyforecast import ForecastResult, LazyForecast

# Use tiny hyperparameters so tests finish quickly
FAST_KWARGS = dict(
    n_periods=3,
    n_steps=5,
    n_members=2,
    epochs=3,
    batch_size=16,
    lr=1e-3,
    device="cpu",
    random_state=42,
)

ALL_MODEL_NAMES = {
    "ARIMA",
    "MLP",
    "VANILLA LSTM",
    "STACKED LSTM",
    "BIDIRECTIONAL LSTM",
    "RNN",
    "GRU",
}


@pytest.fixture
def lf() -> LazyForecast:
    return LazyForecast(**FAST_KWARGS)


@pytest.fixture
def series() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(np.cumsum(rng.normal(0, 1, 60)))


def test_fit_returns_forecast_result(lf, series):
    result = lf.fit(series)
    assert isinstance(result, ForecastResult)
    assert isinstance(result.eval_table, pd.DataFrame)
    assert isinstance(result.predictions, dict)
    assert isinstance(result.confidence, dict)
    assert isinstance(result.best_model, str)


def test_fit_eval_table_has_all_models(lf, series):
    result = lf.fit(series)
    assert set(result.eval_table.index) == ALL_MODEL_NAMES


def test_fit_stores_result_on_self(lf, series):
    result = lf.fit(series)
    assert lf.result_ is result


def test_fit_prediction_shapes(lf, series):
    result = lf.fit(series)
    for name, fc in result.predictions.items():
        assert fc.shape == (FAST_KWARGS["n_periods"],), f"{name}: wrong shape"
    for name, ci in result.confidence.items():
        assert ci.shape == (FAST_KWARGS["n_periods"], 2), f"{name}: wrong confint shape"


def test_predict_raises_before_fit():
    lf_fresh = LazyForecast(**FAST_KWARGS)
    with pytest.raises(RuntimeError, match="fit()"):
        lf_fresh.predict(pd.Series(np.arange(20.0)))


def test_plot_raises_before_fit():
    lf_fresh = LazyForecast(**FAST_KWARGS)
    with pytest.raises(RuntimeError, match="fit()"):
        lf_fresh.plot()


def test_fit_reproducible_with_random_state(series):
    """Two fits with the same seed must produce identical eval tables."""
    result1 = LazyForecast(**FAST_KWARGS).fit(series)
    result2 = LazyForecast(**FAST_KWARGS).fit(series)
    pd.testing.assert_frame_equal(result1.eval_table, result2.eval_table)


def test_fit_validation_n_periods_zero():
    with pytest.raises(ValueError, match="n_periods"):
        LazyForecast(**{**FAST_KWARGS, "n_periods": 0}).fit(pd.Series(np.arange(50.0)))


def test_fit_validation_series_too_short():
    lf_val = LazyForecast(**FAST_KWARGS)
    tiny = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="too short"):
        lf_val.fit(tiny)


def test_fit_with_dataframe(lf):
    df = pd.DataFrame({"Close": np.cumsum(np.random.default_rng(3).normal(0, 1, 60))})
    result = lf.fit(df, target_col="Close")
    assert isinstance(result, ForecastResult)


def test_fit_with_ndarray(lf):
    arr = np.cumsum(np.random.default_rng(4).normal(0, 1, 60))
    result = lf.fit(arr)
    assert isinstance(result, ForecastResult)

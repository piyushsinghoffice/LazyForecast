"""Tests for src/lazyforecast/core.py."""

import numpy as np
import pandas as pd
import pytest

from lazyforecast import CVResult, ForecastResult, LazyForecast

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

# v0.2.0 model registry: 7 deep/ARIMA + 3 baselines
ALL_MODEL_NAMES = {
    "ARIMA",
    "MLP",
    "VANILLA LSTM",
    "STACKED LSTM",
    "BIDIRECTIONAL LSTM",
    "RNN",
    "GRU",
    "NAIVE",
    "SEASONAL NAIVE",
    "DRIFT",
}


@pytest.fixture
def lf() -> LazyForecast:
    return LazyForecast(**FAST_KWARGS)


@pytest.fixture
def series() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(np.cumsum(rng.normal(0, 1, 60)))


# ---------------------------------------------------------------------------
# Basic fit contract
# ---------------------------------------------------------------------------


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


def test_fit_stores_training_data(lf, series):
    result = lf.fit(series)
    assert result.training_data is not None
    assert len(result.training_data) == len(series)


def test_fit_eval_table_has_coverage_column(lf, series):
    """coverage metric must be present in eval table (always computed)."""
    result = lf.fit(series)
    assert "coverage" in result.eval_table.columns


# ---------------------------------------------------------------------------
# predict / plot guards
# ---------------------------------------------------------------------------


def test_predict_raises_before_fit():
    lf_fresh = LazyForecast(**FAST_KWARGS)
    with pytest.raises(RuntimeError, match="fit()"):
        lf_fresh.predict(pd.Series(np.arange(20.0)))


def test_plot_raises_before_fit():
    lf_fresh = LazyForecast(**FAST_KWARGS)
    with pytest.raises(RuntimeError, match="fit()"):
        lf_fresh.plot()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_fit_reproducible_with_random_state(series):
    """Two fits with the same seed must produce identical eval tables."""
    result1 = LazyForecast(**FAST_KWARGS).fit(series)
    result2 = LazyForecast(**FAST_KWARGS).fit(series)
    pd.testing.assert_frame_equal(result1.eval_table, result2.eval_table)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Feature 1: Rolling-origin CV
# ---------------------------------------------------------------------------


def test_fit_cv_rolling_returns_cvresult(series):
    lf = LazyForecast(**FAST_KWARGS)
    cv_result = lf.fit(series, cv="rolling", n_splits=2, horizon=3)
    assert isinstance(cv_result, CVResult)


def test_fit_cv_stores_cv_result(series):
    lf = LazyForecast(**FAST_KWARGS)
    lf.fit(series, cv="rolling", n_splits=2, horizon=3)
    assert lf.cv_result_ is not None
    assert isinstance(lf.cv_result_, CVResult)


def test_fit_cv_correct_n_splits(series):
    lf = LazyForecast(**FAST_KWARGS)
    result = lf.fit(series, cv="rolling", n_splits=2, horizon=3)
    assert result.n_splits == 2


def test_fit_cv_mean_metrics_has_models(series):
    lf = LazyForecast(**FAST_KWARGS)
    result = lf.fit(series, cv="rolling", n_splits=2, horizon=3)
    # At least the baselines must appear (fast; deep models may vary)
    for name in ("NAIVE", "DRIFT"):
        assert name in result.mean_metrics.index


# ---------------------------------------------------------------------------
# Feature 3: Conformal intervals
# ---------------------------------------------------------------------------


def test_fit_conformal_returns_forecast_result(series):
    """Series long enough for conformal (> 2*n_periods + n_steps + 1 = 14)."""
    lf = LazyForecast(**FAST_KWARGS)
    result = lf.fit(series, interval="conformal", alpha=0.1)
    assert isinstance(result, ForecastResult)


def test_fit_conformal_intervals_ordered(series):
    """Conformal CIs must have lower ≤ upper for all models and periods."""
    lf = LazyForecast(**FAST_KWARGS)
    result = lf.fit(series, interval="conformal", alpha=0.1)
    for name, ci in result.confidence.items():
        assert (ci[:, 0] <= ci[:, 1]).all(), f"{name}: conformal interval inverted"


def test_fit_conformal_symmetric(series):
    """For any model the conformal interval must be symmetric around the forecast."""
    lf = LazyForecast(**FAST_KWARGS)
    result = lf.fit(series, interval="conformal", alpha=0.1)
    for name in result.predictions:
        fc = result.predictions[name]
        ci = result.confidence[name]
        half_width_lo = fc - ci[:, 0]
        half_width_hi = ci[:, 1] - fc
        np.testing.assert_allclose(half_width_lo, half_width_hi, rtol=1e-4)


def test_fit_conformal_too_short_raises():
    """Series long enough for standard fit but too short for conformal raises ValueError.

    FAST_KWARGS: n_periods=3, n_steps=5
    - standard min_len = 3+5+1 = 9   → need len > 9
    - conformal min_len = 2*3+5+1 = 12 → need len > 12; len=12 triggers the check
    """
    lf = LazyForecast(**FAST_KWARGS)
    s = pd.Series(np.arange(12.0))  # 12 > 9 (passes standard) but 12 <= 12 (fails conformal)
    with pytest.raises(ValueError, match="Conformal"):
        lf.fit(s, interval="conformal")

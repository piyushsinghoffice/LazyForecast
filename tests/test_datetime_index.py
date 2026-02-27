"""Unit tests for DatetimeIndex support in utils.infer_forecast_index."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lazyforecast.utils import infer_forecast_index

# ---------------------------------------------------------------------------
# infer_forecast_index
# ---------------------------------------------------------------------------


class TestInferForecastIndex:
    def test_monthly_continuation(self):
        idx = pd.date_range("2020-01-01", periods=24, freq="MS")
        future = infer_forecast_index(idx, n_periods=6)
        assert isinstance(future, pd.DatetimeIndex)
        assert len(future) == 6
        # First future date = one month after the last observed
        assert future[0] == pd.Timestamp("2022-01-01")

    def test_daily_continuation(self):
        idx = pd.date_range("2023-01-01", periods=30, freq="D")
        future = infer_forecast_index(idx, n_periods=7)
        assert isinstance(future, pd.DatetimeIndex)
        assert len(future) == 7
        assert future[0] == pd.Timestamp("2023-01-31")

    def test_no_overlap_with_original(self):
        """Every future date must be strictly after the last observed date."""
        idx = pd.date_range("2020-01-01", periods=12, freq="MS")
        future = infer_forecast_index(idx, n_periods=3)
        assert (future > idx[-1]).all()

    def test_fallback_non_datetime(self):
        """RangeIndex input must produce a RangeIndex starting after last position."""
        idx = pd.RangeIndex(50)
        future = infer_forecast_index(idx, n_periods=5)
        assert isinstance(future, pd.RangeIndex)
        assert list(future) == [50, 51, 52, 53, 54]

    def test_fallback_irregular_datetimes(self):
        """Irregular DatetimeIndex (no inferable freq) must fall back to RangeIndex."""
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-03", "2020-01-10"])
        future = infer_forecast_index(idx, n_periods=3)
        assert isinstance(future, pd.RangeIndex)
        assert len(future) == 3

    def test_weekly_continuation(self):
        idx = pd.date_range("2021-01-04", periods=10, freq="W-MON")
        future = infer_forecast_index(idx, n_periods=4)
        assert isinstance(future, pd.DatetimeIndex)
        assert len(future) == 4
        # Each step should be 7 days apart
        deltas = (future[1:] - future[:-1]).days
        assert (deltas == 7).all()

    def test_integer_index_fallback(self):
        idx = pd.Index([10, 20, 30, 40, 50])
        future = infer_forecast_index(idx, n_periods=3)
        # Non-DatetimeIndex → RangeIndex
        assert isinstance(future, pd.RangeIndex)
        assert len(future) == 3


# ---------------------------------------------------------------------------
# Integration: DatetimeIndex preserved in ForecastResult.forecast_index
# ---------------------------------------------------------------------------


def test_fit_preserves_monthly_forecast_index():
    """When a monthly Series is passed to LazyForecast.fit, forecast_index
    must be a DatetimeIndex with monthly frequency."""
    from lazyforecast import LazyForecast

    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-01", periods=60, freq="MS")
    s = pd.Series(np.cumsum(rng.normal(0, 1, 60)), index=idx)

    lf = LazyForecast(
        n_periods=3, n_steps=5, n_members=1, epochs=2, batch_size=16, device="cpu"
    )
    result = lf.fit(s)

    assert result.forecast_index is not None
    assert isinstance(result.forecast_index, pd.DatetimeIndex)
    assert len(result.forecast_index) == 3
    # All future dates must be after the last observed date
    assert (result.forecast_index > s.index[-1]).all()


def test_fit_fallback_rangeidx_for_no_freq():
    """A Series with a non-datetime index must produce a RangeIndex."""
    from lazyforecast import LazyForecast

    rng = np.random.default_rng(7)
    s = pd.Series(np.cumsum(rng.normal(0, 1, 60)))  # default RangeIndex

    lf = LazyForecast(
        n_periods=3, n_steps=5, n_members=1, epochs=2, batch_size=16, device="cpu"
    )
    result = lf.fit(s)
    # forecast_index is None when input has no DatetimeIndex
    assert result.forecast_index is None

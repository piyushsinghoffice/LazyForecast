"""Unit tests for src/lazyforecast/validation.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazyforecast.validation import CVResult, make_splits, run_cv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0, 1, n)))


def _naive_fn(n_periods: int):
    from lazyforecast.baselines import naive_forecast

    def fn(data: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        return naive_forecast(data, n_periods)

    return fn


# ---------------------------------------------------------------------------
# make_splits
# ---------------------------------------------------------------------------


class TestMakeSplits:
    def test_correct_number_of_folds(self):
        s = _series(100)
        splits = make_splits(s, n_splits=4, horizon=10, step=10)
        assert len(splits) == 4

    def test_test_size_equals_horizon(self):
        s = _series(100)
        horizon = 8
        splits = make_splits(s, n_splits=3, horizon=horizon, step=10)
        for train, test in splits:
            assert len(test) == horizon

    def test_no_leakage(self):
        """Train and test windows must not overlap — tested via integer index."""
        s = _series(100)
        splits = make_splits(s, n_splits=4, horizon=10, step=10)
        for train, test in splits:
            assert int(train.index[-1]) < int(test.index[0])

    def test_expanding_train_window(self):
        """Training windows should grow fold-by-fold (expanding window)."""
        s = _series(120)
        splits = make_splits(s, n_splits=5, horizon=5, step=5)
        train_sizes = [len(tr) for tr, _ in splits]
        assert train_sizes == sorted(train_sizes)

    def test_step_none_auto(self):
        """With step=None the function should still produce the right number of folds."""
        s = _series(100)
        splits = make_splits(s, n_splits=5, horizon=10)
        assert len(splits) == 5

    def test_insufficient_data_raises(self):
        tiny = _series(15)
        with pytest.raises(ValueError, match="Not enough data"):
            make_splits(tiny, n_splits=10, horizon=10, step=5)

    def test_invalid_n_splits_raises(self):
        with pytest.raises(ValueError, match="n_splits"):
            make_splits(_series(), n_splits=0, horizon=5)

    def test_invalid_horizon_raises(self):
        with pytest.raises(ValueError, match="horizon"):
            make_splits(_series(), n_splits=3, horizon=0)


# ---------------------------------------------------------------------------
# run_cv
# ---------------------------------------------------------------------------


class TestRunCV:
    def _forecasters(self, horizon: int) -> dict:
        return {
            "NAIVE": _naive_fn(horizon),
        }

    def test_returns_cvresult(self):
        s = _series(100)
        h = 5
        result = run_cv(
            data=s,
            forecasters=self._forecasters(h),
            n_splits=3,
            horizon=h,
        )
        assert isinstance(result, CVResult)

    def test_n_splits_stored(self):
        s = _series(100)
        h = 5
        result = run_cv(
            data=s,
            forecasters=self._forecasters(h),
            n_splits=3,
            horizon=h,
        )
        assert result.n_splits == 3

    def test_horizon_stored(self):
        s = _series(100)
        h = 7
        result = run_cv(
            data=s,
            forecasters=self._forecasters(h),
            n_splits=3,
            horizon=h,
        )
        assert result.horizon == h

    def test_fold_metrics_count(self):
        s = _series(100)
        h = 5
        result = run_cv(
            data=s,
            forecasters=self._forecasters(h),
            n_splits=4,
            horizon=h,
        )
        assert len(result.fold_metrics) == 4

    def test_mean_std_metrics_shapes_match(self):
        s = _series(100)
        h = 5
        result = run_cv(
            data=s,
            forecasters=self._forecasters(h),
            n_splits=3,
            horizon=h,
        )
        assert result.mean_metrics.shape == result.std_metrics.shape

    def test_mean_metrics_model_in_index(self):
        s = _series(100)
        h = 5
        result = run_cv(
            data=s,
            forecasters=self._forecasters(h),
            n_splits=3,
            horizon=h,
        )
        assert "NAIVE" in result.mean_metrics.index

    def test_metric_aggregation_correctness(self):
        """mean_metrics must equal the manual mean over fold DataFrames."""
        s = _series(120)
        h = 5
        result = run_cv(
            data=s,
            forecasters=self._forecasters(h),
            n_splits=4,
            horizon=h,
        )
        # Manually compute mean rmse across folds
        rmse_per_fold = [float(fm.loc["NAIVE", "rmse"]) for fm in result.fold_metrics]
        expected_mean_rmse = float(np.mean(rmse_per_fold))
        actual_mean_rmse = float(result.mean_metrics.loc["NAIVE", "rmse"])
        assert actual_mean_rmse == pytest.approx(expected_mean_rmse, rel=1e-5)

    def test_std_metrics_non_negative(self):
        s = _series(120)
        h = 5
        result = run_cv(
            data=s,
            forecasters=self._forecasters(h),
            n_splits=4,
            horizon=h,
        )
        assert (result.std_metrics >= -1e-8).all().all()

    def test_mean_metrics_sorted_best_first(self):
        """With multiple models, mean_metrics must be sorted from best to worst."""
        from lazyforecast.baselines import drift_forecast, naive_forecast

        h = 5
        s = _series(120)

        forecasters = {
            "NAIVE": lambda data: naive_forecast(data, h),
            "DRIFT": lambda data: drift_forecast(data, h),
        }
        result = run_cv(data=s, forecasters=forecasters, n_splits=3, horizon=h)
        # Verify ascending RMSE (or at least that the table is sorted)
        rmse_vals = result.mean_metrics["rmse"].values
        # Should be non-decreasing (best = lowest RMSE first when MDA ties)
        assert list(rmse_vals) == sorted(rmse_vals) or len(rmse_vals) <= 1

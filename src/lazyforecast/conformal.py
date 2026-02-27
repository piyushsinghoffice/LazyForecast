"""Split-conformal prediction intervals for any point forecaster."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def compute_conformal_quantile(
    cal_data: pd.Series,
    forecaster_fn: Callable[[pd.Series], tuple[np.ndarray, np.ndarray]],
    n_periods: int,
    alpha: float,
) -> float:
    """Compute the empirical conformal quantile from a calibration window.

    Strategy (split conformal):
    - ``cal_data`` is the training data (i.e. data[:-n_periods]).
    - We call ``forecaster_fn(cal_data)`` which forecasts ``cal_data[-n_periods:]``
      (the last n_periods of cal_data serve as the calibration test window).
    - Residuals = |calibration_actual - calibration_forecast|.
    - Return the (1 - alpha) empirical quantile of those residuals.

    Parameters
    ----------
    cal_data      : pd.Series of length N - n_periods (training data only)
    forecaster_fn : Callable[[pd.Series], (fc, confint)] with n_periods baked in
    n_periods     : forecast horizon
    alpha         : miscoverage level (e.g. 0.1 for 90% coverage)

    Returns
    -------
    float — conformal quantile q̂
    """
    cal_fc, _ = forecaster_fn(cal_data)
    cal_actual = cal_data.iloc[-n_periods:].values.astype(np.float64)
    residuals = np.abs(cal_actual - cal_fc.astype(np.float64))
    q = float(np.quantile(residuals, 1.0 - alpha))
    return q


def apply_conformal_interval(
    fc: np.ndarray,
    q: float,
) -> np.ndarray:
    """Build symmetric conformal prediction interval [ŷ - q, ŷ + q].

    Parameters
    ----------
    fc : point forecasts of shape (n_periods,)
    q  : conformal quantile

    Returns
    -------
    np.ndarray of shape (n_periods, 2) — [lower, upper]
    """
    fc = np.asarray(fc, dtype=np.float32)
    lower = fc - q
    upper = fc + q
    return np.column_stack([lower, upper]).astype(np.float32)

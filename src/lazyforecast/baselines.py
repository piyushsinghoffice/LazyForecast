"""Baseline forecasters: Naive, Seasonal Naive, Drift."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _naive_prediction_interval(
    train: np.ndarray,
    fc: np.ndarray,
) -> np.ndarray:
    """Symmetric prediction interval: ŷ ± 1.96 * σ * sqrt(h).

    σ = std of first differences of the training series.
    h = 1, 2, ..., n_periods (horizon index).
    """
    sigma = float(np.std(np.diff(train))) if len(train) > 1 else 1.0
    if sigma == 0.0:
        sigma = 1.0
    n = len(fc)
    half_width = 1.96 * sigma * np.sqrt(np.arange(1, n + 1, dtype=np.float64))
    lower = fc - half_width
    upper = fc + half_width
    return np.column_stack([lower, upper]).astype(np.float32)


def naive_forecast(
    data: pd.Series,
    n_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Random-walk (flat) forecast: ŷ_{t+h} = y_t for all h.

    Parameters
    ----------
    data      : full series (train + test)
    n_periods : forecast horizon / test window size

    Returns
    -------
    forecasts  : np.ndarray of shape (n_periods,)
    confidence : np.ndarray of shape (n_periods, 2) — [lower, upper]
    """
    train = data.iloc[:-n_periods].values.astype(np.float64)
    last = train[-1]
    fc = np.full(n_periods, last, dtype=np.float64)
    confint = _naive_prediction_interval(train, fc)
    return fc.astype(np.float32), confint


def seasonal_naive_forecast(
    data: pd.Series,
    n_periods: int,
    season_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Seasonal random-walk forecast: ŷ_{t+h} = y_{t+h-m}.

    Repeats the most recent full season cyclically.

    Parameters
    ----------
    data          : full series (train + test)
    n_periods     : forecast horizon / test window size
    season_length : seasonal period m (e.g., 12 for monthly, 7 for daily-weekly)

    Returns
    -------
    forecasts  : np.ndarray of shape (n_periods,)
    confidence : np.ndarray of shape (n_periods, 2) — [lower, upper]
    """
    train = data.iloc[:-n_periods].values.astype(np.float64)
    train_len = len(train)

    # For h = 1, …, n_periods:  idx = (train_len - season_length) + (h-1) % season_length
    fc = np.empty(n_periods, dtype=np.float64)
    for h in range(1, n_periods + 1):
        idx = train_len - season_length + (h - 1) % season_length
        # Clamp to valid range in case train is shorter than season_length
        idx = max(0, min(idx, train_len - 1))
        fc[h - 1] = train[idx]

    confint = _naive_prediction_interval(train, fc)
    return fc.astype(np.float32), confint


def drift_forecast(
    data: pd.Series,
    n_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Drift forecast: linear extrapolation from first to last training observation.

    ŷ_{t+h} = y_t + h * (y_t - y_1) / (T - 1)

    Parameters
    ----------
    data      : full series (train + test)
    n_periods : forecast horizon / test window size

    Returns
    -------
    forecasts  : np.ndarray of shape (n_periods,)
    confidence : np.ndarray of shape (n_periods, 2) — [lower, upper]
    """
    train = data.iloc[:-n_periods].values.astype(np.float64)
    y1 = train[0]
    yt = train[-1]
    T = len(train)
    slope = (yt - y1) / (T - 1) if T > 1 else 0.0

    h_vals = np.arange(1, n_periods + 1, dtype=np.float64)
    fc = yt + slope * h_vals
    confint = _naive_prediction_interval(train, fc)
    return fc.astype(np.float32), confint

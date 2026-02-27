"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def mean_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of periods where direction of change is correctly forecast.

    Parameters
    ----------
    actual    : 1-D array of true values
    predicted : 1-D array of predicted values (same length as actual)

    Returns
    -------
    float in [0, 1] — 1.0 means perfect directional accuracy
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    actual_signs = np.sign(np.diff(actual))
    predicted_signs = np.sign(np.diff(predicted))

    return float(np.sum(actual_signs == predicted_signs) / len(actual_signs))


def interval_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of actual values that fall within [lower, upper].

    Parameters
    ----------
    actual : 1-D array of true values
    lower  : 1-D array of lower confidence bounds
    upper  : 1-D array of upper confidence bounds

    Returns
    -------
    float in [0, 1]
    """
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def forecast_accuracy(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
    confint: np.ndarray | None = None,
) -> dict:
    """Compute a standard suite of regression / forecast metrics.

    Parameters
    ----------
    actual     : 1-D array-like of true values
    predicted  : 1-D array-like of predicted values
    model_name : display name for this row
    confint    : optional (n_periods, 2) array — if provided, adds a
                 ``coverage`` key (fraction of actuals within interval)

    Returns
    -------
    dict with keys: model, mda, rmse, mape, R2, mae, corr[, coverage]
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mda = mean_directional_accuracy(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mape = float(mean_absolute_percentage_error(actual, predicted))
    r2 = float(r2_score(actual, predicted))
    mae = float(mean_absolute_error(actual, predicted))
    corr = float(np.corrcoef(actual, predicted)[0, 1])

    result: dict = {
        "model": model_name,
        "mda": mda,
        "rmse": rmse,
        "mape": mape,
        "R2": r2,
        "mae": mae,
        "corr": corr,
    }

    if confint is not None:
        confint = np.asarray(confint, dtype=float)
        result["coverage"] = interval_coverage(actual, confint[:, 0], confint[:, 1])

    return result


def build_eval_table(eval_data: list[dict]) -> pd.DataFrame:
    """Build an evaluation DataFrame sorted by [mda desc, rmse asc, mape asc, R2 desc, mae asc].

    Parameters
    ----------
    eval_data : list of dicts, each produced by :func:`forecast_accuracy`

    Returns
    -------
    pd.DataFrame indexed by model name, sorted from best to worst
    """
    df = pd.DataFrame(eval_data).set_index("model")
    df.sort_values(
        by=["mda", "rmse", "mape", "R2", "mae"],
        ascending=[False, True, True, False, True],
        inplace=True,
    )
    return df

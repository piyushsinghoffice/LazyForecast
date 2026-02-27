"""Rolling-origin (expanding-window) cross-validation for time-series models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .metrics import build_eval_table, forecast_accuracy


@dataclass
class CVResult:
    """Output of :func:`run_cv`.

    Attributes
    ----------
    fold_metrics : one ranked eval DataFrame per fold
    mean_metrics : metrics averaged across folds, sorted best-first
    std_metrics  : standard deviation of metrics across folds
    n_splits     : number of folds actually computed
    horizon      : forecast horizon used in each fold
    """

    fold_metrics: list[pd.DataFrame] = field(default_factory=list)
    mean_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    std_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_splits: int = 0
    horizon: int = 0


def make_splits(
    data: pd.Series,
    n_splits: int,
    horizon: int,
    step: int | None = None,
) -> list[tuple[pd.Series, pd.Series]]:
    """Generate (train, test) pairs for rolling-origin cross-validation.

    Training windows expand over time (no data leakage); each test window
    is exactly ``horizon`` steps long.

    Parameters
    ----------
    data     : full series to split
    n_splits : number of folds
    horizon  : number of test steps per fold
    step     : gap between consecutive fold cut-points.
               If None, evenly distributes folds over the available range.

    Returns
    -------
    list of (train, test) pd.Series tuples, ordered earliest → latest
    """
    n = len(data)
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    if step is None:
        # Spread folds evenly: last fold ends at n, first fold cut-point is
        # roughly at n - (n_splits * step) - horizon.
        available = n - horizon
        step = max(1, available // (n_splits + 1))

    splits: list[tuple[pd.Series, pd.Series]] = []
    for i in range(n_splits):
        # Most-recent fold ends at n; earlier folds end earlier.
        test_end = n - (n_splits - 1 - i) * step
        test_start = test_end - horizon

        if test_start <= 0:
            raise ValueError(
                f"Not enough data for {n_splits} splits with "
                f"horizon={horizon}, step={step}. "
                f"Series length: {n}. "
                f"Fold {i + 1} would need test_start={test_start}."
            )

        train = data.iloc[:test_start]
        test = data.iloc[test_start:test_end]
        splits.append((train, test))

    return splits


def run_cv(
    data: pd.Series,
    forecasters: dict[str, Callable[[pd.Series], tuple[np.ndarray, np.ndarray]]],
    n_splits: int,
    horizon: int,
    step: int | None = None,
) -> CVResult:
    """Run rolling-origin cross-validation for all provided forecasters.

    For each fold the forecaster is called with ``fold_data`` (train + test
    concatenated, reset index), where the last ``horizon`` points are the
    test window — exactly how each forecaster already works.

    Parameters
    ----------
    data        : full series
    forecasters : dict of ``{name: callable(data) → (fc, confint)}``
                  with the horizon baked into the callable
    n_splits    : number of CV folds
    horizon     : forecast horizon per fold
    step        : step between fold cut-points (None = auto)

    Returns
    -------
    CVResult
    """
    splits = make_splits(data, n_splits=n_splits, horizon=horizon, step=step)
    fold_metrics: list[pd.DataFrame] = []

    for train, test in splits:
        # Concatenate so the forecaster sees train context + test horizon
        fold_data = pd.concat([train, test]).reset_index(drop=True)
        actual = test.values
        eval_data: list[dict] = []

        for name, fn in forecasters.items():
            try:
                fc, confint = fn(fold_data)
                metrics = forecast_accuracy(actual, fc, name, confint=confint)
                eval_data.append(metrics)
            except Exception:
                # Skip this model for this fold rather than aborting the whole CV
                pass

        if eval_data:
            fold_metrics.append(build_eval_table(eval_data))

    if not fold_metrics:
        raise RuntimeError(
            "No fold metrics were computed. "
            "Check that the series is long enough for the requested splits."
        )

    # Aggregate across folds
    stacked = pd.concat(fold_metrics)
    numeric_cols = stacked.select_dtypes(include=np.number).columns

    mean_metrics = stacked.groupby(stacked.index)[numeric_cols].mean()
    std_metrics = stacked.groupby(stacked.index)[numeric_cols].std(ddof=1).fillna(0.0)

    # Sort mean_metrics by the same criteria as build_eval_table
    sort_cols = [c for c in ["mda", "rmse", "mape", "R2", "mae"] if c in mean_metrics.columns]
    sort_asc = [c in ("rmse", "mape", "mae") for c in sort_cols]
    mean_metrics = mean_metrics.sort_values(by=sort_cols, ascending=sort_asc)
    std_metrics = std_metrics.loc[mean_metrics.index]

    return CVResult(
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        n_splits=len(splits),
        horizon=horizon,
    )

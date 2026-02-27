"""Utility functions for data preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_sequence(sequence: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised sliding window using np.lib.stride_tricks.

    Parameters
    ----------
    sequence : np.ndarray of shape (n,)
    n_steps  : int — lookback window size

    Returns
    -------
    X : np.ndarray of shape (n - n_steps, n_steps)
    y : np.ndarray of shape (n - n_steps,)
    """
    windows = np.lib.stride_tricks.sliding_window_view(sequence, window_shape=n_steps + 1)
    X = windows[:, :n_steps].astype(np.float32)
    y = windows[:, n_steps].astype(np.float32)
    return X, y


def prepare_series(
    data: np.ndarray | pd.Series | pd.DataFrame,
    target_col: str | None = None,
) -> tuple[pd.Series, pd.Series | None]:
    """Normalise input into a pd.Series with NaNs interpolated.

    Parameters
    ----------
    data       : array-like or DataFrame
    target_col : column name when data is a DataFrame

    Returns
    -------
    y_values   : pd.Series — the target values
    x_values   : pd.Series | None — index / x-axis column if present in DataFrame
    """
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            y_values = pd.Series(data.astype(float))
            x_values = None
        elif data.ndim == 2 and data.shape[1] == 1:
            y_values = pd.Series(data.flatten().astype(float))
            x_values = None
        else:
            raise ValueError(
                "When passing a numpy array, it must be 1-D or a single-column 2-D array. "
                f"Got shape {data.shape}. Pass a DataFrame with target_col to specify the column."
            )
    elif isinstance(data, pd.Series):
        y_values = data.reset_index(drop=True).astype(float)
        x_values = None
    elif isinstance(data, pd.DataFrame):
        if target_col is None:
            raise ValueError(
                "When passing a DataFrame you must specify target_col (the column to forecast)."
            )
        if target_col not in data.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in DataFrame columns: {list(data.columns)}"
            )
        col = data[target_col]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        y_values = pd.Series(col.values.astype(float)).reset_index(drop=True)
        x_values = None
    else:
        raise ValueError(
            f"Unsupported data type: {type(data)}. "
            "Pass a pd.Series, pd.DataFrame, or np.ndarray."
        )

    if y_values.isnull().any():
        y_values = y_values.interpolate(method="linear").bfill().ffill()

    return y_values, x_values

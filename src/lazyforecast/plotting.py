"""Standalone forecast plotting utility."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple


def plot_forecast(
    data: pd.Series,
    fc: np.ndarray,
    confint: np.ndarray,
    model_name: str,
    n_periods: int,
    x_values=None,
    x_label: str | None = None,
    y_label: str | None = None,
) -> matplotlib.figure.Figure:
    """Create a forecast plot and return the Figure (does NOT call plt.show()).

    Parameters
    ----------
    data       : full series of observed values
    fc         : forecast array of shape (n_periods,)
    confint    : confidence array of shape (n_periods, 2) — [lower, upper]
    model_name : title label
    n_periods  : number of forecast periods
    x_values   : optional array-like for the x-axis (defaults to integer index)
    x_label    : x-axis label
    y_label    : y-axis label

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(data)
    if x_values is None:
        x_values = np.arange(n)
    else:
        x_values = np.asarray(x_values)

    index_of_fc = np.arange(n - n_periods, n)

    fc_series = pd.Series(fc.flatten(), index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Find the period with the narrowest confidence interval
    widths = confint[:, 1] - confint[:, 0]
    lowest_intervals = np.where(widths == widths.min())[0]

    # Error bar size: 5% of data range
    data_range = float(data.max()) - float(data.min())
    yerr = 0.05 * data_range if data_range > 0 else 1.0

    fig, ax = plt.subplots(figsize=(15, 8))

    (line_actual,) = ax.plot(x_values, data.values, label="Actual")
    (line_fc,) = ax.plot(
        x_values[index_of_fc], fc_series, color="red", linestyle="--", label="Forecasted"
    )

    fill = ax.fill_between(
        x_values[lower_series.index],
        lower_series,
        upper_series,
        color=(1, 0, 0, 0.15),
        edgecolor=(1, 0, 0, 1),
        label="Confidence Interval",
    )

    low_color = "black"
    X_low = index_of_fc[lowest_intervals]
    Y_low = fc_series.iloc[lowest_intervals].values

    err_line, _, _ = ax.errorbar(
        x_values[X_low], Y_low, yerr=yerr, ls="-", color=low_color
    )
    ax.plot(
        x_values[X_low], Y_low + yerr,
        marker="v", ls="", markerfacecolor=low_color, markeredgecolor=low_color, ms=10,
    )
    ax.plot(
        x_values[X_low], Y_low - yerr,
        marker="^", ls="", markerfacecolor=low_color, markeredgecolor=low_color, ms=10,
    )

    em1, = ax.plot([], [], marker=">", ls="", markerfacecolor=low_color, markeredgecolor=low_color, ms=10)
    em2, = ax.plot([], [], marker="<", ls="", markerfacecolor=low_color, markeredgecolor=low_color, ms=10)

    ax.set_title(f"{model_name} Forecast")
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    marker_error_handler = HandlerTuple(ndivide=None, pad=-0.2)
    ax.legend(
        [line_actual, line_fc, fill, (em1, err_line, em2)],
        ["Actual", "Forecasted", "Confidence Interval", "Lowest Confidence Interval"],
        prop={"size": 18},
        handler_map={tuple: marker_error_handler},
        facecolor=(1, 1, 1, 0.5),
    )

    return fig

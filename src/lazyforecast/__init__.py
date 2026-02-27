"""LazyForecast — automatic univariate time-series forecasting."""

from .core import ForecastResult, LazyForecast
from .plotting import plot_forecast

__all__ = ["LazyForecast", "ForecastResult", "plot_forecast"]

"""LazyForecast — automatic univariate time-series forecasting."""

from .core import ForecastResult, LazyForecast
from .plotting import plot_forecast
from .validation import CVResult

__all__ = ["LazyForecast", "ForecastResult", "CVResult", "plot_forecast"]

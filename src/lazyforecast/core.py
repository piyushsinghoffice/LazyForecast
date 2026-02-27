"""Core LazyForecast class."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .forecasters import arima_forecast, deep_forecast
from .metrics import build_eval_table, forecast_accuracy
from .models import TorchMLP, TorchRNNBase
from .plotting import plot_forecast
from .utils import prepare_series


@dataclass
class ForecastResult:
    """Container for the output of :meth:`LazyForecast.fit`.

    Attributes
    ----------
    eval_table  : pd.DataFrame — all models ranked best-first
    predictions : dict[str, np.ndarray] — point forecasts per model
    confidence  : dict[str, np.ndarray] — (n_periods, 2) intervals per model
    best_model  : str — name of the top-ranked model
    """

    eval_table: pd.DataFrame
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    confidence: dict[str, np.ndarray] = field(default_factory=dict)
    best_model: str = ""


class LazyForecast:
    """Fit multiple time-series forecasters to data and rank them.

    Parameters
    ----------
    n_periods    : test window size / forecast horizon
    n_steps      : lookback window for deep models
    n_members    : ensemble size per deep model
    random_state : seeds torch + numpy for reproducibility (None = non-deterministic)
    device       : 'cpu', 'cuda', etc. — auto-detects if None
    epochs       : training epochs per ensemble member
    batch_size   : DataLoader batch size
    lr           : Adam learning rate
    verbose      : print training progress if True
    """

    def __init__(
        self,
        n_periods: int = 5,
        n_steps: int = 5,
        n_members: int = 10,
        random_state: int | None = None,
        device: str | None = None,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = False,
    ):
        self.n_periods = n_periods
        self.n_steps = n_steps
        self.n_members = n_members
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.result_: ForecastResult | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deep_kwargs(self) -> dict:
        return dict(
            n_periods=self.n_periods,
            n_steps=self.n_steps,
            n_members=self.n_members,
            device=self.device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            verbose=self.verbose,
            random_state=self.random_state,
        )

    def _build_forecasters(self) -> dict:
        """Return ordered dict of model_name → callable(data) → (fc, confint)."""
        kw = self._deep_kwargs()
        n_steps = self.n_steps

        def make_deep(factory):
            def run(data):
                return deep_forecast(model_factory=factory, data=data, **kw)
            return run

        return {
            "ARIMA": lambda data: arima_forecast(data, self.n_periods),
            "MLP": make_deep(lambda: TorchMLP(n_steps=n_steps, n_features=1, hidden=64)),
            "VANILLA LSTM": make_deep(
                lambda: TorchRNNBase("lstm", input_size=1, hidden_size=32, bidirectional=False, num_layers=1)
            ),
            "STACKED LSTM": make_deep(
                lambda: TorchRNNBase("lstm", input_size=1, hidden_size=32, bidirectional=False, num_layers=2)
            ),
            "BIDIRECTIONAL LSTM": make_deep(
                lambda: TorchRNNBase("lstm", input_size=1, hidden_size=50, bidirectional=True, num_layers=1)
            ),
            "RNN": make_deep(
                lambda: TorchRNNBase("rnn", input_size=1, hidden_size=32, bidirectional=False, num_layers=1)
            ),
            "GRU": make_deep(
                lambda: TorchRNNBase("gru", input_size=1, hidden_size=32, bidirectional=False, num_layers=1)
            ),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data,
        target_col: str | None = None,
        x_col: str | None = None,
    ) -> ForecastResult:
        """Train all forecasters and return a ranked :class:`ForecastResult`.

        Parameters
        ----------
        data       : pd.Series, pd.DataFrame, or np.ndarray
        target_col : column to forecast when data is a DataFrame
        x_col      : (unused; reserved for future exogenous support)

        Returns
        -------
        ForecastResult — also stored on ``self.result_``
        """
        if self.n_periods <= 0:
            raise ValueError(f"n_periods must be > 0, got {self.n_periods}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")

        y_values, _ = prepare_series(data, target_col=target_col)

        min_len = self.n_periods + self.n_steps + 1
        if len(y_values) <= min_len:
            raise ValueError(
                f"Series is too short: need > {min_len} observations "
                f"(n_periods={self.n_periods} + n_steps={self.n_steps} + 1), "
                f"got {len(y_values)}."
            )

        actual = y_values.iloc[-self.n_periods:].values
        forecasters = self._build_forecasters()

        eval_data: list[dict] = []
        predictions: dict[str, np.ndarray] = {}
        confidence: dict[str, np.ndarray] = {}

        for model_name, run_fn in forecasters.items():
            print(f"Fitting {model_name}...")
            fc, confint = run_fn(y_values)
            predictions[model_name] = fc
            confidence[model_name] = confint
            metrics = forecast_accuracy(actual, fc, model_name)
            eval_data.append(metrics)

        eval_table = build_eval_table(eval_data)
        best_model = str(eval_table.index[0])

        self.result_ = ForecastResult(
            eval_table=eval_table,
            predictions=predictions,
            confidence=confidence,
            best_model=best_model,
        )
        return self.result_

    def predict(
        self,
        data,
        target_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce forecasts using the best model found during :meth:`fit`.

        The new data must be at least ``n_steps`` observations long so the
        deep models can form their first input window.

        Parameters
        ----------
        data       : pd.Series, pd.DataFrame, or np.ndarray
        target_col : required when data is a DataFrame

        Returns
        -------
        (forecasts, confidence) — same shapes as :meth:`fit` outputs
        """
        if self.result_ is None:
            raise RuntimeError("Call fit() before predict().")

        y_values, _ = prepare_series(data, target_col=target_col)
        best = self.result_.best_model

        forecasters = self._build_forecasters()
        fc, confint = forecasters[best](y_values)
        return fc, confint

    def plot(
        self,
        model_name: str | None = None,
        x_col: str | None = None,
    ):
        """Plot forecast(s) using stored :attr:`result_`.

        Parameters
        ----------
        model_name : model to plot; None plots the best model
        x_col      : x-axis values (passed through to :func:`plot_forecast`)

        Returns
        -------
        matplotlib.figure.Figure | list[matplotlib.figure.Figure]
        """
        if self.result_ is None:
            raise RuntimeError("Call fit() before plot().")

        result = self.result_

        if model_name is not None:
            if model_name not in result.predictions:
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Available: {list(result.predictions.keys())}"
                )
            names = [model_name]
        else:
            names = [result.best_model]

        figs = []
        for name in names:
            fig = plot_forecast(
                data=pd.Series(list(result.predictions[name])),
                fc=result.predictions[name],
                confint=result.confidence[name],
                model_name=name,
                n_periods=self.n_periods,
                x_values=x_col,
                x_label=x_col,
            )
            figs.append(fig)

        return figs[0] if len(figs) == 1 else figs

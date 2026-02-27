"""Core LazyForecast class and ForecastResult container."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import drift_forecast, naive_forecast, seasonal_naive_forecast
from .conformal import apply_conformal_interval, compute_conformal_quantile
from .forecasters import arima_forecast, deep_forecast
from .metrics import build_eval_table, forecast_accuracy
from .models import TorchMLP, TorchRNNBase
from .plotting import plot_forecast
from .utils import infer_forecast_index, prepare_series


@dataclass
class ForecastResult:
    """Container for the output of :meth:`LazyForecast.fit`.

    Attributes
    ----------
    eval_table     : pd.DataFrame — all models ranked best-first
    predictions    : dict[str, np.ndarray] — point forecasts per model
    confidence     : dict[str, np.ndarray] — (n_periods, 2) intervals per model
    best_model     : str — name of the top-ranked model
    forecast_index : DatetimeIndex or RangeIndex for the forecast horizon
                     (None when input had no detectable datetime index)
    training_data  : original series used during fit (needed for plot / save)
    """

    eval_table: pd.DataFrame
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    confidence: dict[str, np.ndarray] = field(default_factory=dict)
    best_model: str = ""
    forecast_index: pd.Index | None = field(default=None, repr=False)
    training_data: pd.Series | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Artifact persistence  (Feature 5)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist all forecast artifacts to a directory.

        Creates the directory (and parents) if it does not exist.
        Existing files are silently overwritten.

        Output
        ------
        config.json    — metadata (best model, model list, horizon)
        metrics.csv    — ranked evaluation table
        forecast.csv   — point forecasts for every model
        intervals.csv  — confidence intervals (model/period/lower/upper)
        plot.png       — best-model forecast chart (requires ``training_data``)
        """
        import matplotlib.pyplot as plt

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        n_periods = (
            len(next(iter(self.predictions.values()))) if self.predictions else 0
        )

        # config.json
        config: dict = {
            "best_model": self.best_model,
            "models": list(self.predictions.keys()),
            "n_periods": n_periods,
        }
        (out / "config.json").write_text(json.dumps(config, indent=2))

        # metrics.csv
        self.eval_table.to_csv(out / "metrics.csv")

        # forecast.csv
        if self.predictions:
            fc_df = pd.DataFrame(self.predictions)
            if self.forecast_index is not None:
                fc_df.index = self.forecast_index[:n_periods]
            fc_df.to_csv(out / "forecast.csv")

        # intervals.csv
        if self.confidence:
            rows = []
            for model, ci in self.confidence.items():
                for i, (lo, hi) in enumerate(ci):
                    period = (
                        str(self.forecast_index[i])
                        if self.forecast_index is not None
                        else i
                    )
                    rows.append(
                        {
                            "model": model,
                            "period": period,
                            "lower": float(lo),
                            "upper": float(hi),
                        }
                    )
            pd.DataFrame(rows).to_csv(out / "intervals.csv", index=False)

        # plot.png
        if self.training_data is not None and self.best_model in self.predictions:
            fc = self.predictions[self.best_model]
            ci = self.confidence.get(
                self.best_model,
                np.zeros((n_periods, 2), dtype=np.float32),
            )
            fig = plot_forecast(
                data=self.training_data,
                fc=fc,
                confint=ci,
                model_name=self.best_model,
                n_periods=n_periods,
            )
            fig.savefig(out / "plot.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


# ---------------------------------------------------------------------------
# LazyForecast
# ---------------------------------------------------------------------------


class LazyForecast:
    """Fit multiple time-series forecasters to data and rank them.

    Parameters
    ----------
    n_periods     : test window size / forecast horizon
    n_steps       : lookback window for deep models
    n_members     : ensemble size per deep model
    random_state  : seeds torch + numpy for reproducibility (None = non-deterministic)
    device        : 'cpu', 'cuda', etc. — auto-detects if None
    epochs        : training epochs per ensemble member
    batch_size    : DataLoader batch size
    lr            : Adam learning rate
    verbose       : print training progress if True
    season_length : seasonal period for Seasonal Naive (12=monthly, 7=weekly)
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
        season_length: int = 12,
    ):
        self.n_periods = n_periods
        self.n_steps = n_steps
        self.n_members = n_members
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.season_length = season_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.result_: ForecastResult | None = None
        self.cv_result_: object | None = None  # CVResult, imported lazily

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_forecasters(
        self,
        n_periods: int | None = None,
    ) -> dict:
        """Return ``{name: Callable[[pd.Series], (fc, confint)]}`` with n_periods baked in."""
        n = n_periods if n_periods is not None else self.n_periods
        n_steps = self.n_steps
        season_length = self.season_length

        deep_kw: dict = dict(
            n_periods=n,
            n_steps=n_steps,
            n_members=self.n_members,
            device=self.device,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            verbose=self.verbose,
            random_state=self.random_state,
        )

        def make_deep(factory):
            def run(data: pd.Series) -> tuple[np.ndarray, np.ndarray]:
                return deep_forecast(model_factory=factory, data=data, **deep_kw)

            return run

        return {
            "ARIMA": lambda data: arima_forecast(data, n),
            "MLP": make_deep(
                lambda: TorchMLP(n_steps=n_steps, n_features=1, hidden=64)
            ),
            "VANILLA LSTM": make_deep(
                lambda: TorchRNNBase(
                    "lstm",
                    input_size=1,
                    hidden_size=32,
                    bidirectional=False,
                    num_layers=1,
                )
            ),
            "STACKED LSTM": make_deep(
                lambda: TorchRNNBase(
                    "lstm",
                    input_size=1,
                    hidden_size=32,
                    bidirectional=False,
                    num_layers=2,
                )
            ),
            "BIDIRECTIONAL LSTM": make_deep(
                lambda: TorchRNNBase(
                    "lstm",
                    input_size=1,
                    hidden_size=50,
                    bidirectional=True,
                    num_layers=1,
                )
            ),
            "RNN": make_deep(
                lambda: TorchRNNBase(
                    "rnn",
                    input_size=1,
                    hidden_size=32,
                    bidirectional=False,
                    num_layers=1,
                )
            ),
            "GRU": make_deep(
                lambda: TorchRNNBase(
                    "gru",
                    input_size=1,
                    hidden_size=32,
                    bidirectional=False,
                    num_layers=1,
                )
            ),
            "NAIVE": lambda data: naive_forecast(data, n),
            "SEASONAL NAIVE": lambda data: seasonal_naive_forecast(data, n, season_length),
            "DRIFT": lambda data: drift_forecast(data, n),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: np.ndarray | pd.Series | pd.DataFrame,
        target_col: str | None = None,
        # Rolling-origin CV  (Feature 1)
        cv: str | None = None,
        n_splits: int = 5,
        horizon: int | None = None,
        step: int | None = None,
        # Conformal intervals  (Feature 3)
        interval: str | None = None,
        alpha: float = 0.1,
    ) -> ForecastResult:
        """Train all forecasters and return a ranked :class:`ForecastResult`.

        Parameters
        ----------
        data       : pd.Series, pd.DataFrame, or np.ndarray
        target_col : column to forecast when data is a DataFrame
        cv         : ``"rolling"`` → rolling-origin cross-validation;
                     returns a :class:`~lazyforecast.validation.CVResult`
        n_splits   : CV folds (only used when ``cv="rolling"``)
        horizon    : CV test horizon; defaults to ``self.n_periods``
        step       : step between CV fold cut-points (None = auto)
        interval   : ``"conformal"`` → replace model CIs with split-conformal intervals
        alpha      : miscoverage level (default 0.1 → 90% coverage target)

        Returns
        -------
        :class:`ForecastResult` stored on ``self.result_``

        When ``cv="rolling"`` is requested a
        :class:`~lazyforecast.validation.CVResult` is returned (and stored on
        ``self.cv_result_``) instead.
        """
        if self.n_periods <= 0:
            raise ValueError(f"n_periods must be > 0, got {self.n_periods}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be > 0, got {self.n_steps}")

        # Detect DatetimeIndex before prepare_series resets it  (Feature 4)
        orig_index: pd.Index | None = None
        if isinstance(data, (pd.Series, pd.DataFrame)) and isinstance(
            data.index, pd.DatetimeIndex
        ):
            orig_index = data.index

        y_values, _ = prepare_series(data, target_col=target_col)

        min_len = self.n_periods + self.n_steps + 1
        if len(y_values) <= min_len:
            raise ValueError(
                f"Series is too short: need > {min_len} observations "
                f"(n_periods={self.n_periods} + n_steps={self.n_steps} + 1), "
                f"got {len(y_values)}."
            )

        # ── Rolling-origin CV  (Feature 1) ────────────────────────────
        if cv == "rolling":
            from .validation import run_cv

            h = horizon if horizon is not None else self.n_periods
            forecasters = self._build_forecasters(n_periods=h)
            self.cv_result_ = run_cv(
                data=y_values,
                forecasters=forecasters,
                n_splits=n_splits,
                horizon=h,
                step=step,
            )
            return self.cv_result_  # type: ignore[return-value]

        # ── Conformal requires a 2×n_periods calibration window ───────
        if interval == "conformal":
            min_len_conformal = 2 * self.n_periods + self.n_steps + 1
            if len(y_values) <= min_len_conformal:
                raise ValueError(
                    f"Conformal intervals need > {min_len_conformal} observations "
                    f"(2*n_periods + n_steps + 1), got {len(y_values)}."
                )

        # ── Standard single fit ───────────────────────────────────────
        actual = y_values.iloc[-self.n_periods :].values
        forecasters = self._build_forecasters()

        predictions: dict[str, np.ndarray] = {}
        confidence_native: dict[str, np.ndarray] = {}

        for model_name, run_fn in forecasters.items():
            print(f"Fitting {model_name}...")
            fc, confint = run_fn(y_values)
            predictions[model_name] = fc
            confidence_native[model_name] = confint

        # ── Conformal interval replacement  (Feature 3) ───────────────
        if interval == "conformal":
            cal_data = y_values.iloc[: -self.n_periods]
            confidence: dict[str, np.ndarray] = {}
            for model_name, run_fn in forecasters.items():
                q = compute_conformal_quantile(cal_data, run_fn, self.n_periods, alpha)
                confidence[model_name] = apply_conformal_interval(
                    predictions[model_name], q
                )
        else:
            confidence = confidence_native

        # ── Build eval table (coverage always included) ───────────────
        eval_data: list[dict] = []
        for model_name in forecasters:
            metrics = forecast_accuracy(
                actual,
                predictions[model_name],
                model_name,
                confint=confidence[model_name],
            )
            eval_data.append(metrics)

        eval_table = build_eval_table(eval_data)
        best_model = str(eval_table.index[0])

        # ── Forecast index  (Feature 4) ───────────────────────────────
        forecast_index: pd.Index | None = None
        if orig_index is not None:
            forecast_index = infer_forecast_index(orig_index, self.n_periods)

        self.result_ = ForecastResult(
            eval_table=eval_table,
            predictions=predictions,
            confidence=confidence,
            best_model=best_model,
            forecast_index=forecast_index,
            training_data=y_values,
        )
        return self.result_

    def predict(
        self,
        data: np.ndarray | pd.Series | pd.DataFrame,
        target_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce forecasts using the best model found during :meth:`fit`.

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
        fc, confint = self._build_forecasters()[best](y_values)
        return fc, confint

    def plot(
        self,
        model_name: str | None = None,
        x_col: str | None = None,
    ) -> object:
        """Plot forecast(s) using stored :attr:`result_`.

        Parameters
        ----------
        model_name : model to plot; None plots the best model
        x_col      : x-axis label (passed through to :func:`plot_forecast`)

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
            plot_data = (
                result.training_data
                if result.training_data is not None
                else pd.Series(result.predictions[name])
            )
            fig = plot_forecast(
                data=plot_data,
                fc=result.predictions[name],
                confint=result.confidence[name],
                model_name=name,
                n_periods=self.n_periods,
                x_values=x_col,
                x_label=x_col,
            )
            figs.append(fig)

        return figs[0] if len(figs) == 1 else figs

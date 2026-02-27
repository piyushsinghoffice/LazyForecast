# LazyForecast

LazyForecast is a Python library for univariate time series forecasting. It automatically fits
multiple models to your data, ranks them by accuracy, and returns the best forecast — no manual
model selection or parameter tuning required.

[![CI](https://github.com/piyushsinghoffice/LazyForecast/actions/workflows/ci.yml/badge.svg)](https://github.com/piyushsinghoffice/LazyForecast/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lazyforecast)](https://pypi.org/project/lazyforecast/)
[![Python](https://img.shields.io/pypi/pyversions/lazyforecast)](https://pypi.org/project/lazyforecast/)

## Table of Contents

- [Installation](#installation)
- [Models](#models)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Output](#output)

---

## Installation

```bash
pip install lazyforecast
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

---

## Models

LazyForecast trains all of the following models and ranks them on the held-out test window:

| Model | Type |
|---|---|
| ARIMA | Classical statistical (auto-selected order) |
| MLP | Multi-layer perceptron (PyTorch) |
| VANILLA LSTM | Single-layer LSTM (PyTorch) |
| STACKED LSTM | Two-layer LSTM (PyTorch) |
| BIDIRECTIONAL LSTM | Bidirectional LSTM (PyTorch) |
| RNN | Vanilla RNN (PyTorch) |
| GRU | Gated Recurrent Unit (PyTorch) |

Deep models use ensemble learning (multiple independently-initialised models) and return
95% confidence intervals from the spread of ensemble predictions.

---

## Quick Start

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lazyforecast import LazyForecast

# Any pd.Series, pd.DataFrame, or np.ndarray works
series = pd.Series(np.cumsum(np.random.default_rng(42).normal(0, 1, 200)))

lf = LazyForecast(
    n_periods=10,      # forecast horizon / test window
    n_steps=5,         # lookback window for deep models
    n_members=5,       # ensemble size per deep model
    random_state=42,   # reproducible results
    epochs=200,
)

result = lf.fit(series)

print(result.eval_table)   # ranked metrics for all models
print(result.best_model)   # name of the top-ranked model

fig = lf.plot()            # forecast chart — best model
plt.show()
```

### Using a DataFrame

```python
import yfinance as yf

df = yf.download("GOOGL", start="2021-01-01", end="2022-12-31").reset_index()

result = lf.fit(df, target_col="Close")
```

### Plotting a specific model

```python
fig = lf.plot(model_name="GRU")
fig.savefig("gru_forecast.png")
```

### Standalone plot function

```python
from lazyforecast import plot_forecast

fig = plot_forecast(
    data=series,
    fc=result.predictions["ARIMA"],
    confint=result.confidence["ARIMA"],
    model_name="ARIMA",
    n_periods=10,
)
```

---

## API Reference

### `LazyForecast(n_periods, n_steps, n_members, random_state, device, epochs, batch_size, lr, verbose)`

| Parameter | Default | Description |
|---|---|---|
| `n_periods` | 5 | Forecast horizon (test window size) |
| `n_steps` | 5 | Lookback window for deep models |
| `n_members` | 10 | Ensemble size per deep model |
| `random_state` | `None` | Seed for reproducibility |
| `device` | `None` | `'cpu'` or `'cuda'` (auto-detects) |
| `epochs` | 200 | Training epochs per ensemble member |
| `batch_size` | 32 | DataLoader batch size |
| `lr` | 1e-3 | Adam learning rate |
| `verbose` | `False` | Print training progress |

### `.fit(data, target_col=None) → ForecastResult`

Trains all models on `data[:-n_periods]` and evaluates on `data[-n_periods:]`.

- `data` — `pd.Series`, `pd.DataFrame`, or `np.ndarray`
- `target_col` — required when `data` is a `DataFrame`

Returns a `ForecastResult` with:

| Attribute | Type | Description |
|---|---|---|
| `eval_table` | `pd.DataFrame` | All models ranked best-first |
| `predictions` | `dict[str, np.ndarray]` | Point forecasts per model |
| `confidence` | `dict[str, np.ndarray]` | `(n_periods, 2)` intervals per model |
| `best_model` | `str` | Name of the top-ranked model |

### `.plot(model_name=None) → Figure`

Returns a `matplotlib.figure.Figure`. `model_name=None` plots the best model.

### `.predict(data, target_col=None) → (forecasts, confidence)`

Re-runs the best model on new data. Requires `fit()` to have been called first.

---

## Output

### Evaluation Table

| model | mda | rmse | mape | R2 | mae | corr |
|:---|---:|---:|---:|---:|---:|---:|
| ARIMA | 0.551 | 2.629 | 0.021 | 0.725 | 1.953 | 0.863 |
| GRU | 0.551 | 2.699 | 0.022 | 0.668 | 2.039 | 0.851 |
| BIDIRECTIONAL LSTM | 0.551 | 2.725 | 0.022 | 0.640 | 2.086 | 0.852 |
| VANILLA LSTM | 0.551 | 2.797 | 0.023 | 0.630 | 2.137 | 0.849 |
| STACKED LSTM | 0.510 | 2.908 | 0.023 | 0.582 | 2.222 | 0.852 |
| RNN | 0.469 | 2.969 | 0.024 | 0.609 | 2.273 | 0.822 |
| MLP | 0.510 | 7.579 | 0.066 | -4.667 | 6.667 | 0.698 |

### Sample Forecast Charts

#### ARIMA
![ARIMA forecast](https://i.imgur.com/ML90i3D.png)

#### Vanilla LSTM
![Vanilla LSTM forecast](https://i.imgur.com/R8KDORF.png)

#### GRU
![GRU forecast](https://i.imgur.com/8hiG805.png)

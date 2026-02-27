# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LazyForecast is a Python library for univariate time series forecasting. It automatically fits
multiple models to a time series and returns a ranked evaluation table, point forecasts, and
confidence intervals from the best-performing model.

The package is published to PyPI as `lazyforecast` (`pip install lazyforecast`).

## Directory Layout

```
LazyForecast/               ← repo root
├── pyproject.toml          ← packaging (replaces setup.py)
├── README.md
├── LICENSE
├── ex.py                   ← example script (needs yfinance)
├── .gitignore
│
├── src/
│   └── lazyforecast/
│       ├── __init__.py     (exports LazyForecast, ForecastResult, plot_forecast)
│       ├── core.py         (LazyForecast class, ForecastResult dataclass)
│       ├── models.py       (TorchMLP, TorchRNNBase)
│       ├── forecasters.py  (train_deep_ensemble, deep_forecast, arima_forecast)
│       ├── metrics.py      (mean_directional_accuracy, forecast_accuracy, build_eval_table)
│       ├── plotting.py     (plot_forecast — standalone, returns Figure)
│       ├── utils.py        (split_sequence, prepare_series)
│       └── py.typed
│
└── tests/
    ├── conftest.py
    ├── test_metrics.py
    ├── test_utils.py
    ├── test_forecasters.py
    └── test_core.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Key dependencies (from `pyproject.toml`): `torch>=2.0`, `pmdarima>=2.0`, `scikit-learn>=1.0,<2.0`,
`numpy>=1.22`, `pandas>=1.5`, `matplotlib>=3.5`, `tqdm>=4.60`.

Dev extras: `pytest>=7`, `ruff`, `twine>=4`.

## Running Tests

```bash
pytest tests/ -v
```

## Linting

```bash
ruff check src/ tests/
```

## Running the Example

```bash
pip install yfinance  # not in main deps
python ex.py
```

## Publishing to PyPI

Triggered automatically by creating a GitHub release (see `.github/workflows/python-publish.yml`).
To build manually:

```bash
python -m build
twine check dist/*
twine upload dist/*
```

Requires `PYPI_USERS` and `PYPI_PASSWORD` secrets in GitHub.

## Architecture

### Public API

```python
import lazyforecast as lf

model = lf.LazyForecast(n_periods=10, n_steps=5, n_members=5, random_state=42)
result = model.fit(df, target_col="Close")   # → ForecastResult
print(result.eval_table)                     # ranked metrics table
print(result.best_model)                     # top model name

fig = model.plot()                           # best-model forecast plot
fig.savefig("forecast.png")

# Standalone plotting function
fig = lf.plot_forecast(data, fc, confint, model_name="GRU", n_periods=10)
```

### ForecastResult dataclass

| Field       | Type                      | Description                        |
|-------------|---------------------------|------------------------------------|
| eval_table  | pd.DataFrame              | All models ranked best-first       |
| predictions | dict[str, np.ndarray]     | Point forecasts per model          |
| confidence  | dict[str, np.ndarray]     | (n_periods, 2) intervals per model |
| best_model  | str                       | Name of top-ranked model           |

### LazyForecast constructor parameters

| Parameter    | Default | Description                                  |
|--------------|---------|----------------------------------------------|
| n_periods    | 5       | Test window size / forecast horizon          |
| n_steps      | 5       | Lookback window for deep models              |
| n_members    | 10      | Ensemble size per deep model                 |
| random_state | None    | Seeds torch + numpy for reproducibility      |
| device       | None    | 'cpu', 'cuda', etc. (auto-detects if None)  |
| epochs       | 200     | Training epochs per ensemble member          |
| batch_size   | 32      | DataLoader batch size                        |
| lr           | 1e-3    | Adam learning rate                           |
| verbose      | False   | Print training progress                      |

### Model Registry

| Display name        | Architecture                            |
|---------------------|-----------------------------------------|
| ARIMA               | `pmdarima.auto_arima` + walk-forward    |
| MLP                 | TorchMLP (2 hidden layers, 64 units)    |
| VANILLA LSTM        | TorchRNNBase('lstm', layers=1)          |
| STACKED LSTM        | TorchRNNBase('lstm', layers=2)          |
| BIDIRECTIONAL LSTM  | TorchRNNBase('lstm', bidirectional=True)|
| RNN                 | TorchRNNBase('rnn')                     |
| GRU                 | TorchRNNBase('gru')                     |

### Key design decisions

- **No plotting side effects in `fit()`** — use `model.plot()` or `plot_forecast()` explicitly.
- **`ForecastResult` stores everything** — predictions and confidence for every model.
- **MinMaxScaler in deep_forecast** — inputs are normalised before training, inverse-transformed after.
- **Walk-forward ARIMA** — `auto_arima` runs once; `model.update()` is used for subsequent steps.
- **Warnings scoped locally** — `catch_warnings` in `arima_forecast` only; no global suppression.
- **Reproducibility** — `torch.manual_seed(random_state + i)` per ensemble member.
- **RMSE** — `np.sqrt(mean_squared_error(...))` (no deprecated `squared=False`).

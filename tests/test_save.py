"""Unit tests for ForecastResult.save() — Feature 5."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lazyforecast import ForecastResult, LazyForecast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAST = dict(
    n_periods=3, n_steps=5, n_members=1, epochs=2, batch_size=16, device="cpu"
)


def _quick_result() -> ForecastResult:
    rng = np.random.default_rng(0)
    s = pd.Series(np.cumsum(rng.normal(0, 1, 60)))
    lf = LazyForecast(**FAST)
    return lf.fit(s)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_creates_directory(tmp_path):
    result = _quick_result()
    out = tmp_path / "artifacts"
    result.save(str(out))
    assert out.is_dir()


def test_save_creates_config_json(tmp_path):
    result = _quick_result()
    result.save(str(tmp_path))
    cfg_path = tmp_path / "config.json"
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text())
    assert "best_model" in cfg
    assert "models" in cfg
    assert "n_periods" in cfg
    assert cfg["n_periods"] == FAST["n_periods"]


def test_save_config_best_model_matches(tmp_path):
    result = _quick_result()
    result.save(str(tmp_path))
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["best_model"] == result.best_model


def test_save_creates_metrics_csv(tmp_path):
    result = _quick_result()
    result.save(str(tmp_path))
    assert (tmp_path / "metrics.csv").exists()
    df = pd.read_csv(tmp_path / "metrics.csv", index_col=0)
    assert len(df) == len(result.predictions)


def test_save_creates_forecast_csv(tmp_path):
    result = _quick_result()
    result.save(str(tmp_path))
    assert (tmp_path / "forecast.csv").exists()
    df = pd.read_csv(tmp_path / "forecast.csv", index_col=0)
    assert len(df) == FAST["n_periods"]
    assert set(df.columns) == set(result.predictions.keys())


def test_save_creates_intervals_csv(tmp_path):
    result = _quick_result()
    result.save(str(tmp_path))
    assert (tmp_path / "intervals.csv").exists()
    df = pd.read_csv(tmp_path / "intervals.csv")
    assert {"model", "period", "lower", "upper"}.issubset(df.columns)
    # One row per model per period
    expected_rows = len(result.predictions) * FAST["n_periods"]
    assert len(df) == expected_rows


def test_save_creates_plot_png(tmp_path):
    result = _quick_result()
    result.save(str(tmp_path))
    assert (tmp_path / "plot.png").exists()
    assert (tmp_path / "plot.png").stat().st_size > 0


def test_save_safe_overwrite(tmp_path):
    """Calling save() twice must not raise and must produce valid files."""
    result = _quick_result()
    result.save(str(tmp_path))
    result.save(str(tmp_path))  # second call must not raise
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["best_model"] == result.best_model


def test_save_no_training_data_no_plot(tmp_path):
    """If training_data is None, plot.png must not be created."""
    rng = np.random.default_rng(1)
    fc = rng.normal(0, 1, 3).astype(np.float32)
    ci = np.column_stack([fc - 1, fc + 1]).astype(np.float32)

    dummy_table = pd.DataFrame(
        {"mda": [0.5], "rmse": [1.0], "mape": [0.1], "R2": [0.5], "mae": [0.8], "corr": [0.9]},
        index=pd.Index(["NAIVE"], name="model"),
    )
    result = ForecastResult(
        eval_table=dummy_table,
        predictions={"NAIVE": fc},
        confidence={"NAIVE": ci},
        best_model="NAIVE",
        training_data=None,  # explicitly no training data
    )
    result.save(str(tmp_path))
    assert not (tmp_path / "plot.png").exists()

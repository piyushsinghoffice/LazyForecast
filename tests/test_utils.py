"""Tests for src/lazyforecast/utils.py."""

import numpy as np
import pandas as pd
import pytest

from lazyforecast.utils import prepare_series, split_sequence

# ---------------------------------------------------------------------------
# split_sequence
# ---------------------------------------------------------------------------

def test_split_sequence_shapes():
    """len=100, n_steps=5 → X shape (95, 5), y shape (95,)."""
    seq = np.arange(100, dtype=np.float32)
    X, y = split_sequence(seq, n_steps=5)
    assert X.shape == (95, 5)
    assert y.shape == (95,)


def test_split_sequence_values():
    """First X row equals seq[0:5]; y[0] equals seq[5]."""
    seq = np.arange(20, dtype=np.float32)
    X, y = split_sequence(seq, n_steps=5)
    np.testing.assert_array_equal(X[0], seq[:5])
    assert y[0] == seq[5]


def test_split_sequence_last_window():
    """Last row of X equals seq[-6:-1]; y[-1] equals seq[-1]."""
    seq = np.arange(20, dtype=np.float32)
    X, y = split_sequence(seq, n_steps=5)
    np.testing.assert_array_equal(X[-1], seq[-6:-1])
    assert y[-1] == seq[-1]


# ---------------------------------------------------------------------------
# prepare_series
# ---------------------------------------------------------------------------

def test_prepare_series_ndarray():
    arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    y, x = prepare_series(arr)
    assert isinstance(y, pd.Series)
    assert x is None
    assert not y.isnull().any()
    assert len(y) == 5


def test_prepare_series_series():
    s = pd.Series([10.0, 20.0, 30.0])
    y, x = prepare_series(s)
    assert isinstance(y, pd.Series)
    assert x is None
    assert len(y) == 3


def test_prepare_series_dataframe():
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Date": ["a", "b", "c"]})
    y, x = prepare_series(df, target_col="Close")
    assert isinstance(y, pd.Series)
    np.testing.assert_array_almost_equal(y.values, [1.0, 2.0, 3.0])


def test_prepare_series_dataframe_missing_target_col_raises():
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="target_col"):
        prepare_series(df)  # target_col not provided


def test_prepare_series_dataframe_bad_col_raises():
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not found"):
        prepare_series(df, target_col="Volume")


def test_prepare_series_bad_type_raises():
    with pytest.raises(ValueError, match="Unsupported data type"):
        prepare_series("not a valid input")  # type: ignore[arg-type]


def test_prepare_series_interpolates_nans():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    y, _ = prepare_series(s)
    assert not y.isnull().any()

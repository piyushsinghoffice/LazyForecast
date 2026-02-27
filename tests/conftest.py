"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_series() -> pd.Series:
    """200-point random walk series with a fixed seed."""
    rng = np.random.default_rng(42)
    return pd.Series(np.cumsum(rng.normal(0, 1, 200)))


@pytest.fixture
def short_series() -> pd.Series:
    """Short series for edge-case tests."""
    rng = np.random.default_rng(0)
    return pd.Series(np.cumsum(rng.normal(0, 1, 50)))

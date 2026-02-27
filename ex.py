"""Example: fit LazyForecast on GOOGL historical close prices."""

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from lazyforecast import LazyForecast

# ── Download data ──────────────────────────────────────────────────────────
df = yf.download("GOOGL", start="2021-01-01", end="2022-12-31")
df = df.reset_index()

# ── Configure and fit ──────────────────────────────────────────────────────
lf = LazyForecast(
    n_periods=50,
    n_steps=5,
    n_members=5,
    random_state=42,
    device="cpu",   # change to 'cuda' if available
    epochs=200,
)

result = lf.fit(df, target_col="Close")

# ── Print evaluation table ─────────────────────────────────────────────────
print("\n=== Evaluation Table ===")
print(result.eval_table)
print(f"\nBest model: {result.best_model}")

# ── Plot the best model's forecast ────────────────────────────────────────
fig = lf.plot()
plt.show()

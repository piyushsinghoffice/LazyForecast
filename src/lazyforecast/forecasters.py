"""Model training and forecasting functions."""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .utils import split_sequence


def train_deep_ensemble(
    model_factory: Callable[[], nn.Module],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_members: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    verbose: bool,
    random_state: int | None,
) -> list[nn.Module]:
    """Train an ensemble of independently-initialised models.

    Tensors and DataLoader are built once, before the member loop, to avoid
    redundant work. Each member is seeded reproducibly when random_state is given.

    Parameters
    ----------
    model_factory : zero-arg callable that returns a fresh nn.Module
    X_train       : (n_samples, n_steps) float32 array
    y_train       : (n_samples,) float32 array
    n_members     : number of ensemble members
    device        : torch device
    epochs        : training epochs per member
    batch_size    : DataLoader batch size
    lr            : Adam learning rate
    verbose       : print per-epoch loss if True
    random_state  : base seed (member i uses random_state + i)

    Returns
    -------
    list of trained nn.Module (eval mode)
    """
    n_features = 1
    # Pre-compute tensors once
    X_tensor = torch.from_numpy(X_train.reshape(-1, X_train.shape[1], n_features)).to(device)
    y_tensor = torch.from_numpy(y_train.reshape(-1, 1)).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)

    loss_fn = nn.MSELoss()
    ensemble: list[nn.Module] = []

    for i in tqdm(range(n_members), desc="Training ensemble", leave=False):
        if random_state is not None:
            torch.manual_seed(random_state + i)
            np.random.seed(random_state + i)

        model = model_factory().to(device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)

            if verbose and (epoch % 25 == 0 or epoch == 1 or epoch == epochs):
                avg = epoch_loss / len(dataset)
                print(f"  member {i + 1}/{n_members} | epoch {epoch}/{epochs} | loss {avg:.6f}")

        model.eval()
        ensemble.append(model)

    return ensemble


@torch.no_grad()
def predict_ensemble(
    ensemble: list[nn.Module],
    X_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float]:
    """Predict with an ensemble and return mean + 95% interval.

    Parameters
    ----------
    ensemble : list of trained nn.Module in eval mode
    X_tensor : (1, n_steps, 1) tensor
    device   : torch device

    Returns
    -------
    (mean, lower_95, upper_95)
    """
    preds = []
    for model in ensemble:
        yhat = model(X_tensor.to(device)).detach().cpu().numpy().reshape(-1)
        preds.append(yhat)
    preds_arr = np.asarray(preds)  # (n_members, 1)

    mean = float(preds_arr.mean())
    std = float(preds_arr.std())
    interval = 1.96 * std
    return mean, mean - interval, mean + interval


def deep_forecast(
    model_factory: Callable[[], nn.Module],
    data: pd.Series,
    n_periods: int,
    n_steps: int,
    n_members: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    verbose: bool,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train an ensemble on data[:-n_periods] and produce out-of-sample forecasts.

    MinMaxScaling is applied before training and inverted after prediction to
    keep gradients well-conditioned regardless of the data scale.

    Parameters
    ----------
    model_factory : zero-arg callable → nn.Module
    data          : full series (train + test)
    n_periods     : forecast horizon / test window
    n_steps       : lookback window
    n_members     : ensemble size
    device        : torch device
    epochs        : training epochs
    batch_size    : DataLoader batch size
    lr            : Adam learning rate
    verbose       : print training progress
    random_state  : reproducibility seed

    Returns
    -------
    forecasts  : np.ndarray of shape (n_periods,)
    confidence : np.ndarray of shape (n_periods, 2)  — [lower, upper]
    """
    values = data.values.reshape(-1, 1).astype(np.float32)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()
    scaled_series = pd.Series(scaled)

    train_size = len(scaled_series) - n_periods
    train = scaled_series.iloc[:train_size]
    # test window includes n_steps lookback to build X_test
    test = scaled_series.iloc[train_size - n_steps:]

    X_train, y_train = split_sequence(train.values, n_steps)
    X_test, _ = split_sequence(test.values, n_steps)

    ensemble = train_deep_ensemble(
        model_factory=model_factory,
        X_train=X_train,
        y_train=y_train,
        n_members=n_members,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose,
        random_state=random_state,
    )

    n_features = 1
    predictions = np.zeros(n_periods, dtype=np.float32)
    confidence = np.zeros((n_periods, 2), dtype=np.float32)

    for period in range(n_periods):
        x_input = X_test[period].reshape(1, n_steps, n_features)
        x_tensor = torch.from_numpy(x_input)

        mean_s, lower_s, upper_s = predict_ensemble(ensemble, x_tensor, device)

        # Inverse-transform scalar predictions
        mean_orig = scaler.inverse_transform([[mean_s]])[0, 0]
        lower_orig = scaler.inverse_transform([[lower_s]])[0, 0]
        upper_orig = scaler.inverse_transform([[upper_s]])[0, 0]

        predictions[period] = mean_orig
        confidence[period] = (lower_orig, upper_orig)

    return predictions, confidence


def arima_forecast(
    data: pd.Series,
    n_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward ARIMA forecast.

    Calls ``auto_arima`` once to select order, then uses ``model.update(obs)``
    for each subsequent step — much faster than re-fitting from scratch each period.
    Warnings are suppressed locally, not globally.

    Parameters
    ----------
    data      : full series (train + test)
    n_periods : forecast horizon / test window

    Returns
    -------
    forecasts  : np.ndarray of shape (n_periods,)
    confidence : np.ndarray of shape (n_periods, 2)  — [lower, upper]
    """
    import pmdarima as pm

    train_size = len(data) - n_periods
    train = data.iloc[:train_size]

    forecasts = np.zeros(n_periods, dtype=np.float32)
    confidence = np.zeros((n_periods, 2), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(
            train,
            start_p=1,
            start_q=0,
            test="pp",
            max_p=3,
            max_q=2,
            m=1,
            d=None,
            seasonal=False,
            start_P=0,
            D=0,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )

        for period in tqdm(range(n_periods), desc="ARIMA walk-forward", leave=False):
            fc, confint = model.predict(n_periods=1, return_conf_int=True)
            fc_arr = np.asarray(fc).flatten()
            ci_arr = np.asarray(confint).reshape(1, 2)
            forecasts[period] = float(fc_arr[0])
            confidence[period] = (float(ci_arr[0, 0]), float(ci_arr[0, 1]))

            # Update model with the true observation (walk-forward)
            if period < n_periods - 1:
                obs_idx = train_size + period
                model.update(data.iloc[obs_idx : obs_idx + 1])

    return forecasts, confidence

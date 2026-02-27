"""PyTorch model definitions for LazyForecast."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["TorchMLP", "TorchRNNBase"]


class TorchMLP(nn.Module):
    """Two-hidden-layer feedforward network for univariate forecasting."""

    def __init__(self, n_steps: int, n_features: int = 1, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_steps * n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchRNNBase(nn.Module):
    """Shared wrapper for RNN / LSTM / GRU variants.

    Parameters
    ----------
    cell_type     : 'rnn', 'lstm', or 'gru'
    input_size    : number of input features (1 for univariate)
    hidden_size   : number of hidden units
    bidirectional : whether to use a bidirectional variant
    num_layers    : number of stacked recurrent layers
    """

    def __init__(
        self,
        cell_type: str,
        input_size: int = 1,
        hidden_size: int = 32,
        bidirectional: bool = False,
        num_layers: int = 1,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        rnn_kwargs = dict(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        if cell_type == "rnn":
            self.rnn = nn.RNN(**rnn_kwargs, nonlinearity="relu")
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif cell_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            raise ValueError(f"cell_type must be 'rnn', 'lstm', or 'gru'. Got '{cell_type}'.")

        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # (batch, hidden * dirs)
        return self.head(last)

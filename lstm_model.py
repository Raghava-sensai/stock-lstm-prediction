"""
LSTM Architecture for Stock Price Time Series Forecasting
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Multi-layer Stacked LSTM with Dropout and fully connected regression head.
    Supports arbitrary number of input features (OHLCV + technical indicators).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super(StockLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        num_directions = 2 if bidirectional else 1
        fc_input_dim = hidden_size * num_directions

        # Layer Normalization and Dense Head
        self.layer_norm = nn.LayerNorm(fc_input_dim)
        self.dropout = nn.Dropout(dropout)
        dense_dim = max(64, output_size * 4)
        self.fc1 = nn.Linear(fc_input_dim, dense_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(dense_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            out: Tensor of shape (batch_size, output_size)
        """
        # lstm_out shape: (batch_size, seq_len, hidden_size * num_directions)
        lstm_out, _ = self.lstm(x)

        # Extract last time-step representation
        last_step = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Regressor head
        normed = self.layer_norm(last_step)
        dropped = self.dropout(normed)
        dense1 = self.relu(self.fc1(dropped))
        out = self.fc_out(dense1)
        return out

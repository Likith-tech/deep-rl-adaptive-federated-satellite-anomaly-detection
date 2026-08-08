"""Temporal GRU + attention anomaly detector.

    Input sequence (batch, seq_len, input_dim)
        -> Linear projection -> ReLU
        -> GRU (num_gru_layers)
        -> Temporal attention (learned per-timestep weighting)
        -> Dropout
        -> Dense -> logit

This is the TEMPORAL foundation described in Phase 3 — it learns from
ORDERED/SEQUENTIAL records (the dataset's own row ordering), not
genuine timestamped time-series, and has no spatial/multi-satellite
component yet (see module docstring in temporal_attention.py and
docs/project-progress/04-phase-3-spatio-temporal.md).
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.temporal_attention import TemporalAttention


class TemporalGRUAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int,
        num_gru_layers: int = 1,
        dropout: float = 0.3,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.projection = nn.Linear(input_dim, projection_dim)
        self.activation = nn.ReLU()
        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, input_dim)
        Returns (logits (batch, output_dim), attention_weights (batch, seq_len))."""
        projected = self.activation(self.projection(x))  # (batch, seq_len, projection_dim)
        gru_out, _ = self.gru(projected)  # (batch, seq_len, hidden_dim)
        context, attn_weights = self.attention(gru_out)  # (batch, hidden_dim), (batch, seq_len)
        context = self.dropout(context)
        logits = self.output_layer(context)  # (batch, output_dim)
        return logits, attn_weights

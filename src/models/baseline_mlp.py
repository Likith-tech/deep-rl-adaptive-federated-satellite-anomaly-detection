"""Baseline feed-forward MLP anomaly detector.

    Input features (121-dim)
        -> Dense -> ReLU -> Dropout
        -> Dense -> ReLU -> Dropout
        -> Dense -> single logit
        -> sigmoid (applied at inference/loss time) -> P(anomaly)

Deliberately simple: this model exists to be a trustworthy reference
point for later spatio-temporal / federated / DRL-driven comparisons,
not to be the best possible classifier.
"""

from __future__ import annotations

import torch
from torch import nn


class BaselineMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dimensions: list[int],
        dropout: float = 0.3,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dimensions:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (batch, output_dim). Apply sigmoid
        for probabilities — kept separate so BCEWithLogitsLoss can be
        used directly for numerical stability."""
        return self.network(x)

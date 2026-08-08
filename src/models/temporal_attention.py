"""Learnable temporal attention over a sequence of GRU hidden states.

Instead of treating every timestep in a sequence equally (e.g. always
using only the last GRU hidden state), this module lets the model learn
which timesteps matter more for the final anomaly decision, and exposes
those weights so they can be inspected/visualized.

This is the TEMPORAL half of the eventual "Spatio-Temporal Gated
Attention Mechanism" — there is no spatial (multi-satellite) component
yet. That is introduced later, once a genuine multi-client satellite
simulation exists (see docs/project-progress/04-phase-3-spatio-temporal.md).
"""

from __future__ import annotations

import torch
from torch import nn


class TemporalAttention(nn.Module):
    """Additive attention over the timestep dimension of a GRU output.

    Input:  (batch, seq_len, hidden_dim)
    Output: context (batch, hidden_dim), weights (batch, seq_len)

    `weights` sums to 1.0 across the sequence dimension for every
    sample (guaranteed by softmax), so it can be read as "how much of
    the final decision came from each timestep."
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score_layer = nn.Linear(hidden_dim, 1)

    def forward(self, gru_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score_layer(gru_outputs).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len), sums to 1 per sample
        context = torch.sum(weights.unsqueeze(-1) * gru_outputs, dim=1)  # (batch, hidden_dim)
        return context, weights

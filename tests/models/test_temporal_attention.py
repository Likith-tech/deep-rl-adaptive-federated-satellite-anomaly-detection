from __future__ import annotations

import torch

from src.models.temporal_attention import TemporalAttention


def test_attention_output_shapes():
    attn = TemporalAttention(hidden_dim=8)
    gru_outputs = torch.randn(4, 6, 8)  # (batch, seq_len, hidden_dim)
    context, weights = attn(gru_outputs)
    assert context.shape == (4, 8)
    assert weights.shape == (4, 6)


def test_attention_weights_sum_to_one_per_sample():
    attn = TemporalAttention(hidden_dim=5)
    gru_outputs = torch.randn(3, 7, 5)
    _, weights = attn(gru_outputs)
    sums = weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)


def test_attention_weights_are_non_negative():
    attn = TemporalAttention(hidden_dim=4)
    gru_outputs = torch.randn(2, 5, 4)
    _, weights = attn(gru_outputs)
    assert (weights >= 0).all()


def test_attention_context_is_weighted_average_of_inputs():
    attn = TemporalAttention(hidden_dim=3)
    gru_outputs = torch.randn(2, 4, 3)
    context, weights = attn(gru_outputs)
    manual_context = torch.sum(weights.unsqueeze(-1) * gru_outputs, dim=1)
    assert torch.allclose(context, manual_context, atol=1e-6)

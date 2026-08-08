from __future__ import annotations

import torch

from src.models.temporal_gru_attention import TemporalGRUAttention


def _make_model(input_dim=6, seq_len=5):
    return TemporalGRUAttention(
        input_dim=input_dim,
        projection_dim=8,
        hidden_dim=8,
        num_gru_layers=1,
        dropout=0.1,
        output_dim=1,
    )


def test_forward_pass_output_shapes():
    model = _make_model(input_dim=6)
    x = torch.randn(4, 5, 6)  # (batch, seq_len, input_dim)
    logits, attn_weights = model(x)
    assert logits.shape == (4, 1)
    assert attn_weights.shape == (4, 5)


def test_forward_pass_returns_finite_values():
    model = _make_model()
    x = torch.randn(3, 5, 6)
    logits, attn_weights = model(x)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attn_weights).all()


def test_attention_weights_sum_to_one():
    model = _make_model()
    x = torch.randn(3, 5, 6)
    _, attn_weights = model(x)
    assert torch.allclose(attn_weights.sum(dim=1), torch.ones(3), atol=1e-5)


def test_forward_pass_handles_multi_layer_gru():
    model = TemporalGRUAttention(
        input_dim=4, projection_dim=6, hidden_dim=6, num_gru_layers=2, dropout=0.2, output_dim=1
    )
    x = torch.randn(2, 8, 4)
    logits, attn_weights = model(x)
    assert logits.shape == (2, 1)
    assert attn_weights.shape == (2, 8)


def test_model_is_deterministic_with_fixed_seed():
    x = torch.randn(2, 5, 6)

    torch.manual_seed(42)
    model_a = _make_model()
    model_a.eval()

    torch.manual_seed(42)
    model_b = _make_model()
    model_b.eval()

    with torch.no_grad():
        logits_a, _ = model_a(x)
        logits_b, _ = model_b(x)
    assert torch.allclose(logits_a, logits_b)

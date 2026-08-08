from __future__ import annotations

import torch

from src.models.baseline_mlp import BaselineMLP


def test_model_creation_with_expected_layer_shapes():
    model = BaselineMLP(input_dim=10, hidden_dimensions=[8, 4], dropout=0.3, output_dim=1)
    linear_layers = [m for m in model.network if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3  # 2 hidden + 1 output
    assert linear_layers[0].in_features == 10
    assert linear_layers[0].out_features == 8
    assert linear_layers[1].in_features == 8
    assert linear_layers[1].out_features == 4
    assert linear_layers[2].in_features == 4
    assert linear_layers[2].out_features == 1


def test_forward_pass_output_shape():
    model = BaselineMLP(input_dim=6, hidden_dimensions=[4], dropout=0.1, output_dim=1)
    x = torch.randn(5, 6)  # batch of 5
    logits = model(x)
    assert logits.shape == (5, 1)


def test_forward_pass_returns_finite_values():
    model = BaselineMLP(input_dim=6, hidden_dimensions=[4], dropout=0.1, output_dim=1)
    x = torch.randn(3, 6)
    logits = model(x)
    assert torch.isfinite(logits).all()


def test_model_is_deterministic_with_fixed_seed():
    torch.manual_seed(42)
    model_a = BaselineMLP(input_dim=6, hidden_dimensions=[4], dropout=0.0, output_dim=1)
    torch.manual_seed(42)
    model_b = BaselineMLP(input_dim=6, hidden_dimensions=[4], dropout=0.0, output_dim=1)

    x = torch.randn(2, 6)
    model_a.eval()
    model_b.eval()
    torch.manual_seed(0)
    out_a = model_a(x)
    torch.manual_seed(0)
    out_b = model_b(x)
    assert torch.allclose(out_a, out_b)

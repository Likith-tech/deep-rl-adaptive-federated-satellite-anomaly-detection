"""Unit tests for the FedAvg aggregation rule in isolation — numeric
verification of the weighted average, not just "does it run"."""

from __future__ import annotations

import pytest
import torch

from src.federated.fedavg import federated_average


def test_fedavg_matches_hand_computed_weighted_average():
    # Client A: samples=10, Client B: samples=30 -> weights 0.25 / 0.75
    state_a = {"w": torch.tensor([1.0, 2.0, 3.0]), "b": torch.tensor([10.0])}
    state_b = {"w": torch.tensor([5.0, 6.0, 7.0]), "b": torch.tensor([20.0])}

    result = federated_average([state_a, state_b], [10, 30])

    expected_w = 0.25 * state_a["w"] + 0.75 * state_b["w"]
    expected_b = 0.25 * state_a["b"] + 0.75 * state_b["b"]

    assert torch.allclose(result["w"], expected_w)
    assert torch.allclose(result["b"], expected_b)


def test_fedavg_equal_sample_counts_is_plain_average():
    state_a = {"w": torch.tensor([2.0, 4.0])}
    state_b = {"w": torch.tensor([6.0, 8.0])}

    result = federated_average([state_a, state_b], [50, 50])

    assert torch.allclose(result["w"], torch.tensor([4.0, 6.0]))


def test_fedavg_single_client_returns_unchanged_weights():
    state_a = {"w": torch.tensor([1.0, -2.0, 3.5])}
    result = federated_average([state_a], [100])
    assert torch.allclose(result["w"], state_a["w"])


def test_fedavg_preserves_parameter_shapes():
    state_a = {"layer.weight": torch.randn(8, 4), "layer.bias": torch.randn(8)}
    state_b = {"layer.weight": torch.randn(8, 4), "layer.bias": torch.randn(8)}

    result = federated_average([state_a, state_b], [20, 5])

    assert result["layer.weight"].shape == state_a["layer.weight"].shape
    assert result["layer.bias"].shape == state_a["layer.bias"].shape


def test_fedavg_is_deterministic():
    state_a = {"w": torch.tensor([1.0, 2.0])}
    state_b = {"w": torch.tensor([3.0, 4.0])}
    state_c = {"w": torch.tensor([5.0, 6.0])}

    result_1 = federated_average([state_a, state_b, state_c], [10, 20, 30])
    result_2 = federated_average([state_a, state_b, state_c], [10, 20, 30])

    assert torch.equal(result_1["w"], result_2["w"])


def test_fedavg_rejects_empty_input():
    with pytest.raises(ValueError):
        federated_average([], [])


def test_fedavg_rejects_mismatched_lengths():
    state_a = {"w": torch.tensor([1.0])}
    with pytest.raises(ValueError):
        federated_average([state_a], [10, 20])


def test_fedavg_rejects_non_positive_sample_counts():
    state_a = {"w": torch.tensor([1.0])}
    state_b = {"w": torch.tensor([2.0])}
    with pytest.raises(ValueError):
        federated_average([state_a, state_b], [10, 0])


def test_fedavg_rejects_mismatched_keys():
    state_a = {"w": torch.tensor([1.0])}
    state_b = {"different_key": torch.tensor([2.0])}
    with pytest.raises(ValueError):
        federated_average([state_a, state_b], [10, 10])

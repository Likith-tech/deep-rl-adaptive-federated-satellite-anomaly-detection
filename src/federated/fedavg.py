"""Phase 6 — the FedAvg aggregation rule, in isolation. Phase 8 adds a
general-purpose weighted-average primitive on top (`weighted_average`),
which standard FedAvg (`federated_average`) is now expressed in terms
of — its behavior and output are UNCHANGED, only its implementation is
shared with the new rule-based adaptive weighting (src/federated/adaptive.py).

    w_global = sum_k (n_k / N) * w_k

where `w_k` is client k's local model parameters after local training,
`n_k` is client k's local sample count, and `N = sum_k n_k`. This is
the standard McMahan et al. (2017) FedAvg sample-count-weighted average
— NOT equal weighting, NOT resource-based weighting.

This module operates purely on state_dicts and (sample counts or
already-computed weights); it never touches a dataset, a DataLoader,
or a file path. That's deliberate — it keeps the aggregation logic
trivially unit-testable and makes the client/server privacy boundary
explicit (see src/federated/server.py).
"""

from __future__ import annotations

import torch


def weighted_average(
    client_state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """General weighted average of client state_dicts given arbitrary
    (already-computed) per-client weights that must sum to ~1.

    This is the shared primitive underneath both standard FedAvg
    (`federated_average`, weights = sample-count proportions) and
    Phase 8's rule-based adaptive weighting (`src/federated/adaptive.py`,
    weights = normalized client scores) — one aggregation
    implementation, two ways of computing the weights that feed it.
    """
    if not client_state_dicts:
        raise ValueError("weighted_average requires at least one client state_dict")
    if len(client_state_dicts) != len(weights):
        raise ValueError("client_state_dicts and weights must be the same length")
    if any(w < 0 for w in weights):
        raise ValueError("weights must all be non-negative")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1.0 (got {sum(weights)})")

    keys = client_state_dicts[0].keys()
    for state_dict in client_state_dicts[1:]:
        if state_dict.keys() != keys:
            raise ValueError("all client state_dicts must have identical parameter keys")

    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        weighted_sum = torch.zeros_like(client_state_dicts[0][key], dtype=torch.float32)
        for state_dict, weight in zip(client_state_dicts, weights):
            weighted_sum += state_dict[key].to(torch.float32) * weight
        averaged[key] = weighted_sum

    return averaged


def federated_average(
    client_state_dicts: list[dict[str, torch.Tensor]],
    client_sample_counts: list[int],
) -> dict[str, torch.Tensor]:
    """Sample-count-weighted average of one or more client state_dicts.

    All state_dicts must share the same keys and tensor shapes (true by
    construction here, since every client starts each round from the
    same global model and trains the same architecture).
    """
    if not client_state_dicts:
        raise ValueError("federated_average requires at least one client state_dict")
    if len(client_state_dicts) != len(client_sample_counts):
        raise ValueError("client_state_dicts and client_sample_counts must be the same length")
    if any(n <= 0 for n in client_sample_counts):
        raise ValueError("client_sample_counts must all be positive")

    total_samples = sum(client_sample_counts)
    weights = [n / total_samples for n in client_sample_counts]
    return weighted_average(client_state_dicts, weights)

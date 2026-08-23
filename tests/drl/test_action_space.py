from __future__ import annotations

import pytest

from src.drl.action_space import (
    NUM_ACTIONS,
    action_to_aggregation_weights,
    profile_name,
    profile_weights,
)
from src.federated.adaptive import RULE_COMPONENTS


def test_exactly_nine_actions_exist():
    assert NUM_ACTIONS == 9


def test_every_profile_has_valid_finite_nonnegative_weights_summing_to_one():
    for action in range(NUM_ACTIONS):
        weights = profile_weights(action)
        assert set(weights.keys()) == set(RULE_COMPONENTS)
        for value in weights.values():
            assert value == value  # not NaN
            assert value >= 0.0
        assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_every_action_produces_eight_valid_normalized_client_weights(synthetic_signals):
    for action in range(NUM_ACTIONS):
        weights = action_to_aggregation_weights(action, synthetic_signals)
        assert len(weights) == 8
        assert set(weights.keys()) == {s.client_id for s in synthetic_signals}
        for w in weights.values():
            assert w == w  # not NaN
            assert w >= 0.0
        assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_different_actions_produce_different_weighting_profiles(synthetic_signals):
    performance_only_weights = action_to_aggregation_weights(0, synthetic_signals)  # performance_only
    resource_only_weights = action_to_aggregation_weights(1, synthetic_signals)  # resource_only

    # Distinct rule weights over non-degenerate, anti-correlated
    # per-client signals (see conftest.synthetic_signals) must produce
    # distinct aggregation-weight vectors that favor different clients.
    assert performance_only_weights != resource_only_weights
    assert max(performance_only_weights, key=performance_only_weights.get) == "SAT-08"
    assert max(resource_only_weights, key=resource_only_weights.get) == "SAT-01"


def test_action_mapping_is_deterministic(synthetic_signals):
    first = action_to_aggregation_weights(3, synthetic_signals)  # combined
    second = action_to_aggregation_weights(3, synthetic_signals)
    assert first == second


def test_action_meaning_is_documented_by_name():
    names = {profile_name(a) for a in range(NUM_ACTIONS)}
    assert names == {
        "performance_only",
        "resource_only",
        "data_only",
        "combined",
        "uniform",
        "fairness_leaning",
        "performance_leaning",
        "resource_leaning",
        "balanced_performance_fairness",
    }


def test_out_of_range_action_rejected(synthetic_signals):
    with pytest.raises(ValueError):
        action_to_aggregation_weights(9, synthetic_signals)
    with pytest.raises(ValueError):
        action_to_aggregation_weights(-1, synthetic_signals)


def test_action_space_never_selects_a_subset_of_clients(synthetic_signals):
    """Weighting-only: every action must involve all 8 clients (no
    selection), even when the resulting weight for one client is very
    small."""
    for action in range(NUM_ACTIONS):
        weights = action_to_aggregation_weights(action, synthetic_signals)
        assert len(weights) == len(synthetic_signals)

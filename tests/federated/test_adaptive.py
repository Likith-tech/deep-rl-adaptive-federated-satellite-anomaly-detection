"""Phase 8 — tests for rule-based (non-RL) adaptive FedAvg: the scoring
rule in isolation, the weighted-average generalization, and the full
adaptive round loop on tiny synthetic data (never the real ~107k-row
NSL-KDD data or the real 8-client partitions)."""

from __future__ import annotations

import torch

from src.federated.adaptive import (
    ClientSignals,
    build_resource_signal,
    compute_client_scores,
    normalize,
    scores_to_aggregation_weights,
    select_top_k_clients,
)
from src.federated.adaptive_trainer import run_adaptive_federated_training
from src.federated.fedavg import federated_average, weighted_average
from tests.federated.conftest import FEATURE_COLUMNS

EQUAL_WEIGHTS = {"performance": 0.25, "data": 0.25, "resource": 0.25, "fairness": 0.25}


def _signals(perf, data, resource, fairness):
    return [
        ClientSignals(client_id=f"SAT-{i:02d}", performance=p, data=d, resource=r, fairness=f)
        for i, (p, d, r, f) in enumerate(zip(perf, data, resource, fairness), start=1)
    ]


# --- normalize --------------------------------------------------------


def test_normalize_maps_min_to_zero_and_max_to_one():
    result = normalize({"a": 10.0, "b": 20.0, "c": 30.0})
    assert result["a"] == 0.0
    assert result["c"] == 1.0
    assert result["b"] == 0.5


def test_normalize_all_equal_returns_neutral_half():
    result = normalize({"a": 5.0, "b": 5.0, "c": 5.0})
    assert all(v == 0.5 for v in result.values())


def test_normalize_empty_returns_empty():
    assert normalize({}) == {}


# --- compute_client_scores / determinism ------------------------------


def test_compute_client_scores_is_deterministic():
    signals = _signals([0.9, 0.5, 0.7], [100, 50, 200], [3.0, 1.0, 2.0], [1.5, 0.2, 1.0])
    scores_a = compute_client_scores(signals, EQUAL_WEIGHTS)
    scores_b = compute_client_scores(signals, EQUAL_WEIGHTS)
    assert scores_a == scores_b


def test_performance_only_rule_ranks_clients_by_performance():
    signals = _signals([0.9, 0.5, 0.7], [10, 10, 10], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    weights = {"performance": 1.0, "data": 0.0, "resource": 0.0, "fairness": 0.0}
    scores = compute_client_scores(signals, weights)
    assert scores["SAT-01"] > scores["SAT-03"] > scores["SAT-02"]


def test_fairness_component_can_outweigh_lower_performance():
    """The core fairness claim: a client with much higher local
    category diversity (entropy) can outrank a slightly-better-
    performing but low-diversity client, once fairness is weighted."""
    signals = _signals(
        perf=[0.80, 0.82],       # SAT-02 performs marginally better
        data=[100, 100],
        resource=[1.0, 1.0],
        fairness=[2.0, 0.1],     # SAT-01 has much more diverse local categories
    )
    fairness_heavy = {"performance": 0.2, "data": 0.0, "resource": 0.0, "fairness": 0.8}
    scores = compute_client_scores(signals, fairness_heavy)
    assert scores["SAT-01"] > scores["SAT-02"]


# --- scores_to_aggregation_weights -------------------------------------


def test_aggregation_weights_sum_to_one_and_are_nonnegative():
    signals = _signals([0.9, 0.1, 0.5], [100, 10, 50], [3.0, 0.5, 1.5], [1.5, 0.0, 0.8])
    scores = compute_client_scores(signals, EQUAL_WEIGHTS)
    weights = scores_to_aggregation_weights(scores)
    assert all(w >= 0 for w in weights.values())
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_aggregation_weights_fallback_to_uniform_when_all_scores_zero():
    scores = {"SAT-01": 0.0, "SAT-02": 0.0, "SAT-03": 0.0}
    weights = scores_to_aggregation_weights(scores)
    assert all(abs(w - 1 / 3) < 1e-9 for w in weights.values())


# --- select_top_k_clients ------------------------------------------------


def test_select_top_k_returns_highest_scoring_clients():
    scores = {"SAT-01": 0.9, "SAT-02": 0.1, "SAT-03": 0.5}
    assert select_top_k_clients(scores, k=2) == ["SAT-01", "SAT-03"]


def test_select_top_k_returns_all_when_k_exceeds_client_count():
    scores = {"SAT-01": 0.9, "SAT-02": 0.1}
    assert set(select_top_k_clients(scores, k=5)) == {"SAT-01", "SAT-02"}


# --- build_resource_signal ------------------------------------------------


def test_build_resource_signal_rewards_lower_latency():
    fast = {"bandwidth_mbps": 50, "latency_ms": 50, "compute_score": 0.5,
            "availability_probability": 0.5, "connectivity_quality": 0.5}
    slow = dict(fast, latency_ms=500)
    assert build_resource_signal(fast) > build_resource_signal(slow)


# --- weighted_average generalizes federated_average -----------------------


def test_weighted_average_matches_federated_average_for_sample_count_weights():
    state_a = {"w": torch.tensor([1.0, 2.0])}
    state_b = {"w": torch.tensor([3.0, 4.0])}
    counts = [10, 30]
    expected = federated_average([state_a, state_b], counts)

    total = sum(counts)
    explicit_weights = [c / total for c in counts]
    actual = weighted_average([state_a, state_b], explicit_weights)

    assert torch.allclose(expected["w"], actual["w"])


def test_weighted_average_rejects_weights_not_summing_to_one():
    state_a = {"w": torch.tensor([1.0])}
    try:
        weighted_average([state_a], [0.5])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_weighted_average_rejects_negative_weights():
    state_a = {"w": torch.tensor([1.0])}
    state_b = {"w": torch.tensor([2.0])}
    try:
        weighted_average([state_a, state_b], [1.5, -0.5])
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- full adaptive round loop (integration, synthetic data) ---------------


def test_adaptive_training_runs_and_all_clients_participate(
    tmp_path, three_clients_and_global_val, tiny_model_config, tiny_training_config
):
    client_ids = three_clients_and_global_val["client_ids"]
    resource_metadata = {
        cid: {"bandwidth_mbps": 50.0, "latency_ms": 100.0, "compute_score": 0.7,
              "availability_probability": 0.8, "connectivity_quality": 0.6}
        for cid in client_ids
    }
    fairness_entropy = {cid: 1.0 for cid in client_ids}

    result = run_adaptive_federated_training(
        client_ids=client_ids,
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=2,
        shared_init_seed=42,
        checkpoint_dir=tmp_path / "checkpoints",
        rule_weights={"performance": 0.4, "data": 0.25, "resource": 0.15, "fairness": 0.2},
        client_resource_metadata=resource_metadata,
        client_fairness_entropy=fairness_entropy,
    )

    assert len(result.round_history) == 2
    for round_record in result.round_history:
        assert set(round_record.aggregation_weights.keys()) == set(client_ids)
        assert abs(sum(round_record.aggregation_weights.values()) - 1.0) < 1e-6
        assert all(w >= 0 for w in round_record.aggregation_weights.values())
        participating_ids = {u["client_id"] for u in round_record.client_updates}
        assert participating_ids == set(client_ids)  # weighting, not selection — all participate


def test_adaptive_training_is_reproducible_with_fixed_seed(
    tmp_path, three_clients_and_global_val, tiny_model_config, tiny_training_config
):
    client_ids = three_clients_and_global_val["client_ids"]
    resource_metadata = {
        cid: {"bandwidth_mbps": 50.0, "latency_ms": 100.0, "compute_score": 0.7,
              "availability_probability": 0.8, "connectivity_quality": 0.6}
        for cid in client_ids
    }
    fairness_entropy = {cid: 1.0 for cid in client_ids}
    rule_weights = {"performance": 0.4, "data": 0.25, "resource": 0.15, "fairness": 0.2}

    kwargs = dict(
        client_ids=client_ids,
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=2,
        shared_init_seed=42,
        rule_weights=rule_weights,
        client_resource_metadata=resource_metadata,
        client_fairness_entropy=fairness_entropy,
    )
    result_a = run_adaptive_federated_training(checkpoint_dir=tmp_path / "run_a", **kwargs)
    result_b = run_adaptive_federated_training(checkpoint_dir=tmp_path / "run_b", **kwargs)

    losses_a = [round(r.val_loss, 6) for r in result_a.round_history]
    losses_b = [round(r.val_loss, 6) for r in result_b.round_history]
    assert losses_a == losses_b
    assert result_a.best_round == result_b.best_round


def test_adaptive_training_produces_different_trajectory_than_equal_weighting(
    tmp_path, three_clients_and_global_val, tiny_model_config, tiny_training_config
):
    """Sanity check that the adaptive rule actually influences training
    — a resource-only rule with deliberately lopsided simulated
    conditions should produce different round-by-round aggregation
    weights than a fairness-only rule."""
    client_ids = three_clients_and_global_val["client_ids"]
    lopsided_resources = {
        client_ids[0]: {"bandwidth_mbps": 100.0, "latency_ms": 10.0, "compute_score": 1.0,
                         "availability_probability": 1.0, "connectivity_quality": 1.0},
        client_ids[1]: {"bandwidth_mbps": 5.0, "latency_ms": 500.0, "compute_score": 0.1,
                         "availability_probability": 0.1, "connectivity_quality": 0.1},
        client_ids[2]: {"bandwidth_mbps": 50.0, "latency_ms": 100.0, "compute_score": 0.5,
                         "availability_probability": 0.5, "connectivity_quality": 0.5},
    }
    fairness_entropy = {client_ids[0]: 0.1, client_ids[1]: 2.0, client_ids[2]: 1.0}

    resource_only = run_adaptive_federated_training(
        client_ids=client_ids,
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=1,
        shared_init_seed=42,
        checkpoint_dir=tmp_path / "resource_only",
        rule_weights={"performance": 0.0, "data": 0.0, "resource": 1.0, "fairness": 0.0},
        client_resource_metadata=lopsided_resources,
        client_fairness_entropy=fairness_entropy,
    )
    fairness_only = run_adaptive_federated_training(
        client_ids=client_ids,
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=1,
        shared_init_seed=42,
        checkpoint_dir=tmp_path / "fairness_only",
        rule_weights={"performance": 0.0, "data": 0.0, "resource": 0.0, "fairness": 1.0},
        client_resource_metadata=lopsided_resources,
        client_fairness_entropy=fairness_entropy,
    )

    resource_weights = resource_only.round_history[0].aggregation_weights
    fairness_weights = fairness_only.round_history[0].aggregation_weights
    # Best-resourced client should win under resource_only...
    assert resource_weights[client_ids[0]] > resource_weights[client_ids[1]]
    # ...but the LEAST-resourced client has the highest entropy, so it
    # should win under fairness_only instead.
    assert fairness_weights[client_ids[1]] > fairness_weights[client_ids[0]]

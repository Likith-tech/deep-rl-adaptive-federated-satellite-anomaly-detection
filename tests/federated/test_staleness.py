"""Phase 9 — tests for staleness- and resource-aware FedAvg: the
participation simulator and weighting rule in isolation, plus the full
staleness-aware round loop on tiny synthetic data (never the real
~107k-row NSL-KDD data or the real 8-client partitions)."""

from __future__ import annotations

import torch

from src.federated.protocol import ClientUpdate
from src.federated.staleness import (
    ParticipationDecision,
    combine_with_sample_weights,
    decide_participation,
    staleness_weight,
)
from src.training.staleness_trainer import run_staleness_federated_training
from tests.federated.conftest import FEATURE_COLUMNS

# --- ClientUpdate.origin_round is additive / backward-compatible ----------


def test_client_update_origin_round_defaults_to_zero():
    update = ClientUpdate(
        client_id="SAT-01", state_dict={}, num_samples=10, local_epochs=1,
        train_loss=0.1, training_seconds=1.0,
    )
    assert update.origin_round == 0
    assert update.to_summary_dict()["origin_round"] == 0


# --- decide_participation ---------------------------------------------


def test_round_one_always_syncs_with_zero_staleness():
    decision = decide_participation(
        client_id="SAT-01", round_number=1, availability_probability=0.0,
        previous_origin_round=0, seed=42,
    )
    assert decision.synced_this_round is True
    assert decision.origin_round == 0
    assert decision.staleness == 0


def test_zero_availability_never_syncs_after_round_one():
    previous_origin_round = 0
    for round_number in range(2, 11):
        decision = decide_participation(
            client_id="SAT-01", round_number=round_number, availability_probability=0.0,
            previous_origin_round=previous_origin_round, seed=42,
        )
        assert decision.synced_this_round is False
        previous_origin_round = decision.origin_round
    # Never having synced since round 1 (origin_round=0), staleness at
    # round 10 should be (10-1) - 0 = 9.
    assert decision.staleness == 9
    assert decision.origin_round == 0


def test_full_availability_always_syncs_with_zero_staleness():
    previous_origin_round = 0
    for round_number in range(2, 11):
        decision = decide_participation(
            client_id="SAT-01", round_number=round_number, availability_probability=1.0,
            previous_origin_round=previous_origin_round, seed=42,
        )
        assert decision.synced_this_round is True
        assert decision.staleness == 0
        previous_origin_round = decision.origin_round


def test_decide_participation_is_deterministic_given_same_seed():
    kwargs = dict(client_id="SAT-03", round_number=5, availability_probability=0.5, previous_origin_round=2, seed=42)
    a = decide_participation(**kwargs)
    b = decide_participation(**kwargs)
    assert a == b


def test_decide_participation_differs_by_client_id_at_same_round():
    """Two clients with the same availability shouldn't be forced into
    lockstep sync/skip patterns — the RNG must be client-specific."""
    outcomes = {
        cid: decide_participation(
            client_id=cid, round_number=3, availability_probability=0.5,
            previous_origin_round=1, seed=42,
        ).synced_this_round
        for cid in [f"SAT-{i:02d}" for i in range(1, 9)]
    }
    # With 8 clients at p=0.5, expect a non-degenerate mix (not all True/all False).
    assert len(set(outcomes.values())) == 2


# --- staleness_weight ---------------------------------------------------


def test_staleness_weight_none_mode_always_one():
    assert staleness_weight(0, "none") == 1.0
    assert staleness_weight(5, "none") == 1.0


def test_staleness_weight_linear_decay():
    assert staleness_weight(0, "linear") == 1.0
    assert staleness_weight(1, "linear") == 0.5
    assert abs(staleness_weight(3, "linear") - 0.25) < 1e-9


def test_staleness_weight_exponential_decay():
    assert staleness_weight(0, "exponential", decay_rate=0.5) == 1.0
    assert staleness_weight(1, "exponential", decay_rate=0.5) == 0.5
    assert abs(staleness_weight(2, "exponential", decay_rate=0.5) - 0.25) < 1e-9


def test_staleness_weight_more_staleness_never_increases_weight():
    for mode in ("linear", "exponential"):
        weights = [staleness_weight(s, mode) for s in range(6)]
        assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))


def test_staleness_weight_rejects_negative_staleness():
    try:
        staleness_weight(-1, "linear")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_staleness_weight_rejects_unknown_mode():
    try:
        staleness_weight(1, "quadratic")
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- combine_with_sample_weights ----------------------------------------


def test_combine_with_sample_weights_sums_to_one():
    counts = {"SAT-01": 100, "SAT-02": 50, "SAT-03": 25}
    stale = {"SAT-01": 1.0, "SAT-02": 0.5, "SAT-03": 1.0}
    weights = combine_with_sample_weights(counts, stale)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(w >= 0 for w in weights.values())


def test_combine_with_sample_weights_discounts_stale_clients():
    counts = {"SAT-01": 100, "SAT-02": 100}
    stale = {"SAT-01": 1.0, "SAT-02": 0.1}
    weights = combine_with_sample_weights(counts, stale)
    assert weights["SAT-01"] > weights["SAT-02"]


def test_combine_with_sample_weights_fallback_to_uniform_when_all_zero():
    counts = {"SAT-01": 100, "SAT-02": 50}
    stale = {"SAT-01": 0.0, "SAT-02": 0.0}
    weights = combine_with_sample_weights(counts, stale)
    assert abs(weights["SAT-01"] - 0.5) < 1e-9
    assert abs(weights["SAT-02"] - 0.5) < 1e-9


# --- full staleness-aware round loop (integration, synthetic data) --------


def test_staleness_training_runs_and_all_clients_always_participate(
    tmp_path, three_clients_and_global_val, tiny_model_config, tiny_training_config
):
    client_ids = three_clients_and_global_val["client_ids"]
    # Deliberately lopsided availability so staleness genuinely varies.
    availability = {client_ids[0]: 1.0, client_ids[1]: 0.0, client_ids[2]: 0.5}

    result = run_staleness_federated_training(
        client_ids=client_ids,
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=4,
        shared_init_seed=42,
        checkpoint_dir=tmp_path / "checkpoints",
        client_availability=availability,
        staleness_mode="linear",
        staleness_decay_rate=0.5,
        participation_seed=42,
    )

    assert len(result.round_history) == 4
    for round_record in result.round_history:
        participating_ids = {u["client_id"] for u in round_record.client_updates}
        assert participating_ids == set(client_ids)  # weighting, not selection — all participate every round
        assert abs(sum(round_record.aggregation_weights.values()) - 1.0) < 1e-6
        assert all(w >= 0 for w in round_record.aggregation_weights.values())
        assert round_record.client_staleness[client_ids[0]] == 0  # availability=1.0 -> always synced

    # availability=0.0 client should accumulate real, positive staleness
    # by the final round (it can never sync after round 1).
    final_round = result.round_history[-1]
    assert final_round.client_staleness[client_ids[1]] > 0


def test_staleness_training_is_reproducible_with_fixed_seed(
    tmp_path, three_clients_and_global_val, tiny_model_config, tiny_training_config
):
    client_ids = three_clients_and_global_val["client_ids"]
    availability = {client_ids[0]: 1.0, client_ids[1]: 0.3, client_ids[2]: 0.6}

    kwargs = dict(
        client_ids=client_ids,
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=3,
        shared_init_seed=42,
        client_availability=availability,
        staleness_mode="exponential",
        staleness_decay_rate=0.5,
        participation_seed=42,
    )
    result_a = run_staleness_federated_training(checkpoint_dir=tmp_path / "run_a", **kwargs)
    result_b = run_staleness_federated_training(checkpoint_dir=tmp_path / "run_b", **kwargs)

    losses_a = [round(r.val_loss, 6) for r in result_a.round_history]
    losses_b = [round(r.val_loss, 6) for r in result_b.round_history]
    assert losses_a == losses_b
    staleness_a = [r.client_staleness for r in result_a.round_history]
    staleness_b = [r.client_staleness for r in result_b.round_history]
    assert staleness_a == staleness_b
    assert result_a.best_round == result_b.best_round


def test_no_correction_mode_ignores_measured_staleness_in_weights(
    tmp_path, three_clients_and_global_val, tiny_model_config, tiny_training_config
):
    """Even though staleness is genuinely measured under `none` mode,
    the aggregation weight should reduce to plain sample-count weighting
    (Phase 6's rule) — the control condition must be a real no-op on the
    weights, not just on paper."""
    client_ids = three_clients_and_global_val["client_ids"]
    availability = {client_ids[0]: 1.0, client_ids[1]: 0.0, client_ids[2]: 0.5}

    result = run_staleness_federated_training(
        client_ids=client_ids,
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=3,
        shared_init_seed=42,
        checkpoint_dir=tmp_path / "checkpoints",
        client_availability=availability,
        staleness_mode="none",
        staleness_decay_rate=0.5,
        participation_seed=42,
    )

    for round_record in result.round_history:
        counts = {u["client_id"]: u["num_samples"] for u in round_record.client_updates}
        total = sum(counts.values())
        expected = {cid: n / total for cid, n in counts.items()}
        for cid in client_ids:
            assert abs(round_record.aggregation_weights[cid] - expected[cid]) < 1e-9
        # Staleness for the never-syncing client must still be > 0 by
        # round 3, EVEN THOUGH it has no effect on the weights above —
        # confirms staleness is measured, not just suppressed.
        if round_record.round_number > 1:
            assert round_record.client_staleness[client_ids[1]] > 0

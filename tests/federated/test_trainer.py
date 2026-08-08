"""Integration tests for the full federated round loop, using tiny
synthetic client partitions — never the real Phase 4 data."""

from __future__ import annotations

import torch

from src.federated.trainer import run_federated_training
from tests.federated.conftest import FEATURE_COLUMNS


def test_runs_the_configured_number_of_rounds(tmp_path, three_clients_and_global_val, tiny_model_config, tiny_training_config):
    result = run_federated_training(
        client_ids=three_clients_and_global_val["client_ids"],
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=2,
        shared_init_seed=42,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert len(result.round_history) == 2
    assert [r.round_number for r in result.round_history] == [1, 2]


def test_all_clients_participate_every_round(tmp_path, three_clients_and_global_val, tiny_model_config, tiny_training_config):
    result = run_federated_training(
        client_ids=three_clients_and_global_val["client_ids"],
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=2,
        shared_init_seed=42,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    for round_record in result.round_history:
        participating_ids = {u["client_id"] for u in round_record.client_updates}
        assert participating_ids == set(three_clients_and_global_val["client_ids"])


def test_global_model_changes_across_rounds(tmp_path, three_clients_and_global_val, tiny_model_config, tiny_training_config):
    round_checkpoints_dir = tmp_path / "checkpoints" / "round_checkpoints"
    run_federated_training(
        client_ids=three_clients_and_global_val["client_ids"],
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=2,
        shared_init_seed=42,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    round_1 = torch.load(round_checkpoints_dir / "round_01.pt", weights_only=False)
    round_2 = torch.load(round_checkpoints_dir / "round_02.pt", weights_only=False)

    any_changed = any(
        not torch.allclose(round_1["model_state_dict"][k], round_2["model_state_dict"][k])
        for k in round_1["model_state_dict"]
    )
    assert any_changed


def test_best_checkpoint_and_history_files_are_written(tmp_path, three_clients_and_global_val, tiny_model_config, tiny_training_config):
    checkpoint_dir = tmp_path / "checkpoints"
    run_federated_training(
        client_ids=three_clients_and_global_val["client_ids"],
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=2,
        shared_init_seed=42,
        checkpoint_dir=checkpoint_dir,
    )

    assert (checkpoint_dir / "best_global_model.pt").exists()
    assert (checkpoint_dir / "round_history.json").exists()


def test_federated_training_is_reproducible_with_fixed_seed(tmp_path, three_clients_and_global_val, tiny_model_config, tiny_training_config):
    result_a = run_federated_training(
        client_ids=three_clients_and_global_val["client_ids"],
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=2,
        shared_init_seed=42,
        checkpoint_dir=tmp_path / "run_a",
    )
    result_b = run_federated_training(
        client_ids=three_clients_and_global_val["client_ids"],
        partitions_dir=three_clients_and_global_val["partitions_dir"],
        global_validation_parquet=three_clients_and_global_val["global_val"],
        feature_columns=FEATURE_COLUMNS,
        model_config=tiny_model_config,
        training_config=tiny_training_config,
        num_rounds=2,
        shared_init_seed=42,
        checkpoint_dir=tmp_path / "run_b",
    )

    losses_a = [round(r.val_loss, 6) for r in result_a.round_history]
    losses_b = [round(r.val_loss, 6) for r in result_b.round_history]
    assert losses_a == losses_b
    assert result_a.best_round == result_b.best_round

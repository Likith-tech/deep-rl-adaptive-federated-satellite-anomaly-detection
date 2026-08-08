from __future__ import annotations

import torch

from src.federated.client import FederatedClient
from src.training.local_trainer import build_initial_state_dict
from tests.federated.conftest import FEATURE_COLUMNS


def test_client_receives_global_model_and_returns_update(three_clients_and_global_val, tiny_model_config, tiny_training_config):
    client = FederatedClient("SAT-A", three_clients_and_global_val["SAT-A"], FEATURE_COLUMNS)
    global_state = build_initial_state_dict(tiny_model_config, seed=42)

    update = client.train_round(global_state, tiny_model_config, tiny_training_config, seed=42)

    assert update.client_id == "SAT-A"
    assert update.num_samples == 120  # SAT-A's own row count
    assert update.local_epochs == tiny_training_config["local_epochs"]
    assert set(update.state_dict.keys()) == set(global_state.keys())


def test_client_only_trains_on_its_own_local_partition(three_clients_and_global_val, tiny_model_config, tiny_training_config):
    client_a = FederatedClient("SAT-A", three_clients_and_global_val["SAT-A"], FEATURE_COLUMNS)
    client_c = FederatedClient("SAT-C", three_clients_and_global_val["SAT-C"], FEATURE_COLUMNS)
    global_state = build_initial_state_dict(tiny_model_config, seed=42)

    update_a = client_a.train_round(global_state, tiny_model_config, tiny_training_config, seed=42)
    update_c = client_c.train_round(global_state, tiny_model_config, tiny_training_config, seed=42)

    assert update_a.num_samples == 120
    assert update_c.num_samples == 30  # not SAT-A's 120


def test_client_local_training_changes_weights_from_global(three_clients_and_global_val, tiny_model_config, tiny_training_config):
    client = FederatedClient("SAT-B", three_clients_and_global_val["SAT-B"], FEATURE_COLUMNS)
    global_state = build_initial_state_dict(tiny_model_config, seed=42)

    update = client.train_round(global_state, tiny_model_config, tiny_training_config, seed=42)

    any_param_changed = any(
        not torch.allclose(update.state_dict[k], global_state[k].cpu())
        for k in global_state
    )
    assert any_param_changed


def test_client_training_is_reproducible_with_fixed_seed(three_clients_and_global_val, tiny_model_config, tiny_training_config):
    client = FederatedClient("SAT-A", three_clients_and_global_val["SAT-A"], FEATURE_COLUMNS)
    global_state = build_initial_state_dict(tiny_model_config, seed=42)

    update_1 = client.train_round(global_state, tiny_model_config, tiny_training_config, seed=42)
    update_2 = client.train_round(global_state, tiny_model_config, tiny_training_config, seed=42)

    for key in update_1.state_dict:
        assert torch.allclose(update_1.state_dict[key], update_2.state_dict[key])

"""Focused tests for `src.federated.drl_round_provider` — the REAL
NSL-KDD adapter around the unmodified Phase 6/8 `FederatedClient`/
`FederatedServer`. Uses the REAL Phase 4/6 partition
(`data/partitions/`) and REAL global validation split
(`data/processed/validation.parquet`) — this is intentional and
required to validate the provider against real data, per this turn's
explicit scope. KDDTest+ (`data/processed/test.parquet`) is never
referenced anywhere in this file or in the module under test.

Real local training runs are kept to the minimum needed to prove
correctness: one shared provider fixture (module-scoped) is reused
across most tests; a second, separately-built provider is used only
for the one explicit determinism check.
"""

from __future__ import annotations

import ast
import inspect
import time
from pathlib import Path

import numpy as np
import pytest

from src.drl.action_space import action_to_aggregation_weights
from src.federated import drl_round_provider as drl_round_provider_module
from src.federated.drl_round_provider import RealNSLKDDRoundProvider, build_real_round_provider

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_ROUNDS = 2  # minimum needed to exercise round_provider's "continue" branch once


def _model_config() -> dict:
    return {"type": "mlp", "input_dim": 121, "hidden_dimensions": [128, 64], "dropout": 0.3, "output_dim": 1}


def _training_config() -> dict:
    return {"local_epochs": 1, "batch_size": 128, "learning_rate": 0.001, "optimizer": "adam", "loss": "bce_with_logits", "seed": 42}


def _skip_if_data_missing():
    partitions_dir = REPO_ROOT / "data" / "partitions"
    validation_path = REPO_ROOT / "data" / "processed" / "validation.parquet"
    if not partitions_dir.exists() or not validation_path.exists():
        pytest.skip("real Phase 4/6 data/partitions or data/processed/validation.parquet not present")


@pytest.fixture(scope="module")
def real_provider() -> RealNSLKDDRoundProvider:
    _skip_if_data_missing()
    return build_real_round_provider(
        repo_root=REPO_ROOT, model_config=_model_config(), training_config=_training_config(),
        shared_init_seed=42, num_rounds=NUM_ROUNDS,
    )


# ---------------------------------------------------------------------------
# A. Provider construction
# ---------------------------------------------------------------------------


def test_provider_has_exactly_eight_clients(real_provider):
    assert len(real_provider.clients) == 8
    assert real_provider.client_ids == sorted(real_provider.client_ids)


def test_provider_client_partition_paths_are_correct_and_isolated(real_provider):
    seen_paths = set()
    for client in real_provider.clients:
        assert client.train_parquet == REPO_ROOT / "data" / "partitions" / client.client_id / "train.parquet"
        assert client.train_parquet.exists()
        assert client.train_parquet not in seen_paths  # no client shares another's path
        seen_paths.add(client.train_parquet)


def test_provider_validation_dataloader_is_loaded(real_provider):
    assert real_provider.val_loader is not None
    total = sum(len(batch_y) for _, batch_y in real_provider.val_loader)
    assert total > 0


def test_provider_construction_is_deterministic_given_same_seed():
    _skip_if_data_missing()
    provider_a = build_real_round_provider(REPO_ROOT, _model_config(), _training_config(), shared_init_seed=42, num_rounds=NUM_ROUNDS)
    provider_b = build_real_round_provider(REPO_ROOT, _model_config(), _training_config(), shared_init_seed=42, num_rounds=NUM_ROUNDS)
    initial_a = provider_a.server.get_global_state()
    initial_b = provider_b.server.get_global_state()
    for key in initial_a:
        assert np.allclose(initial_a[key].numpy(), initial_b[key].numpy())


def test_build_real_round_provider_raises_clear_error_when_data_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="required Phase 4/6 artifact not found"):
        build_real_round_provider(tmp_path, _model_config(), _training_config(), shared_init_seed=42, num_rounds=NUM_ROUNDS)


# ---------------------------------------------------------------------------
# B. Privacy
# ---------------------------------------------------------------------------


def test_module_source_never_imports_pandas():
    source = inspect.getsource(drl_round_provider_module)
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "pandas" not in imported
    assert "pyarrow" not in imported


def test_module_source_never_calls_read_parquet_read_csv_or_opens_files_directly():
    source = inspect.getsource(drl_round_provider_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("read_parquet", "read_csv"):
            pytest.fail(f"drl_round_provider.py must not call {node.attr} — privacy boundary violation")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
            pytest.fail("drl_round_provider.py must not call open() directly — data access must go through FederatedClient")


def test_module_never_constructs_a_pandas_dataframe():
    source = inspect.getsource(drl_round_provider_module)
    assert "DataFrame" not in source
    assert "pd." not in source


def test_module_never_references_kddtest():
    """No literal path to the held-out test split appears anywhere in
    the module's executable code. (The module's own docstring contains
    the word "KDDTest+" only to explicitly DISCLOSE that it is never
    referenced — checking for that word itself would be a
    self-defeating false positive, so this checks the concrete file
    path instead, the same pattern used for `data/partitions`/
    `train.parquet` checks in tests/drl/test_privacy_boundary.py.)"""
    source = inspect.getsource(drl_round_provider_module)
    assert "test.parquet" not in source
    assert "data/processed/test" not in source


def test_server_privacy_boundary_test_still_applies_unmodified():
    """Sanity: the provider reuses `FederatedServer` unmodified, so its
    existing AST-based privacy test (Phase 6) still governs it."""
    from src.federated import server as server_module

    source = inspect.getsource(server_module)
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "pandas" not in imported


# ---------------------------------------------------------------------------
# C. Client participation
# ---------------------------------------------------------------------------


def test_initial_signals_provider_trains_and_returns_all_eight_clients(real_provider):
    signals, initial_val_f1 = real_provider.initial_signals_provider()
    assert len(signals) == 8
    assert {s.client_id for s in signals} == set(real_provider.client_ids)
    assert np.isfinite(initial_val_f1)


def test_no_client_selection_path_exists_in_provider_source():
    source = inspect.getsource(drl_round_provider_module)
    assert "select_top_k_clients" not in source
    assert "drop" not in source.lower()


# ---------------------------------------------------------------------------
# D. Round behavior (one real round, minimum needed to prove the contract)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_real_round_result(real_provider):
    signals, initial_val_f1 = real_provider.initial_signals_provider()
    action = 3  # "combined" — Phase 8's own primary rule, used as a fixed, non-DQN action for this correctness check
    weights = action_to_aggregation_weights(action, signals)
    next_signals, new_val_f1 = real_provider.round_provider(1, weights, signals)
    return {
        "initial_val_f1": initial_val_f1, "signals": signals, "weights": weights,
        "next_signals": next_signals, "new_val_f1": new_val_f1,
    }


def test_every_client_has_all_four_signal_components(one_real_round_result):
    for s in one_real_round_result["signals"]:
        assert np.isfinite(s.performance)
        assert np.isfinite(s.data) and s.data > 0
        assert np.isfinite(s.resource)
        assert np.isfinite(s.fairness)


def test_round_provider_accepts_valid_weights_and_they_sum_to_one(one_real_round_result):
    weights = one_real_round_result["weights"]
    assert len(weights) == 8
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(w >= 0.0 for w in weights.values())


def test_round_provider_returns_next_signals_and_finite_validation_f1(one_real_round_result):
    assert len(one_real_round_result["next_signals"]) == 8
    assert np.isfinite(one_real_round_result["new_val_f1"])
    assert 0.0 <= one_real_round_result["new_val_f1"] <= 1.0


def test_server_round_number_incremented_after_real_aggregation(real_provider):
    assert real_provider.server.round_number >= 1


# ---------------------------------------------------------------------------
# E. Determinism (separate providers, same seed)
# ---------------------------------------------------------------------------


def test_initial_val_f1_is_reproducible_across_separately_built_providers():
    _skip_if_data_missing()
    provider_a = build_real_round_provider(REPO_ROOT, _model_config(), _training_config(), shared_init_seed=42, num_rounds=NUM_ROUNDS)
    provider_b = build_real_round_provider(REPO_ROOT, _model_config(), _training_config(), shared_init_seed=42, num_rounds=NUM_ROUNDS)

    signals_a, val_f1_a = provider_a.initial_signals_provider()
    signals_b, val_f1_b = provider_b.initial_signals_provider()

    assert val_f1_a == pytest.approx(val_f1_b, abs=1e-6)
    performance_a = sorted((s.client_id, s.performance) for s in signals_a)
    performance_b = sorted((s.client_id, s.performance) for s in signals_b)
    for (cid_a, perf_a), (cid_b, perf_b) in zip(performance_a, performance_b):
        assert cid_a == cid_b
        assert perf_a == pytest.approx(perf_b, abs=1e-6)

from __future__ import annotations

import ast
import inspect

import pytest
import torch

from src.federated import server as server_module
from src.federated.protocol import ClientUpdate
from src.federated.server import FederatedServer


def _make_update(client_id: str, value: float, num_samples: int) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        state_dict={"w": torch.tensor([value, value])},
        num_samples=num_samples,
        local_epochs=1,
        train_loss=0.1,
        training_seconds=0.01,
    )


def test_server_distributes_identical_global_state_to_every_caller():
    initial_state = {"w": torch.tensor([1.0, 2.0])}
    server = FederatedServer(initial_state)

    state_1 = server.get_global_state()
    state_2 = server.get_global_state()

    assert torch.equal(state_1["w"], state_2["w"])
    # Must be independent copies, not the same tensor object (a client
    # mutating its copy must not corrupt the server's global state).
    state_1["w"] += 100
    assert torch.equal(server.global_state_dict["w"], torch.tensor([1.0, 2.0]))


def test_server_aggregates_client_updates_via_fedavg():
    initial_state = {"w": torch.tensor([0.0, 0.0])}
    server = FederatedServer(initial_state)

    updates = [_make_update("SAT-A", value=1.0, num_samples=10), _make_update("SAT-B", value=5.0, num_samples=30)]
    new_state = server.aggregate(updates)

    expected = 0.25 * 1.0 + 0.75 * 5.0
    assert torch.allclose(new_state["w"], torch.tensor([expected, expected]))
    assert torch.allclose(server.global_state_dict["w"], torch.tensor([expected, expected]))


def test_server_round_number_increments_on_aggregate():
    server = FederatedServer({"w": torch.tensor([0.0])})
    assert server.round_number == 0
    server.aggregate([_make_update("SAT-A", 1.0, 10)])
    assert server.round_number == 1
    server.aggregate([_make_update("SAT-A", 2.0, 10)])
    assert server.round_number == 2


def test_server_module_never_imports_pandas_or_touches_parquet():
    """Privacy boundary: the server aggregation module must not be able
    to read raw client records — verified by inspecting its actual
    import statements (not prose in comments/docstrings)."""
    source = inspect.getsource(server_module)
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module.split(".")[0])

    assert "pandas" not in imported_names
    assert "pyarrow" not in imported_names
    # No function in this module should reference reading a parquet/csv file.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("read_parquet", "read_csv"):
            pytest.fail(f"server.py must not call {node.attr} — privacy boundary violation")

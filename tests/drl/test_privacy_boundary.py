"""Phase 9B privacy boundary — same style as
tests/federated/test_server.py's
test_server_module_never_imports_pandas_or_touches_parquet: verified by
inspecting actual import statements and call sites (AST), not prose in
comments/docstrings.

Every module the reward/state/action/DQN pipeline touches must be
unable to read a client's raw dataset."""

from __future__ import annotations

import ast
import inspect

import pytest

from src.drl import action_space as action_space_module
from src.drl import dqn_agent as dqn_agent_module
from src.drl import replay_buffer as replay_buffer_module
from src.drl import reward as reward_module
from src.drl import state as state_module
from src.federated import dqn_controller as dqn_controller_module
from src.federated import drl_environment as drl_environment_module

MODULES_UNDER_TEST = (
    reward_module,
    state_module,
    action_space_module,
    dqn_agent_module,
    replay_buffer_module,
    drl_environment_module,
    dqn_controller_module,
)


def _imported_top_level_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("module", MODULES_UNDER_TEST, ids=[m.__name__ for m in MODULES_UNDER_TEST])
def test_module_never_imports_pandas_or_pyarrow(module):
    source = inspect.getsource(module)
    tree = ast.parse(source)
    imported = _imported_top_level_names(tree)
    assert "pandas" not in imported, f"{module.__name__} must not import pandas — privacy boundary violation"
    assert "pyarrow" not in imported, f"{module.__name__} must not import pyarrow — privacy boundary violation"


@pytest.mark.parametrize("module", MODULES_UNDER_TEST, ids=[m.__name__ for m in MODULES_UNDER_TEST])
def test_module_never_calls_read_parquet_or_read_csv(module):
    source = inspect.getsource(module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("read_parquet", "read_csv"):
            pytest.fail(f"{module.__name__} must not call {node.attr} — privacy boundary violation")


@pytest.mark.parametrize("module", MODULES_UNDER_TEST, ids=[m.__name__ for m in MODULES_UNDER_TEST])
def test_module_never_opens_a_file_path_under_data_partitions(module):
    """No literal reference to a client partition path should appear in
    the reward/state/action/DQN code at all — these modules should
    operate purely on already-aggregated signals, never on a path."""
    source = inspect.getsource(module)
    assert "data/partitions" not in source
    assert "train.parquet" not in source


def test_reward_function_signature_takes_no_path_or_dataframe_like_argument():
    """Belt-and-braces: inspect compute_reward's parameter names — none
    should suggest a file path or raw dataset is expected."""
    sig = inspect.signature(reward_module.compute_reward)
    suspicious = {"path", "parquet", "dataframe", "df", "dataset", "file"}
    for param_name in sig.parameters:
        assert not any(word in param_name.lower() for word in suspicious), (
            f"compute_reward parameter {param_name!r} suggests raw-data access — privacy boundary risk"
        )

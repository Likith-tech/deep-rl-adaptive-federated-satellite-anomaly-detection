"""Focused tests for scripts/run_phase9b_improvement_01.py — the ONE
controlled variable (per-episode local-training seed) and its
artifact-path separation from the original, frozen Phase 9B run.
Does not require the real dataset (imports the script as a module and
tests its pure logic); a separate real-data pilot (not a pytest test)
verifies the end-to-end real-data behavior."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_phase9b_improvement_01 as improvement_module  # noqa: E402


def test_episode_training_config_varies_seed_per_episode_deterministically():
    cfg_1a = improvement_module.episode_training_config(1)
    cfg_1b = improvement_module.episode_training_config(1)
    cfg_2 = improvement_module.episode_training_config(2)

    assert cfg_1a["seed"] == cfg_1b["seed"]  # deterministic given the same episode number
    assert cfg_1a["seed"] != cfg_2["seed"]   # genuinely varies across episodes
    assert cfg_1a["seed"] == improvement_module.BASE_LOCAL_TRAINING_SEED + 1
    assert cfg_2["seed"] == improvement_module.BASE_LOCAL_TRAINING_SEED + 2


def test_episode_training_config_changes_only_the_seed():
    base = improvement_module.BASE_TRAINING_CONFIG
    cfg = improvement_module.episode_training_config(5)
    for key in base:
        if key == "seed":
            continue
        assert cfg[key] == base[key], f"{key} must be unchanged from configs/drl.yaml — only seed is the controlled variable"


def test_shared_init_seed_and_budget_match_the_original_run_for_a_fair_comparison():
    assert improvement_module.SHARED_INIT_SEED == 42  # identical starting model every episode, same as the original run
    assert improvement_module.EPISODES == 50
    assert improvement_module.ROUNDS_PER_EPISODE == 15
    assert improvement_module.FINAL_RUN_ROUNDS == 10


def test_final_run_seed_is_the_original_constant_not_episode_varied():
    """The seed-diversity manipulation (H1's independent variable) is a
    PRETRAINING-only intervention — the final frozen-policy run must
    use the exact same seed=42 protocol as the original run, or the
    comparison would no longer isolate one variable."""
    source = inspect.getsource(improvement_module.run_final_frozen_evaluation)
    assert 'final_training_config["seed"] = BASE_LOCAL_TRAINING_SEED' in source


def test_artifact_paths_are_separate_from_the_original_run():
    assert "improvement_01" in str(improvement_module.OUTPUT_DIR)
    assert "improvement_01" in str(improvement_module.PLOTS_DIR)
    assert "improvement_01" in str(improvement_module.REPORT_PATH)
    assert improvement_module.OUTPUT_DIR != REPO_ROOT / "experiments" / "drl_adaptive_fl" / "full_run"
    assert improvement_module.REPORT_PATH != REPO_ROOT / "results" / "reports" / "drl_adaptive_fl_results.md"


def test_module_never_references_kddtest_path():
    source = inspect.getsource(improvement_module)
    assert "test.parquet" not in source.split("def _evaluate_on_parquet")[0]  # not referenced before the one designated eval call site


def test_module_never_imports_pandas_directly():
    source = inspect.getsource(improvement_module)
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "pandas" not in imported


def test_reused_infrastructure_is_unmodified_by_import_identity():
    """Sanity: this script imports the SAME classes/functions the
    original run used — no parallel/forked FL implementation."""
    from src.federated import drl_environment, drl_round_provider
    from src.federated.dqn_controller import DQNController as OriginalDQNController
    from src.federated.drl_environment import DRLFederatedEnvironment as OriginalEnv
    from src.federated.drl_round_provider import build_real_round_provider as original_factory

    assert improvement_module.DQNController is OriginalDQNController
    assert improvement_module.DRLFederatedEnvironment is OriginalEnv
    assert improvement_module.build_real_round_provider is original_factory

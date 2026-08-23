from __future__ import annotations

from pathlib import Path

import pytest

from src.drl.config import load_drl_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DRL_CONFIG_PATH = REPO_ROOT / "configs" / "drl.yaml"


def test_configs_drl_yaml_loads_and_validates():
    config = load_drl_config(DRL_CONFIG_PATH)
    assert config["state"]["dimension"] == 44
    assert config["action"]["count"] == 9
    assert config["federation"]["num_clients"] == 8
    assert config["reproducibility"]["seed"] == 42


def test_load_drl_config_rejects_missing_sections(tmp_path):
    bad_config = tmp_path / "bad_drl.yaml"
    bad_config.write_text("state:\n  dimension: 44\n")
    with pytest.raises(ValueError, match="missing required"):
        load_drl_config(bad_config)


def test_load_drl_config_rejects_wrong_state_dimension(tmp_path):
    bad_config = tmp_path / "bad_drl.yaml"
    bad_config.write_text(
        "reproducibility: {seed: 42}\n"
        "federation: {num_clients: 8}\n"
        "model: {}\n"
        "state: {dimension: 10}\n"
        "action: {count: 9}\n"
        "reward: {}\n"
        "dqn: {}\n"
        "paths: {}\n"
        "data: {}\n"
    )
    with pytest.raises(ValueError, match="state.dimension"):
        load_drl_config(bad_config)


def test_load_drl_config_rejects_wrong_action_count(tmp_path):
    bad_config = tmp_path / "bad_drl.yaml"
    bad_config.write_text(
        "reproducibility: {seed: 42}\n"
        "federation: {num_clients: 8}\n"
        "model: {}\n"
        "state: {dimension: 44}\n"
        "action: {count: 4}\n"
        "reward: {}\n"
        "dqn: {}\n"
        "paths: {}\n"
        "data: {}\n"
    )
    with pytest.raises(ValueError, match="action.count"):
        load_drl_config(bad_config)

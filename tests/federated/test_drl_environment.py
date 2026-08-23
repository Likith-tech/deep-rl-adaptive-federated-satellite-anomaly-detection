"""Environment reset()/step() tests — entirely on deterministic
synthetic fixtures. No real dataset, no client parquet file, no
FederatedClient/FederatedServer training is exercised here."""

from __future__ import annotations

import numpy as np
import pytest

from src.drl.state import STATE_DIM
from src.federated.adaptive import ClientSignals
from src.federated.drl_environment import DRLFederatedEnvironment, StepResult

NUM_ROUNDS = 3


def _synthetic_signals(round_number: int) -> list[ClientSignals]:
    rng = np.random.default_rng(100 + round_number)
    return [
        ClientSignals(
            client_id=f"SAT-{i + 1:02d}",
            performance=float(rng.uniform(0.6, 0.95)),
            data=float(rng.uniform(1000, 10000)),
            resource=float(rng.uniform(0.2, 1.0)),
            fairness=float(rng.uniform(0.5, 3.0)),
        )
        for i in range(8)
    ]


def _initial_signals_provider():
    return _synthetic_signals(1), 0.70


def _deterministic_round_provider(round_number, weights, signals):
    # Purely deterministic function of (round_number, weights) — never
    # reads a file, never touches raw data.
    weight_sum_bonus = sum(w * w for w in weights.values())  # arbitrary deterministic function of the weights
    new_val_f1 = 0.70 + 0.01 * round_number + 0.05 * weight_sum_bonus
    return _synthetic_signals(round_number + 1), new_val_f1


def _make_env(lambda_fair=0.1, lambda_stab=0.05, num_rounds=NUM_ROUNDS):
    return DRLFederatedEnvironment(
        initial_signals_provider=_initial_signals_provider,
        round_provider=_deterministic_round_provider,
        num_rounds=num_rounds,
        lambda_fair=lambda_fair,
        lambda_stab=lambda_stab,
    )


def test_reset_returns_correctly_shaped_first_state():
    env = _make_env()
    state = env.reset()
    assert state.shape == (STATE_DIM,) == (44,)
    assert np.all(np.isfinite(state))
    # First round: no previous action -> one-hot block all zero.
    assert np.all(state[-9:] == 0.0)


def test_reset_is_deterministic():
    env_a = _make_env()
    env_b = _make_env()
    assert np.array_equal(env_a.reset(), env_b.reset())


def test_step_returns_step_result_with_expected_fields():
    env = _make_env()
    env.reset()
    result = env.step(action=3)
    assert isinstance(result, StepResult)
    assert result.observation.shape == (44,)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert "weights" in result.info and "action" in result.info


def test_step_is_deterministic_given_the_same_actions():
    env_a = _make_env()
    env_a.reset()
    result_a = env_a.step(3)

    env_b = _make_env()
    env_b.reset()
    result_b = env_b.step(3)

    assert np.array_equal(result_a.observation, result_b.observation)
    assert result_a.reward == pytest.approx(result_b.reward)


def test_episode_terminates_after_configured_number_of_rounds():
    env = _make_env(num_rounds=2)
    env.reset()
    result_1 = env.step(0)
    assert result_1.done is False
    result_2 = env.step(0)
    assert result_2.done is True


def test_step_before_reset_raises():
    env = _make_env()
    with pytest.raises(RuntimeError):
        env.step(0)


def test_step_weights_are_valid_aggregation_weights():
    env = _make_env()
    env.reset()
    result = env.step(4)
    weights = result.info["weights"]
    assert len(weights) == 8
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert all(w >= 0.0 for w in weights.values())

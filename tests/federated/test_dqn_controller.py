from __future__ import annotations

import numpy as np
import pytest
import torch

from src.federated.dqn_controller import DQNController


def _make_controller(**overrides):
    kwargs = dict(
        input_dim=44, output_dim=9, hidden_dimensions=(16, 16), gamma=0.95, learning_rate=0.01,
        replay_buffer_size=50, batch_size=4, target_update_frequency=3,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10, seed=42,
    )
    kwargs.update(overrides)
    return DQNController(**kwargs)


def test_choose_action_returns_valid_action():
    controller = _make_controller()
    state = np.zeros(44, dtype=np.float32)
    action = controller.choose_action(state)
    assert 0 <= action < 9


def test_choose_action_is_reproducible_with_fresh_seeded_controllers():
    state = np.random.default_rng(0).normal(size=44).astype(np.float32)
    controller_a = _make_controller()
    controller_b = _make_controller()
    actions_a = [controller_a.choose_action(state) for _ in range(5)]
    actions_b = [controller_b.choose_action(state) for _ in range(5)]
    assert actions_a == actions_b


def test_epsilon_decays_from_start_to_end():
    controller = _make_controller(epsilon_decay_steps=10)
    assert controller.epsilon() == pytest.approx(1.0)
    for _ in range(10):
        controller.remember(np.zeros(44, dtype=np.float32), 0, 0.0, np.zeros(44, dtype=np.float32), False)
    assert controller.epsilon() == pytest.approx(0.05)


def test_remember_updates_target_network_at_configured_frequency():
    controller = _make_controller(target_update_frequency=3)
    rng = np.random.default_rng(5)

    # Diverge the main network from the target directly (bypassing
    # remember()'s own warmup/frequency gating), so we can test the
    # sync trigger in isolation.
    for _ in range(6):
        controller.buffer.push(rng.normal(size=44).astype(np.float32), 0, 1.0, rng.normal(size=44).astype(np.float32), False)
    controller.agent.learn(controller.buffer.sample(4, rng))

    diverged = any(
        not torch.equal(p_main, p_target)
        for p_main, p_target in zip(controller.agent.network.parameters(), controller.agent.target_network.parameters())
    )
    assert diverged  # sanity: the manual learn() step actually diverged the networks

    # Advance remember() exactly target_update_frequency times -> exactly one sync.
    for _ in range(3):
        controller.remember(rng.normal(size=44).astype(np.float32), 0, 1.0, rng.normal(size=44).astype(np.float32), False)

    for p_main, p_target in zip(controller.agent.network.parameters(), controller.agent.target_network.parameters()):
        assert torch.equal(p_main, p_target)


def test_maybe_learn_returns_none_below_replay_warmup():
    controller = _make_controller()
    controller.remember(np.zeros(44, dtype=np.float32), 0, 1.0, np.zeros(44, dtype=np.float32), False)
    assert controller.maybe_learn(replay_warmup=10) is None


def test_maybe_learn_returns_loss_once_warmup_reached():
    controller = _make_controller(batch_size=4)
    for i in range(5):
        controller.remember(np.random.default_rng(i).normal(size=44).astype(np.float32), i % 9, 1.0, np.random.default_rng(i + 1).normal(size=44).astype(np.float32), False)
    loss = controller.maybe_learn(replay_warmup=4)
    assert loss is not None
    assert loss >= 0.0

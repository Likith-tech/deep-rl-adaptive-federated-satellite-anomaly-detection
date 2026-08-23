from __future__ import annotations

import numpy as np
import pytest
import torch

from src.drl.dqn_agent import DQNAgent, QNetwork
from src.drl.replay_buffer import ReplayBuffer


def test_network_input_dimension_is_44():
    agent = DQNAgent(seed=42)
    assert agent.network.input_dim == 44


def test_network_output_dimension_is_9():
    agent = DQNAgent(seed=42)
    assert agent.network.output_dim == 9


def test_forward_pass_produces_correct_shape():
    agent = DQNAgent(seed=42)
    state = np.zeros(44, dtype=np.float32)
    q = agent.q_values(state)
    assert q.shape == (9,)
    assert np.all(np.isfinite(q))


def test_forward_pass_rejects_wrong_input_shape():
    agent = DQNAgent(seed=42)
    with pytest.raises(ValueError):
        agent.q_values(np.zeros(10, dtype=np.float32))


def test_greedy_action_selection_is_deterministic_argmax():
    agent = DQNAgent(seed=42)
    state = np.random.default_rng(0).normal(size=44).astype(np.float32)
    rng = np.random.default_rng(1)
    action_1 = agent.act(state, epsilon=0.0, rng=rng)
    action_2 = agent.act(state, epsilon=0.0, rng=rng)
    assert action_1 == action_2
    assert 0 <= action_1 < 9
    assert action_1 == int(np.argmax(agent.q_values(state)))


def test_seeded_action_selection_is_reproducible_across_agents():
    state = np.random.default_rng(0).normal(size=44).astype(np.float32)

    agent_a = DQNAgent(seed=42)
    agent_b = DQNAgent(seed=42)

    # Same network seed -> identical weights -> identical greedy action.
    action_a = agent_a.act(state, epsilon=0.0, rng=np.random.default_rng(7))
    action_b = agent_b.act(state, epsilon=0.0, rng=np.random.default_rng(7))
    assert action_a == action_b

    # Same exploration rng seed, epsilon=1 (always explore) -> identical
    # sequence of "random" actions.
    seq_a = [agent_a.act(state, epsilon=1.0, rng=np.random.default_rng(99)) for _ in range(5)]
    seq_b = [agent_b.act(state, epsilon=1.0, rng=np.random.default_rng(99)) for _ in range(5)]
    assert seq_a == seq_b


def test_different_seeds_produce_different_network_weights():
    agent_a = DQNAgent(seed=1)
    agent_b = DQNAgent(seed=2)
    weights_a = next(agent_a.network.parameters())
    weights_b = next(agent_b.network.parameters())
    assert not (weights_a == weights_b).all()


def test_qnetwork_default_dimensions_match_pilot_spec():
    net = QNetwork()
    assert net.input_dim == 44
    assert net.output_dim == 9


def test_target_network_starts_synced_with_main_network():
    agent = DQNAgent(seed=42)
    for p_main, p_target in zip(agent.network.parameters(), agent.target_network.parameters()):
        assert torch.equal(p_main, p_target)


def test_target_network_update_syncs_weights():
    agent = DQNAgent(seed=42)
    # Force the main network to diverge from the target via a manual learn() step.
    buffer = ReplayBuffer(capacity=10)
    state = np.zeros(44, dtype=np.float32)
    next_state = np.ones(44, dtype=np.float32)
    for i in range(5):
        buffer.push(state, i % 9, 1.0, next_state, False)
    agent.learn(buffer.sample(5, rng=np.random.default_rng(0)))

    diverged = any(
        not torch.equal(p_main, p_target)
        for p_main, p_target in zip(agent.network.parameters(), agent.target_network.parameters())
    )
    assert diverged  # sanity: the learn() step actually changed the main network

    agent.update_target_network()
    for p_main, p_target in zip(agent.network.parameters(), agent.target_network.parameters()):
        assert torch.equal(p_main, p_target)


def test_learn_reduces_loss_over_repeated_steps_on_a_fixed_batch():
    agent = DQNAgent(seed=42, learning_rate=0.01)
    buffer = ReplayBuffer(capacity=10)
    rng = np.random.default_rng(0)
    for i in range(8):
        buffer.push(rng.normal(size=44).astype(np.float32), i % 9, 1.0, rng.normal(size=44).astype(np.float32), False)
    batch = buffer.sample(8, rng=np.random.default_rng(1))

    first_loss = agent.learn(batch)
    for _ in range(20):
        last_loss = agent.learn(batch)
    assert last_loss < first_loss


def test_learn_rejects_empty_batch():
    agent = DQNAgent(seed=42)
    with pytest.raises(ValueError):
        agent.learn([])

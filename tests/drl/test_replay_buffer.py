from __future__ import annotations

import numpy as np
import pytest

from src.drl.replay_buffer import ReplayBuffer


def _push_n(buffer: ReplayBuffer, n: int) -> None:
    for i in range(n):
        buffer.push(state=np.full(4, i, dtype=np.float32), action=i % 9, reward=float(i), next_state=np.full(4, i + 1, dtype=np.float32), done=False)


def test_buffer_stores_transitions_up_to_capacity():
    buffer = ReplayBuffer(capacity=5)
    _push_n(buffer, 3)
    assert len(buffer) == 3
    _push_n(buffer, 10)
    assert len(buffer) == 5  # capacity enforced, oldest evicted


def test_buffer_sample_returns_requested_batch_size():
    buffer = ReplayBuffer(capacity=10)
    _push_n(buffer, 10)
    batch = buffer.sample(4, rng=np.random.default_rng(0))
    assert len(batch) == 4
    for t in batch:
        assert t.state.shape == (4,)


def test_buffer_sample_is_reproducible_with_seeded_rng():
    buffer = ReplayBuffer(capacity=10)
    _push_n(buffer, 10)
    batch_a = buffer.sample(5, rng=np.random.default_rng(7))
    batch_b = buffer.sample(5, rng=np.random.default_rng(7))
    assert [t.action for t in batch_a] == [t.action for t in batch_b]


def test_buffer_sample_rejects_batch_larger_than_buffer():
    buffer = ReplayBuffer(capacity=10)
    _push_n(buffer, 3)
    with pytest.raises(ValueError):
        buffer.sample(4, rng=np.random.default_rng(0))


def test_buffer_rejects_nonpositive_capacity():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)

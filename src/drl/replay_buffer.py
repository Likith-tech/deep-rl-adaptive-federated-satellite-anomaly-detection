"""Phase 9B — a minimal fixed-capacity replay buffer for the DQN pilot.

Pure in-memory structure: stores `(state, action, reward, next_state,
done)` transitions and samples uniformly at random via a caller-supplied
`numpy.random.Generator` for reproducibility. No file I/O, no client
data of any kind.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self._buffer.append(Transition(state=np.asarray(state), action=int(action), reward=float(reward), next_state=np.asarray(next_state), done=bool(done)))

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[Transition]:
        if batch_size > len(self._buffer):
            raise ValueError(f"cannot sample {batch_size} transitions from a buffer of size {len(self._buffer)}")
        indices = rng.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[int(i)] for i in indices]

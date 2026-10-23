"""Phase 9B — the pilot DQN interface (see
docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md
section 12).

Implements the network, epsilon-greedy action selection, a target
network, and a minimal `learn()` gradient step against a
`src.drl.replay_buffer.Transition` batch — enough to prove the DQN's
mechanics (forward pass, target sync, a real gradient update) work
correctly on tiny synthetic data. This is still PILOT SCOPE: no large
pretraining budget is run from this module — see
docs/project-progress/12-phase-9b-drl-pilot.md.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from src.drl.replay_buffer import Transition

INPUT_DIM = 44
OUTPUT_DIM = 9
DEFAULT_HIDDEN_DIMENSIONS = (64, 64)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNetwork(nn.Module):
    """Small MLP Q-network: input_dim -> hidden... -> output_dim, ReLU
    between hidden layers, linear output (raw Q-values, no activation)."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dimensions: tuple[int, ...] = DEFAULT_HIDDEN_DIMENSIONS,
        output_dim: int = OUTPUT_DIM,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dimensions:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Wraps a `QNetwork` + target `QNetwork` with epsilon-greedy action
    selection and a minimal DQN `learn()` step."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dimensions: tuple[int, ...] = DEFAULT_HIDDEN_DIMENSIONS,
        output_dim: int = OUTPUT_DIM,
        seed: int = 42,
        gamma: float = 0.95,
        learning_rate: float = 1e-3,
    ) -> None:
        set_seed(seed)
        self.network = QNetwork(input_dim, hidden_dimensions, output_dim)
        self.target_network = QNetwork(input_dim, hidden_dimensions, output_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.output_dim = output_dim
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.learn_step_count = 0

    def q_values(self, state: np.ndarray, network: nn.Module | None = None) -> np.ndarray:
        network = network or self.network
        if state.shape != (self.network.input_dim,):
            raise ValueError(f"state shape {state.shape} does not match network input_dim {self.network.input_dim}")
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q = network(tensor).squeeze(0)
        return q.numpy()

    def act(self, state: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        """Epsilon-greedy action selection. `rng` is an explicit,
        caller-supplied `numpy.random.Generator` so exploration
        randomness is reproducible under a fixed seed, independent of
        any other random state in the process."""
        if epsilon > 0.0 and rng.random() < epsilon:
            return int(rng.integers(0, self.output_dim))
        q = self.q_values(state)
        return int(np.argmax(q))

    def update_target_network(self) -> None:
        """Hard sync: target network <- current network weights."""
        self.target_network.load_state_dict(self.network.state_dict())

    def learn(self, batch: list[Transition]) -> float:
        """One standard DQN gradient step on a batch of transitions:
        `y = r + gamma * (1 - done) * max_a' Q_target(s', a')`, MSE
        against `Q(s, a)`. Returns the scalar loss value."""
        if not batch:
            raise ValueError("learn() requires a non-empty batch")

        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.int64)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32)
        dones = torch.as_tensor([float(t.done) for t in batch], dtype=torch.float32)

        q_values = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values

        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step_count += 1

        return float(loss.item())

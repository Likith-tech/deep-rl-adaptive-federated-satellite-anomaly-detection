"""Phase 9B — the DQN controller: combines `src.drl.dqn_agent.DQNAgent`
+ `src.drl.replay_buffer.ReplayBuffer` + an epsilon schedule into the
single object a training loop interacts with, kept deliberately
separate from `src.federated.drl_environment.DRLFederatedEnvironment`
(the FL round-transition logic) and from `FederatedClient`/
`FederatedServer` (the actual, unmodified aggregation mechanism)."""

from __future__ import annotations

import numpy as np

from src.drl.dqn_agent import DQNAgent
from src.drl.replay_buffer import ReplayBuffer


class DQNController:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dimensions: tuple[int, ...],
        gamma: float,
        learning_rate: float,
        replay_buffer_size: int,
        batch_size: int,
        target_update_frequency: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        seed: int,
    ) -> None:
        self.agent = DQNAgent(
            input_dim=input_dim, hidden_dimensions=hidden_dimensions, output_dim=output_dim,
            seed=seed, gamma=gamma, learning_rate=learning_rate,
        )
        self.buffer = ReplayBuffer(capacity=replay_buffer_size)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.rng = np.random.default_rng(seed)
        self.total_steps = 0

    def epsilon(self) -> float:
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def choose_action(self, state: np.ndarray) -> int:
        return self.agent.act(state.astype(np.float32), epsilon=self.epsilon(), rng=self.rng)

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state.astype(np.float32), action, reward, next_state.astype(np.float32), done)
        self.total_steps += 1
        if self.total_steps % self.target_update_frequency == 0:
            self.agent.update_target_network()

    def maybe_learn(self, replay_warmup: int) -> float | None:
        """Runs one `learn()` step if the buffer holds at least
        `replay_warmup` transitions; returns the loss, or `None` if the
        buffer is still below the warmup threshold."""
        if len(self.buffer) < max(replay_warmup, self.batch_size):
            return None
        batch = self.buffer.sample(self.batch_size, self.rng)
        return self.agent.learn(batch)

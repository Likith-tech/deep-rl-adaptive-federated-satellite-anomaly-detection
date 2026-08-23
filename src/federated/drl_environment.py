"""Phase 9B — the RL environment abstraction: state -> action ->
(local training + aggregation, delegated) -> evaluation -> reward ->
next state (see
docs/project-progress/11-phase-9b-drl-adaptive-federated-learning.md
sections 5 and 9).

This module intentionally does NOT itself run `FederatedClient`/
`FederatedServer` training — it delegates "what happens when this
round's chosen aggregation weights are applied" to an injected
`round_provider` callable, so:

  - unit tests can supply a tiny, deterministic, synthetic
    `round_provider` and never touch real NSL-KDD data or a client
    partition file (per this turn's explicit test requirement);
  - a future (not-yet-implemented) real training run can supply a
    `round_provider` that wraps the UNMODIFIED
    `src.federated.client.FederatedClient` /
    `src.federated.server.FederatedServer` — this environment never
    reimplements FedAvg or touches raw client data itself.

`round_provider(round_number, weights, signals) -> (next_signals, new_val_f1)`
is the only integration point with actual federated training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from src.drl.action_space import action_to_aggregation_weights
from src.drl.reward import compute_reward
from src.drl.state import build_state
from src.federated.adaptive import ClientSignals

RoundProvider = Callable[[int, dict[str, float], list[ClientSignals]], tuple[list[ClientSignals], float]]


class InitialSignalsProvider(Protocol):
    def __call__(self) -> tuple[list[ClientSignals], float]: ...


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class DRLFederatedEnvironment:
    """One episode = one simulated FL run of `num_rounds` rounds.
    Weighting-only (Phase 9B pilot design): every step's chosen action
    always yields aggregation weights for ALL clients in `signals` —
    no client is ever excluded."""

    def __init__(
        self,
        initial_signals_provider: InitialSignalsProvider,
        round_provider: RoundProvider,
        num_rounds: int,
        lambda_fair: float,
        lambda_stab: float,
    ) -> None:
        self._initial_signals_provider = initial_signals_provider
        self._round_provider = round_provider
        self.num_rounds = num_rounds
        self.lambda_fair = lambda_fair
        self.lambda_stab = lambda_stab

        self.round_number = 0
        self.current_signals: list[ClientSignals] | None = None
        self.val_f1: float | None = None
        self.val_f1_prev: float | None = None
        self.previous_action: int | None = None

    def reset(self) -> np.ndarray:
        self.round_number = 1
        self.current_signals, self.val_f1 = self._initial_signals_provider()
        self.val_f1_prev = None
        self.previous_action = None
        return build_state(
            self.current_signals, self.val_f1, self.val_f1_prev, self.previous_action, self.round_number, self.num_rounds
        )

    def step(self, action: int) -> StepResult:
        if self.current_signals is None:
            raise RuntimeError("step() called before reset()")

        weights = action_to_aggregation_weights(action, self.current_signals)
        performance_signals = {s.client_id: s.performance for s in self.current_signals}

        next_signals, new_val_f1 = self._round_provider(self.round_number, weights, self.current_signals)

        reward = compute_reward(
            val_f1=new_val_f1,
            val_f1_prev=self.val_f1,
            performance_signals=performance_signals,
            action=action,
            previous_action=self.previous_action,
            lambda_fair=self.lambda_fair,
            lambda_stab=self.lambda_stab,
        )

        self.val_f1_prev = self.val_f1
        self.val_f1 = new_val_f1
        self.previous_action = action
        self.current_signals = next_signals
        self.round_number += 1

        done = self.round_number > self.num_rounds
        observation = build_state(
            self.current_signals, self.val_f1, self.val_f1_prev, self.previous_action,
            min(self.round_number, self.num_rounds), self.num_rounds,
        )

        return StepResult(observation=observation, reward=reward, done=done, info={"action": action, "weights": weights, "val_f1": new_val_f1})

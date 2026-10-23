"""Phase 9B pilot — proves a DQN-chosen action's aggregation weights,
fed through the REAL, UNMODIFIED Phase 6/8 aggregation mechanism
(`FederatedServer.aggregate_with_weights`, `src/federated/server.py`),
produce a mathematically correct weighted average of client model
parameters. This is the missing link between `action_to_aggregation_weights`
(tested in isolation in tests/drl/test_action_space.py) and the actual
FedAvg-family code — no second, parallel aggregation implementation is
created anywhere in Phase 9B."""

from __future__ import annotations

import torch

from src.drl.action_space import action_to_aggregation_weights
from src.federated.adaptive import ClientSignals
from src.federated.protocol import ClientUpdate
from src.federated.server import FederatedServer

CLIENT_IDS = [f"SAT-{i + 1:02d}" for i in range(8)]


def _synthetic_signals() -> list[ClientSignals]:
    return [
        ClientSignals(client_id=cid, performance=0.5 + 0.05 * i, data=1000.0 * (i + 1), resource=0.9 - 0.07 * i, fairness=1.0 + 0.1 * i)
        for i, cid in enumerate(CLIENT_IDS)
    ]


def _synthetic_updates() -> list[ClientUpdate]:
    # Each client's "model" is a single scalar tensor equal to its index
    # + 1 — makes the correct weighted average trivially hand-computable.
    return [
        ClientUpdate(client_id=cid, state_dict={"w": torch.tensor([float(i + 1)])}, num_samples=100, local_epochs=1, train_loss=0.1, training_seconds=0.01)
        for i, cid in enumerate(CLIENT_IDS)
    ]


def test_dqn_action_weights_feed_correctly_into_real_fedavg_aggregation():
    signals = _synthetic_signals()
    updates = _synthetic_updates()

    for action in range(9):
        weights = action_to_aggregation_weights(action, signals)

        server = FederatedServer({"w": torch.zeros(1)})
        aggregated = server.aggregate_with_weights(updates, weights)

        expected = sum(weights[u.client_id] * u.state_dict["w"].item() for u in updates)
        assert torch.allclose(aggregated["w"], torch.tensor([expected]), atol=1e-6)


def test_aggregation_uses_all_eight_clients_for_every_action():
    signals = _synthetic_signals()
    updates = _synthetic_updates()

    for action in range(9):
        weights = action_to_aggregation_weights(action, signals)
        server = FederatedServer({"w": torch.zeros(1)})
        # aggregate_with_weights requires every update's client_id to
        # have a weight — if any client were silently dropped by the
        # DQN action space, this call would KeyError.
        server.aggregate_with_weights(updates, weights)
        assert set(weights.keys()) == set(CLIENT_IDS)


def test_server_round_number_increments_after_drl_driven_aggregation():
    signals = _synthetic_signals()
    updates = _synthetic_updates()
    weights = action_to_aggregation_weights(3, signals)  # "combined"

    server = FederatedServer({"w": torch.zeros(1)})
    assert server.round_number == 0
    server.aggregate_with_weights(updates, weights)
    assert server.round_number == 1

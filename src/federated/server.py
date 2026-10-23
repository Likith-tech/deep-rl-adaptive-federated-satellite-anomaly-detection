"""Phase 6 — the federated server (central coordinator).

The server maintains the global model and orchestrates rounds. It is
deliberately kept ignorant of raw data: this module never imports
pandas, never reads a parquet file, and never sees anything about a
client except what arrives in a `ClientUpdate` (client_id, model
state_dict, sample count, and small training metadata) — see
src/federated/protocol.py and src/federated/fedavg.py.
"""

from __future__ import annotations

import torch

from src.federated.fedavg import federated_average, weighted_average
from src.federated.protocol import ClientUpdate


class FederatedServer:
    def __init__(self, initial_state_dict: dict[str, torch.Tensor]) -> None:
        self.global_state_dict: dict[str, torch.Tensor] = {
            k: v.clone() for k, v in initial_state_dict.items()
        }
        self.round_number = 0

    def get_global_state(self) -> dict[str, torch.Tensor]:
        """Distribute a copy of the current global model — every client
        that calls this before a round receives identical weights."""
        return {k: v.clone() for k, v in self.global_state_dict.items()}

    def aggregate(self, updates: list[ClientUpdate]) -> dict[str, torch.Tensor]:
        """Combine client updates via sample-count-weighted FedAvg and
        replace the global model with the result. Unchanged since
        Phase 6 — standard FedAvg."""
        client_states = [u.state_dict for u in updates]
        client_sample_counts = [u.num_samples for u in updates]
        self.global_state_dict = federated_average(client_states, client_sample_counts)
        self.round_number += 1
        return self.get_global_state()

    def aggregate_with_weights(self, updates: list[ClientUpdate], weights: dict[str, float]) -> dict[str, torch.Tensor]:
        """Phase 8 — combine client updates via arbitrary (e.g.
        rule-based adaptive) per-client weights instead of raw sample
        counts. `weights` must map every update's client_id to a
        non-negative weight; weights are used exactly as given (the
        caller, src/federated/adaptive.py, is responsible for ensuring
        they sum to 1 — this method does not re-derive them from
        sample counts). The server still never touches raw data."""
        client_states = [u.state_dict for u in updates]
        ordered_weights = [weights[u.client_id] for u in updates]
        self.global_state_dict = weighted_average(client_states, ordered_weights)
        self.round_number += 1
        return self.get_global_state()

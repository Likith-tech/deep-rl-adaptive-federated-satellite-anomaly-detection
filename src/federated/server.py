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

from src.federated.fedavg import federated_average
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
        replace the global model with the result."""
        client_states = [u.state_dict for u in updates]
        client_sample_counts = [u.num_samples for u in updates]
        self.global_state_dict = federated_average(client_states, client_sample_counts)
        self.round_number += 1
        return self.get_global_state()

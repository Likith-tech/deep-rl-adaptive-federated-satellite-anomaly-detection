"""Phase 6 — the message shapes exchanged between client and server.

This module exists to make the privacy boundary explicit and typed:
what a client is allowed to hand back to the server after local
training, and nothing more. `ClientUpdate.state_dict` and
`num_samples` are the only training-derived information the server
ever sees — never a raw record, never a DataFrame, never a file path
into `data/partitions/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ClientUpdate:
    """What a client returns to the server after one round of local
    training. Deliberately contains no raw data and no reference to
    where the client's data lives on disk."""

    client_id: str
    state_dict: dict[str, torch.Tensor]
    num_samples: int
    local_epochs: int
    train_loss: float
    training_seconds: float
    origin_round: int = 0
    """Phase 9 (staleness-aware FL) — ADDITIVE, defaults to 0 so every
    prior phase (6-9B) is unaffected. Records which global-model round
    number this update was actually trained against. Under Phase 6/8/9B's
    synchronous protocol this always equals the current round (every
    client always trains against the latest global state, so
    `origin_round` is a no-op there). Phase 9's staleness simulation is
    the first caller that ever sets this to anything else — see
    src/federated/staleness.py."""

    def to_summary_dict(self) -> dict:
        """JSON-serializable summary — excludes `state_dict` (model
        weights are not something we log as text)."""
        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "local_epochs": self.local_epochs,
            "train_loss": self.train_loss,
            "training_seconds": self.training_seconds,
            "origin_round": self.origin_round,
        }


@dataclass
class RoundRecord:
    """Server-side record of one communication round."""

    round_number: int
    val_loss: float
    val_metrics: dict
    client_updates: list[dict] = field(default_factory=list)  # ClientUpdate.to_summary_dict() per client
    round_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "val_loss": self.val_loss,
            "val_metrics": self.val_metrics,
            "client_updates": self.client_updates,
            "round_seconds": self.round_seconds,
        }

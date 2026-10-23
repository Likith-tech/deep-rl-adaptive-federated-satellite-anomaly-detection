"""Phase 6 — the federated training loop orchestrator.

Wires together FederatedClient (local training), FederatedServer
(aggregation), and round-by-round evaluation on the GLOBAL validation
set. This is the only module that knows about BOTH "there are 8
clients with local files" AND "there is a server that aggregates" —
client.py and server.py remain independently testable and ignorant of
each other's concerns.

    for each round:
        global_state = server.get_global_state()      # same for every client
        updates = [client.train_round(global_state, ...) for client in clients]
        server.aggregate(updates)                      # -> new global model
        evaluate new global model on global validation
        record round history

KDDTest+ is never touched here — see scripts/evaluate_federated_model.py
for the final, one-time test evaluation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from src.evaluation.metrics import ClassificationMetrics
from src.federated.client import FederatedClient
from src.federated.protocol import RoundRecord
from src.federated.server import FederatedServer
from src.models.baseline_mlp import BaselineMLP
from src.training.baseline_trainer import load_features_and_labels, make_dataloader, run_validation
from src.training.local_trainer import build_initial_state_dict


@dataclass
class FederatedTrainingResult:
    round_history: list[RoundRecord] = field(default_factory=list)
    best_round: int = 0
    best_val_loss: float = float("inf")
    best_val_metrics: ClassificationMetrics | None = None
    total_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "round_history": [r.to_dict() for r in self.round_history],
            "best_round": self.best_round,
            "best_val_loss": self.best_val_loss,
            "best_val_metrics": self.best_val_metrics.to_dict() if self.best_val_metrics else None,
            "total_seconds": self.total_seconds,
        }


def run_federated_training(
    client_ids: list[str],
    partitions_dir: Path,
    global_validation_parquet: Path,
    feature_columns: list[str],
    model_config: dict,
    training_config: dict,
    num_rounds: int,
    shared_init_seed: int,
    checkpoint_dir: Path,
    client_resource_metadata: dict[str, dict] | None = None,
) -> FederatedTrainingResult:
    """`client_resource_metadata` (optional): simulated per-client
    bandwidth/latency/compute/availability/connectivity (Phase 4). Only
    RECORDED in round logs for later adaptive-FL phases to use — never
    used here to select, exclude, or weight clients."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clients = [
        FederatedClient(
            client_id=client_id,
            train_parquet=partitions_dir / client_id / "train.parquet",
            feature_columns=feature_columns,
        )
        for client_id in client_ids
    ]

    initial_state_dict = build_initial_state_dict(model_config, shared_init_seed)
    server = FederatedServer(initial_state_dict)

    X_val, y_val = load_features_and_labels(global_validation_parquet, feature_columns)
    val_loader = make_dataloader(X_val, y_val, training_config["batch_size"], shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    round_checkpoints_dir = checkpoint_dir / "round_checkpoints"
    round_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    result = FederatedTrainingResult()
    overall_start = time.time()

    for round_number in range(1, num_rounds + 1):
        round_start = time.time()
        global_state = server.get_global_state()  # identical copy handed to every client this round

        updates = [
            client.train_round(global_state, model_config, training_config, seed=training_config["seed"])
            for client in clients
        ]

        new_global_state = server.aggregate(updates)

        eval_model = BaselineMLP(
            input_dim=model_config["input_dim"],
            hidden_dimensions=model_config["hidden_dimensions"],
            dropout=model_config["dropout"],
            output_dim=model_config["output_dim"],
        ).to(device)
        eval_model.load_state_dict(new_global_state)
        val_loss, val_metrics = run_validation(eval_model, val_loader, criterion, device)

        client_update_summaries = []
        for u in updates:
            summary = u.to_summary_dict()
            if client_resource_metadata and u.client_id in client_resource_metadata:
                # Recorded for later adaptive-FL phases only — NOT used
                # here to select, exclude, or weight this client.
                summary["simulated_resource_metadata"] = client_resource_metadata[u.client_id]
            client_update_summaries.append(summary)

        round_record = RoundRecord(
            round_number=round_number,
            val_loss=val_loss,
            val_metrics=val_metrics.to_dict(),
            client_updates=client_update_summaries,
            round_seconds=time.time() - round_start,
        )
        result.round_history.append(round_record)

        torch.save(
            {"round_number": round_number, "model_state_dict": new_global_state, "val_loss": val_loss,
             "val_metrics": val_metrics.to_dict()},
            round_checkpoints_dir / f"round_{round_number:02d}.pt",
        )

        if val_loss < result.best_val_loss:
            result.best_val_loss = val_loss
            result.best_round = round_number
            result.best_val_metrics = val_metrics
            torch.save(
                {
                    "round_number": round_number,
                    "model_state_dict": new_global_state,
                    "model_config": model_config,
                    "training_config": training_config,
                    "feature_columns": feature_columns,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics.to_dict(),
                },
                checkpoint_dir / "best_global_model.pt",
            )

    result.total_seconds = time.time() - overall_start
    (checkpoint_dir / "round_history.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result

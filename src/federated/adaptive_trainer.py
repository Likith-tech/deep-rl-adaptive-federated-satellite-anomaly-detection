"""Phase 8 — the rule-based adaptive federated training loop.

Structurally identical to Phase 6's `src/federated/trainer.py`, reusing
`FederatedClient` and `FederatedServer` UNCHANGED. The only addition:
before aggregating, each client's just-trained local update is
evaluated on the global validation set (to get this round's
"performance" signal) and combined with static "data"/"resource"/
"fairness" signals into a deterministic score
(`src/federated/adaptive.py`), which becomes the FedAvg aggregation
weight via `FederatedServer.aggregate_with_weights` instead of
`FederatedServer.aggregate`'s raw sample-count weighting.

    for each round:
        global_state = server.get_global_state()          # same for every client
        updates = [client.train_round(global_state, ...) for client in clients]  # UNCHANGED
        for each update: evaluate its state_dict on global validation -> performance signal
        scores = compute_client_scores(signals, rule_weights)          # deterministic, no RL
        weights = scores_to_aggregation_weights(scores)
        server.aggregate_with_weights(updates, weights)                 # -> new global model
        evaluate new global model on global validation
        record round history (val metrics, scores, weights, signals)

KDDTest+ is never touched here — see scripts/evaluate_adaptive_fl_results.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from src.evaluation.metrics import ClassificationMetrics
from src.federated.adaptive import (
    ClientSignals,
    build_resource_signal,
    compute_client_scores,
    scores_to_aggregation_weights,
)
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.models.baseline_mlp import BaselineMLP
from src.training.baseline_trainer import load_features_and_labels, make_dataloader, run_validation
from src.training.local_trainer import build_initial_state_dict


@dataclass
class AdaptiveRoundRecord:
    round_number: int
    val_loss: float
    val_metrics: dict
    client_updates: list[dict] = field(default_factory=list)
    client_scores: dict[str, float] = field(default_factory=dict)
    aggregation_weights: dict[str, float] = field(default_factory=dict)
    round_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "val_loss": self.val_loss,
            "val_metrics": self.val_metrics,
            "client_updates": self.client_updates,
            "client_scores": self.client_scores,
            "aggregation_weights": self.aggregation_weights,
            "round_seconds": self.round_seconds,
        }


@dataclass
class AdaptiveFederatedTrainingResult:
    round_history: list[AdaptiveRoundRecord] = field(default_factory=list)
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


def _build_eval_model(model_config: dict, state_dict: dict, device: torch.device) -> nn.Module:
    model = BaselineMLP(
        input_dim=model_config["input_dim"],
        hidden_dimensions=model_config["hidden_dimensions"],
        dropout=model_config["dropout"],
        output_dim=model_config["output_dim"],
    ).to(device)
    model.load_state_dict(state_dict)
    return model


def run_adaptive_federated_training(
    client_ids: list[str],
    partitions_dir: Path,
    global_validation_parquet: Path,
    feature_columns: list[str],
    model_config: dict,
    training_config: dict,
    num_rounds: int,
    shared_init_seed: int,
    checkpoint_dir: Path,
    rule_weights: dict[str, float],
    client_resource_metadata: dict[str, dict],
    client_fairness_entropy: dict[str, float],
) -> AdaptiveFederatedTrainingResult:
    """`rule_weights`: {"performance","data","resource","fairness"} ->
    weight, must sum to 1 (see configs/adaptive_fl.yaml). `client_resource_metadata`
    and `client_fairness_entropy` are STATIC per-client signals (Phase 4
    simulated conditions and Phase-4/7-style category-distribution
    entropy respectively) — they don't change round to round, unlike
    the performance signal, which is recomputed every round from that
    round's local update."""
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

    result = AdaptiveFederatedTrainingResult()
    overall_start = time.time()

    for round_number in range(1, num_rounds + 1):
        round_start = time.time()
        global_state = server.get_global_state()

        updates = [
            client.train_round(global_state, model_config, training_config, seed=training_config["seed"])
            for client in clients
        ]

        # This round's "performance" signal: each client's JUST-TRAINED
        # local update, evaluated on the shared global validation set
        # (never on KDDTest+). The client/server code that produced
        # `updates` is completely unmodified from Phase 6.
        signals = []
        for u in updates:
            eval_model = _build_eval_model(model_config, u.state_dict, device)
            _, local_val_metrics = run_validation(eval_model, val_loader, criterion, device)
            signals.append(
                ClientSignals(
                    client_id=u.client_id,
                    performance=local_val_metrics.f1,
                    data=float(u.num_samples),
                    resource=build_resource_signal(client_resource_metadata[u.client_id]),
                    fairness=client_fairness_entropy[u.client_id],
                )
            )

        scores = compute_client_scores(signals, rule_weights)
        weights = scores_to_aggregation_weights(scores)

        new_global_state = server.aggregate_with_weights(updates, weights)

        eval_model = _build_eval_model(model_config, new_global_state, device)
        val_loss, val_metrics = run_validation(eval_model, val_loader, criterion, device)

        round_record = AdaptiveRoundRecord(
            round_number=round_number,
            val_loss=val_loss,
            val_metrics=val_metrics.to_dict(),
            client_updates=[u.to_summary_dict() for u in updates],
            client_scores=scores,
            aggregation_weights=weights,
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

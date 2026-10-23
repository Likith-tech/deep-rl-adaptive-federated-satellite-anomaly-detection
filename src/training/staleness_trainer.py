"""Phase 9 — the staleness- and resource-aware federated training loop.

Structurally identical to Phase 6's `src/federated/trainer.py` and
Phase 8's `src/federated/adaptive_trainer.py`, reusing `FederatedClient`
and `FederatedServer` UNCHANGED. The only addition: each round, each
client's participation is decided by a seeded coin flip weighted by its
Phase 4 `availability_probability` (`src/federated/staleness.py`) — a
client that doesn't sync trains against whatever global state it last
synced to, instead of the current round's latest state. The resulting
per-client staleness is converted into an aggregation-weight discount
and combined with its sample count, then applied through the
UNMODIFIED `FederatedServer.aggregate_with_weights`.

    for each round:
        latest_global_state = server.get_global_state()
        record latest_global_state in history (indexed by round-1)
        for each client:
            decide participation (sync or reuse cached state) -> origin_round, staleness
            train_state = history[origin_round]
            update = client.train_round(train_state, ...)          # UNCHANGED
            update.origin_round = origin_round
        weights = staleness-discounted sample-count weights
        server.aggregate_with_weights(updates, weights)             # -> new global model
        evaluate new global model on global validation
        record round history (val metrics, staleness, weights)

KDDTest+ is never touched here — see scripts/evaluate_staleness_fl_results.py.
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
from src.federated.server import FederatedServer
from src.federated.staleness import combine_with_sample_weights, decide_participation, staleness_weight
from src.models.baseline_mlp import BaselineMLP
from src.training.baseline_trainer import load_features_and_labels, make_dataloader, run_validation
from src.training.local_trainer import build_initial_state_dict


@dataclass
class StalenessRoundRecord:
    round_number: int
    val_loss: float
    val_metrics: dict
    client_updates: list[dict] = field(default_factory=list)
    client_staleness: dict[str, int] = field(default_factory=dict)
    client_synced: dict[str, bool] = field(default_factory=dict)
    aggregation_weights: dict[str, float] = field(default_factory=dict)
    round_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "val_loss": self.val_loss,
            "val_metrics": self.val_metrics,
            "client_updates": self.client_updates,
            "client_staleness": self.client_staleness,
            "client_synced": self.client_synced,
            "aggregation_weights": self.aggregation_weights,
            "round_seconds": self.round_seconds,
        }


@dataclass
class StalenessFederatedTrainingResult:
    round_history: list[StalenessRoundRecord] = field(default_factory=list)
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


def run_staleness_federated_training(
    client_ids: list[str],
    partitions_dir: Path,
    global_validation_parquet: Path,
    feature_columns: list[str],
    model_config: dict,
    training_config: dict,
    num_rounds: int,
    shared_init_seed: int,
    checkpoint_dir: Path,
    client_availability: dict[str, float],
    staleness_mode: str,
    staleness_decay_rate: float,
    participation_seed: int,
) -> StalenessFederatedTrainingResult:
    """`client_availability`: {client_id -> Phase 4
    availability_probability}, the same simulated field Phase 8's
    `resource` signal already reads. `staleness_mode` /
    `staleness_decay_rate`: see `src/federated/staleness.py:staleness_weight`.
    `participation_seed` seeds the per-round sync coin flips
    deterministically, independent of `training_config["seed"]` (which
    seeds model training) — kept separate so a staleness-pattern
    ablation and a training-seed ablation can vary independently."""
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

    # history[0] = shared initial state; history[k] = the global state
    # produced by aggregating round k. Kept in memory (10 rounds x a
    # ~24K-parameter MLP is trivially small) so a client that hasn't
    # synced in several rounds can still train against its real,
    # correct cached state rather than an approximation.
    history: dict[int, dict[str, torch.Tensor]] = {0: {k: v.clone() for k, v in initial_state_dict.items()}}
    previous_origin_round = {cid: 0 for cid in client_ids}

    result = StalenessFederatedTrainingResult()
    overall_start = time.time()

    for round_number in range(1, num_rounds + 1):
        round_start = time.time()
        latest_index = round_number - 1
        history[latest_index] = server.get_global_state()

        updates = []
        client_staleness: dict[str, int] = {}
        client_synced: dict[str, bool] = {}
        for client in clients:
            decision = decide_participation(
                client_id=client.client_id,
                round_number=round_number,
                availability_probability=client_availability[client.client_id],
                previous_origin_round=previous_origin_round[client.client_id],
                seed=participation_seed,
            )
            previous_origin_round[client.client_id] = decision.origin_round

            train_state = history[decision.origin_round]
            update = client.train_round(train_state, model_config, training_config, seed=training_config["seed"])
            update.origin_round = decision.origin_round
            updates.append(update)
            client_staleness[client.client_id] = decision.staleness
            client_synced[client.client_id] = decision.synced_this_round

        sample_counts = {u.client_id: u.num_samples for u in updates}
        weights_stale = {
            cid: staleness_weight(client_staleness[cid], staleness_mode, staleness_decay_rate)
            for cid in sample_counts
        }
        weights = combine_with_sample_weights(sample_counts, weights_stale)

        new_global_state = server.aggregate_with_weights(updates, weights)

        eval_model = _build_eval_model(model_config, new_global_state, device)
        val_loss, val_metrics = run_validation(eval_model, val_loader, criterion, device)

        round_record = StalenessRoundRecord(
            round_number=round_number,
            val_loss=val_loss,
            val_metrics=val_metrics.to_dict(),
            client_updates=[u.to_summary_dict() for u in updates],
            client_staleness=client_staleness,
            client_synced=client_synced,
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

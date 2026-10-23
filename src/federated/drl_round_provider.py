"""Phase 9B — the REAL NSL-KDD round provider: an adapter around the
EXISTING, UNMODIFIED Phase 6/8 federated components
(`FederatedClient`, `FederatedServer`, `build_initial_state_dict`,
`run_validation`) that satisfies `src.federated.drl_environment`'s
`RoundProvider` contract:

    real data -> FederatedClient (Phase 6, unchanged)
              -> FederatedServer (Phase 6/8, unchanged)
              -> real federated round
              -> DRLFederatedEnvironment

No aggregation, training, or client logic is reimplemented here — this
module only wires the DQN environment's `(round_number, weights,
signals) -> (next_signals, new_val_f1)` interface to real calls against
`FederatedClient.train_round` / `FederatedServer.aggregate_with_weights`
/ `FederatedServer.get_global_state`, exactly as
`src.federated.adaptive_trainer.run_adaptive_federated_training`
already does for Phase 8's round loop — this class follows the same
per-round shape, just with the DQN choosing the weights instead of a
fixed rule.

PRIVACY BOUNDARY: this module never imports pandas directly and never
opens a client's `train.parquet` itself — it only ever calls the
already-privacy-tested `FederatedClient.train_round` (which reads its
OWN partition) and `FederatedServer` methods (which the existing
AST-based test already proves never touch pandas/parquet). The static
resource/fairness signals are read from the already-committed,
non-raw-data JSON artifacts `data/partitions/manifest.json` and
`data/partitions/satellite_metadata.json` — the same files
`scripts/run_adaptive_fl_experiments.py` (Phase 8) already reads for
the identical purpose.

KDDTest+ is never referenced anywhere in this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.federated.adaptive import ClientSignals, build_resource_signal
from src.federated.adaptive_trainer import _build_eval_model
from src.federated.client import FederatedClient
from src.federated.protocol import ClientUpdate
from src.federated.server import FederatedServer
from src.simulation.non_iid_metrics import client_category_distribution, shannon_entropy
from src.simulation.partitioner import ClientStats
from src.training.baseline_trainer import load_features_and_labels, make_dataloader, run_validation
from src.training.local_trainer import build_initial_state_dict


def build_fairness_entropy(manifest: dict) -> dict[str, float]:
    """IDENTICAL logic to
    `scripts/run_adaptive_fl_experiments.py:build_fairness_entropy`
    (Phase 8) — relocated here (not reimplemented differently) so it is
    importable by both the script and this real provider without
    importing a `scripts/` module as a library. Computed from the
    ALREADY-COMMITTED `data/partitions/manifest.json` — never from raw
    per-record data."""
    categories = sorted({cat for s in manifest["client_stats"].values() for cat in s["category_counts"]})
    entropy = {}
    for client_id, stats_dict in manifest["client_stats"].items():
        stats = ClientStats(
            client_id=client_id,
            total_samples=stats_dict["total_samples"],
            normal_samples=stats_dict["normal_samples"],
            anomaly_samples=stats_dict["anomaly_samples"],
            category_counts=stats_dict["category_counts"],
        )
        distribution = client_category_distribution(stats, categories)
        entropy[client_id] = shannon_entropy(distribution)
    return entropy


class RealNSLKDDRoundProvider:
    """Stateful adapter: one instance == one real FL episode's worth of
    round-by-round training/aggregation/evaluation, driven externally
    by `DRLFederatedEnvironment`. Holds no dataset in memory beyond
    what `FederatedClient`/the validation `DataLoader` already load —
    nothing here is a second copy of the FL logic."""

    def __init__(
        self,
        client_ids: list[str],
        partitions_dir: Path,
        global_validation_parquet: Path,
        feature_columns: list[str],
        model_config: dict,
        training_config: dict,
        shared_init_seed: int,
        client_resource_metadata: dict[str, dict],
        client_fairness_entropy: dict[str, float],
        num_rounds: int,
    ) -> None:
        self.client_ids = client_ids
        self.model_config = model_config
        self.training_config = training_config
        self.client_resource_metadata = client_resource_metadata
        self.client_fairness_entropy = client_fairness_entropy
        self.num_rounds = num_rounds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clients = [
            FederatedClient(client_id=cid, train_parquet=partitions_dir / cid / "train.parquet", feature_columns=feature_columns)
            for cid in client_ids
        ]

        initial_state_dict = build_initial_state_dict(model_config, shared_init_seed)
        self.server = FederatedServer(initial_state_dict)

        X_val, y_val = load_features_and_labels(global_validation_parquet, feature_columns)
        self.val_loader = make_dataloader(X_val, y_val, training_config["batch_size"], shuffle=False)
        self.criterion = nn.BCEWithLogitsLoss()

        self._pending_updates: dict[int, list[ClientUpdate]] = {}

        # Round-level (aggregated-model) full validation metrics, keyed
        # by round number (0 = the initial, pre-training global model).
        # Populated ONLY for the aggregated model each round (not the
        # per-client update evals inside `_signals_from_updates`), so a
        # caller can log accuracy/precision/recall/ROC-AUC/loss per
        # round without those values being overwritten by the next
        # round's per-client evaluations.
        self.round_eval_history: dict[int, dict] = {}
        self._last_aggregated_state: dict | None = None

    def _evaluate_state_dict(self, state_dict: dict) -> float:
        """Per-client-update evaluation (used only for the `performance`
        signal) — returns F1 only, matching Phase 8's own signal shape."""
        eval_model = _build_eval_model(self.model_config, state_dict, self.device)
        _, val_metrics = run_validation(eval_model, self.val_loader, self.criterion, self.device)
        return val_metrics.f1

    def _evaluate_and_record_round(self, round_number: int, state_dict: dict) -> float:
        """Round-level evaluation of the AGGREGATED global model — records
        full metrics (loss + accuracy/precision/recall/f1/roc_auc/fpr)
        into `self.round_eval_history[round_number]` and returns F1 (the
        only value the DQN state/reward are allowed to use)."""
        eval_model = _build_eval_model(self.model_config, state_dict, self.device)
        val_loss, val_metrics = run_validation(eval_model, self.val_loader, self.criterion, self.device)
        self.round_eval_history[round_number] = {"val_loss": val_loss, "val_metrics": val_metrics.to_dict()}
        return val_metrics.f1

    def _train_round(self, round_number: int) -> list[ClientUpdate]:
        global_state = self.server.get_global_state()  # identical copy to every client, per Phase 6's own contract
        updates = [
            client.train_round(global_state, self.model_config, self.training_config, seed=self.training_config["seed"])
            for client in self.clients
        ]
        self._pending_updates[round_number] = updates
        return updates

    def _signals_from_updates(self, updates: list[ClientUpdate]) -> list[ClientSignals]:
        signals = []
        for u in updates:
            performance = self._evaluate_state_dict(u.state_dict)  # global-validation-only, per approved reward/state design
            signals.append(
                ClientSignals(
                    client_id=u.client_id,
                    performance=performance,
                    data=float(u.num_samples),
                    resource=build_resource_signal(self.client_resource_metadata[u.client_id]),
                    fairness=self.client_fairness_entropy[u.client_id],
                )
            )
        return signals

    def initial_signals_provider(self) -> tuple[list[ClientSignals], float]:
        """`DRLFederatedEnvironment.reset()`'s injection point: trains
        round 1's clients from the shared initial global model,
        evaluates that (untrained) initial model once as the baseline
        val_F1, and returns round 1's signals."""
        initial_val_f1 = self._evaluate_and_record_round(0, self.server.get_global_state())
        updates = self._train_round(1)
        return self._signals_from_updates(updates), initial_val_f1

    def round_provider(self, round_number: int, weights: dict[str, float], signals: list[ClientSignals]) -> tuple[list[ClientSignals], float]:
        """`DRLFederatedEnvironment.step()`'s injection point: aggregates
        THIS round's already-trained updates (cached from
        `initial_signals_provider`/the previous call) via the
        UNCHANGED `FederatedServer.aggregate_with_weights`, evaluates
        the new global model, and — if the episode continues — trains
        the NEXT round's clients so their signals are ready for the
        following state."""
        updates = self._pending_updates.pop(round_number)
        new_global_state = self.server.aggregate_with_weights(updates, weights)
        new_val_f1 = self._evaluate_and_record_round(round_number, new_global_state)
        self._last_aggregated_state = new_global_state  # exposed for a caller doing final model selection

        if round_number < self.num_rounds:
            next_updates = self._train_round(round_number + 1)
            next_signals = self._signals_from_updates(next_updates)
        else:
            # Episode terminates this round; DRLFederatedEnvironment
            # still builds a final observation but the environment
            # marks it `done=True` and this value is never acted on —
            # reusing the current signals is documented, not a silent
            # invention of new information.
            next_signals = signals

        return next_signals, new_val_f1


def build_real_round_provider(
    repo_root: Path,
    model_config: dict,
    training_config: dict,
    shared_init_seed: int,
    num_rounds: int,
) -> RealNSLKDDRoundProvider:
    """Factory: discovers the real Phase 4/6 partition
    (`data/partitions/`) and builds a `RealNSLKDDRoundProvider` from
    it, exactly the same source files
    `scripts/run_adaptive_fl_experiments.py` (Phase 8) reads:
    `manifest.json`, `satellite_metadata.json`,
    `results/models/preprocessing/feature_columns.json`,
    `data/processed/validation.parquet`. Raises `FileNotFoundError`
    with a clear message (never silently falls back to synthetic data)
    if any required artifact is missing — mirroring Phase 8's own
    explicit "run Phase 4 first" check."""
    partitions_dir = repo_root / "data" / "partitions"
    manifest_path = partitions_dir / "manifest.json"
    satellite_metadata_path = partitions_dir / "satellite_metadata.json"
    feature_columns_path = repo_root / "results" / "models" / "preprocessing" / "feature_columns.json"
    global_validation_parquet = repo_root / "data" / "processed" / "validation.parquet"

    for required_path in (manifest_path, satellite_metadata_path, feature_columns_path, global_validation_parquet):
        if not required_path.exists():
            raise FileNotFoundError(f"required Phase 4/6 artifact not found: {required_path}")

    manifest = json.loads(manifest_path.read_text())
    client_ids = sorted(manifest["client_sample_counts"].keys())

    satellite_metadata = json.loads(satellite_metadata_path.read_text())
    client_resource_metadata = {
        cid: {k: v for k, v in meta.items() if k not in ("client_id", "data_size")}
        for cid, meta in satellite_metadata["clients"].items()
    }

    client_fairness_entropy = build_fairness_entropy(manifest)
    feature_columns = json.loads(feature_columns_path.read_text())

    return RealNSLKDDRoundProvider(
        client_ids=client_ids,
        partitions_dir=partitions_dir,
        global_validation_parquet=global_validation_parquet,
        feature_columns=feature_columns,
        model_config=model_config,
        training_config=training_config,
        shared_init_seed=shared_init_seed,
        client_resource_metadata=client_resource_metadata,
        client_fairness_entropy=client_fairness_entropy,
        num_rounds=num_rounds,
    )

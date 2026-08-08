"""Phase 6 — a federated client.

Each `FederatedClient` represents one simulated satellite. It knows
only its OWN local partition path (`data/partitions/<client_id>/train.parquet`,
from Phase 4) and, when asked, trains locally starting from whatever
global model state it's handed — it never sees, and has no way to
load, another client's data.

    global model state -> load local data -> train E local epochs
        -> ClientUpdate(state_dict, num_samples, ...)

The returned `ClientUpdate` (src/federated/protocol.py) is the only
thing that ever leaves the client — never a raw record.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn

from src.federated.protocol import ClientUpdate
from src.models.baseline_mlp import BaselineMLP
from src.training.baseline_trainer import load_features_and_labels, make_dataloader, set_seed


class FederatedClient:
    def __init__(self, client_id: str, train_parquet: Path, feature_columns: list[str]) -> None:
        self.client_id = client_id
        self.train_parquet = train_parquet
        self.feature_columns = feature_columns

    def train_round(
        self,
        global_state_dict: dict[str, torch.Tensor],
        model_config: dict,
        training_config: dict,
        seed: int,
    ) -> ClientUpdate:
        """Train locally for `training_config['local_epochs']` epoch(s),
        starting from `global_state_dict` (this round's global model —
        identical for every client this round). Returns only the
        updated parameters and metadata, never the local data itself.
        """
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train, y_train = load_features_and_labels(self.train_parquet, self.feature_columns)
        train_loader = make_dataloader(X_train, y_train, training_config["batch_size"], shuffle=True)

        model = BaselineMLP(
            input_dim=model_config["input_dim"],
            hidden_dimensions=model_config["hidden_dimensions"],
            dropout=model_config["dropout"],
            output_dim=model_config["output_dim"],
        ).to(device)
        model.load_state_dict({k: v.clone() for k, v in global_state_dict.items()})

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])

        start_time = time.time()
        local_epochs = training_config["local_epochs"]
        total_loss = 0.0
        n_seen = 0

        for _ in range(local_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
                n_seen += len(y_batch)

        training_seconds = time.time() - start_time
        avg_loss = total_loss / n_seen

        return ClientUpdate(
            client_id=self.client_id,
            state_dict={k: v.detach().clone().cpu() for k, v in model.state_dict().items()},
            num_samples=len(y_train),
            local_epochs=local_epochs,
            train_loss=avg_loss,
            training_seconds=training_seconds,
        )

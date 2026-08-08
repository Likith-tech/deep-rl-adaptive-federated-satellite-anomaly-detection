"""Phase 6 — final, one-time KDDTest+ evaluation of the selected FedAvg
global model, plus round-by-round plots and the results report.

Run from the repository root, after scripts/run_federated_training.py
has produced results/models/federated/best_global_model.pt:

    python scripts/evaluate_federated_model.py

Reads:
    results/models/federated/best_global_model.pt
    results/models/federated/round_history.json
    experiments/federated/run_config.json
    data/processed/test.parquet   (KDDTest+ — never used during training/round selection)

Writes:
    results/plots/federated/{val_f1_per_round,val_loss_per_round,val_accuracy_per_round,
                              confusion_matrix,client_training_time_per_round}.png
    results/reports/federated_results.md

This script is meant to be run ONCE per federated experiment version to
report the final test metrics — do not use it in a round-selection loop
against the test set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import pandas/sklearn before torch/matplotlib — see the note in
# scripts/evaluate_local_models.py (Phase 5) about avoiding an
# intermittent access-violation crash on this Windows/Python 3.14 env
# when torch is imported before pyarrow's parquet reader initializes.
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.training.baseline_trainer import load_features_and_labels  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.baseline_mlp import BaselineMLP  # noqa: E402

CHECKPOINT_PATH = REPO_ROOT / "results" / "models" / "federated" / "best_global_model.pt"
ROUND_HISTORY_PATH = REPO_ROOT / "results" / "models" / "federated" / "round_history.json"
RUN_CONFIG_PATH = REPO_ROOT / "experiments" / "federated" / "run_config.json"
TEST_PARQUET = REPO_ROOT / "data" / "processed" / "test.parquet"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "federated"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "federated_results.md"

# Fixed reference values from earlier phases — not re-derived here.
CENTRALIZED_BASELINE_TESTSET = {
    "accuracy": 0.7832, "precision": 0.9278, "recall": 0.6713,
    "f1": 0.7790, "roc_auc": 0.8972, "false_positive_rate": 0.0690,
}
LOCAL_ONLY_VALIDATION_F1 = {"mean": 0.9721, "min": 0.9242, "max": 0.9918}


def plot_metric_per_round(rounds: list[int], values: list[float], ylabel: str, title: str, out_path: Path, best_round: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rounds, values, marker="o", color="#3b82f6")
    ax.axvline(best_round, color="gray", linestyle="--", alpha=0.6, label=f"Best round ({best_round})")
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(rounds)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_client_training_time(round_history: list[dict], out_path: Path) -> None:
    rounds = [r["round_number"] for r in round_history]
    client_ids = [u["client_id"] for u in round_history[0]["client_updates"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for client_id in client_ids:
        times = []
        for r in round_history:
            match = next(u for u in r["client_updates"] if u["client_id"] == client_id)
            times.append(match["training_seconds"])
        ax.plot(rounds, times, marker="o", label=client_id, alpha=0.8)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Local training time (seconds)")
    ax.set_title("Per-Client Local Training Time by Round")
    ax.set_xticks(rounds)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], out_path: Path) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_arr, cmap="Blues")
    labels = ["Normal (0)", "Anomaly (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — FedAvg Global Model on KDDTest+", fontsize=12, pad=15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                     color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: {CHECKPOINT_PATH} not found. Run scripts/run_federated_training.py first.", file=sys.stderr)
        sys.exit(1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model_config = checkpoint["model_config"]
    training_config = checkpoint["training_config"]
    feature_columns = checkpoint["feature_columns"]
    best_round = checkpoint["round_number"]

    round_history = json.loads(ROUND_HISTORY_PATH.read_text())["round_history"]
    run_config = json.loads(RUN_CONFIG_PATH.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineMLP(
        input_dim=model_config["input_dim"],
        hidden_dimensions=model_config["hidden_dimensions"],
        dropout=model_config["dropout"],
        output_dim=model_config["output_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_test, y_test = load_features_and_labels(TEST_PARQUET, feature_columns)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device)).squeeze(-1)
        y_prob = torch.sigmoid(logits).cpu().numpy()
    y_true = y_test.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)
    test_metrics = compute_metrics(y_true, y_pred, y_prob)

    print("\n=== Final FedAvg global model — KDDTest+ metrics ===")
    for key, value in test_metrics.to_dict().items():
        print(f"{key}: {value}")

    # --- Plots ---
    rounds = [r["round_number"] for r in round_history]
    plot_metric_per_round(rounds, [r["val_metrics"]["f1"] for r in round_history],
                           "Validation F1", "FedAvg — Global Validation F1 per Round",
                           PLOTS_DIR / "val_f1_per_round.png", best_round)
    plot_metric_per_round(rounds, [r["val_loss"] for r in round_history],
                           "Validation loss (BCE)", "FedAvg — Global Validation Loss per Round",
                           PLOTS_DIR / "val_loss_per_round.png", best_round)
    plot_metric_per_round(rounds, [r["val_metrics"]["accuracy"] for r in round_history],
                           "Validation accuracy", "FedAvg — Global Validation Accuracy per Round",
                           PLOTS_DIR / "val_accuracy_per_round.png", best_round)
    plot_client_training_time(round_history, PLOTS_DIR / "client_training_time_per_round.png")
    plot_confusion_matrix(test_metrics.confusion_matrix, PLOTS_DIR / "confusion_matrix.png")

    # --- Results report (real numbers only) ---
    federation_config = run_config["config"]["federation"]
    comm = run_config["communication_estimate"]

    lines = [
        "# Phase 6 — Federated Learning Baseline Results (FedAvg)",
        "",
        "## 1. Purpose",
        "",
        "Establish the first genuine Federated Learning result: can "
        "standard synchronous FedAvg, training the same Phase 2 MLP "
        "architecture across the 8 non-IID satellite clients without "
        "sharing raw data, match or beat (a) the Phase 2 centralized "
        "model on KDDTest+ and (b) what any single satellite could "
        "achieve training alone (Phase 5)?",
        "",
        "## 2. FedAvg architecture",
        "",
        "Standard McMahan et al. synchronous FedAvg: `w_global = "
        "sum_k (n_k / N) * w_k`, where `w_k` is client k's local model "
        "after local training and `n_k` its local sample count. "
        "Implemented in `src/federated/fedavg.py` (pure, unit-tested "
        "aggregation), `src/federated/client.py` (local training only), "
        "`src/federated/server.py` (aggregation only — never reads a "
        "parquet file, verified by a dedicated test), and "
        "`src/federated/trainer.py` (orchestrates the round loop). No "
        "raw client data is ever sent to the server — only model "
        "parameters and sample counts (`src/federated/protocol.py`).",
        "",
        "## 3. Client setup",
        "",
        "The 8 simulated satellites from Phase 4 (`data/partitions/`), "
        "each training on ONLY its own real NSL-KDD local partition. "
        "See `docs/project-progress/05-phase-4-satellite-simulation.md`.",
        "",
        f"## 4. Number of clients",
        "",
        f"{federation_config['num_clients']} clients, ALL participating every round "
        f"(`clients_per_round: {federation_config['clients_per_round']}` — no adaptive selection in this phase).",
        "",
        "## 5. Communication rounds",
        "",
        f"{federation_config['num_rounds']} rounds.",
        "",
        "## 6. Local epochs",
        "",
        f"{training_config['local_epochs']} local epoch(s) per client per round.",
        "",
        "## 7. FedAvg weighting",
        "",
        f"Sample-count weighted (`{federation_config['weighting']}`) — NOT equal "
        "weighting. Client k's contribution weight = its local sample "
        "count / total samples across all participating clients that round.",
        "",
        "## 8. Training configuration",
        "",
        f"- Optimizer: {training_config['optimizer']} (Adam)",
        f"- Learning rate: {training_config['learning_rate']}",
        f"- Loss: {training_config['loss']} (BCEWithLogitsLoss)",
        f"- Batch size: {training_config['batch_size']}",
        f"- Seed: {training_config['seed']}",
        f"- Shared initial-weights seed: {run_config['config']['initialization']['shared_init_seed']} "
        "(Round 1's global model, and every client's starting point each "
        "round, comes from this single shared seed)",
        "- Validation strategy: global validation set "
        "(`data/processed/validation.parquet`) evaluated after every "
        "round; used ONLY for round selection. KDDTest+ evaluated once, "
        "after the round was already chosen.",
        "",
        "## 9. Round-by-round validation metrics",
        "",
        "| Round | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |",
        "|------:|---------:|----------:|-------:|---:|--------:|----:|",
    ]
    for r in round_history:
        m = r["val_metrics"]
        roc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "not available"
        lines.append(
            f"| {r['round_number']} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {roc} | {m['false_positive_rate']:.4f} |"
        )

    best_round_record = next(r for r in round_history if r["round_number"] == best_round)
    lines += [
        "",
        "See `results/plots/federated/val_f1_per_round.png`, "
        "`val_loss_per_round.png`, and `val_accuracy_per_round.png` for "
        "the visual versions.",
        "",
        "## 10. Best validation round",
        "",
        f"**Round {best_round}** (selected by lowest global validation loss, "
        f"{best_round_record['val_loss']:.4f}; validation F1 at that round: "
        f"{best_round_record['val_metrics']['f1']:.4f}). KDDTest+ was NOT "
        "consulted for this selection.",
        "",
        "## 11. Final KDDTest+ metrics",
        "",
        "Evaluated ONCE, after round selection, on the untouched KDDTest+ set:",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {test_metrics.accuracy:.4f} |",
        f"| Precision | {test_metrics.precision:.4f} |",
        f"| Recall | {test_metrics.recall:.4f} |",
        f"| F1 | {test_metrics.f1:.4f} |",
        f"| ROC-AUC | {test_metrics.roc_auc:.4f} |" if test_metrics.roc_auc is not None else "| ROC-AUC | not available |",
        f"| False Positive Rate | {test_metrics.false_positive_rate:.4f} |",
        "",
        "## 12. Confusion matrix (KDDTest+)",
        "",
        "```\n"
        "                Predicted Normal   Predicted Anomaly\n"
        f"Actual Normal   {test_metrics.confusion_matrix[0][0]:>16}   {test_metrics.confusion_matrix[0][1]:>18}\n"
        f"Actual Anomaly  {test_metrics.confusion_matrix[1][0]:>16}   {test_metrics.confusion_matrix[1][1]:>18}\n"
        "```",
        "",
        "See `results/plots/federated/confusion_matrix.png` for the visual version.",
        "",
        "## 13. Comparison with centralized baseline",
        "",
        "All three rows below are FINAL KDDTest+ results — a fair, "
        "like-for-like comparison (unlike Phase 5, which only had "
        "global-validation numbers available):",
        "",
        "| Metric | Phase 2 centralized (KDDTest+) | Phase 6 FedAvg (KDDTest+) |",
        "|---|---:|---:|",
    ]
    for key, label in (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                        ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("false_positive_rate", "FPR")):
        centralized_val = CENTRALIZED_BASELINE_TESTSET[key]
        fed_val = test_metrics.to_dict()[key]
        fed_str = f"{fed_val:.4f}" if fed_val is not None else "not available"
        lines.append(f"| {label} | {centralized_val:.4f} | {fed_str} |")

    f1_delta = test_metrics.f1 - CENTRALIZED_BASELINE_TESTSET["f1"]
    if f1_delta > 0:
        delta_sentence = f"FedAvg improved on the centralized baseline's F1 by {f1_delta:+.4f} on KDDTest+."
    elif f1_delta == 0:
        delta_sentence = "FedAvg exactly matched the centralized baseline's F1 on KDDTest+."
    else:
        delta_sentence = (
            f"FedAvg did NOT beat the centralized baseline on KDDTest+ — its F1 was "
            f"{abs(f1_delta):.4f} lower ({test_metrics.f1:.4f} vs {CENTRALIZED_BASELINE_TESTSET['f1']:.4f})."
        )
    lines += [
        "",
        delta_sentence + " (Real, measured difference — not adjusted or rounded favorably.)",
        "",
        "## 14. Comparison with local-only baseline",
        "",
        "**Important dataset caveat:** Phase 5's numbers below are "
        "GLOBAL-VALIDATION results (KDDTest+ was intentionally not used "
        "in Phase 5). Phase 6's FedAvg number in this section is ALSO "
        "shown on global validation for a like-for-like comparison — its "
        "KDDTest+ number is in section 13 instead, compared only against "
        "the other KDDTest+ result (Phase 2).",
        "",
        "| Metric (global validation) | Phase 5 local-only (mean across 8 clients) | Phase 5 local-only (range) | Phase 6 FedAvg (best round) |",
        "|---|---:|---:|---:|",
        f"| F1 | {LOCAL_ONLY_VALIDATION_F1['mean']:.4f} | "
        f"{LOCAL_ONLY_VALIDATION_F1['min']:.4f} – {LOCAL_ONLY_VALIDATION_F1['max']:.4f} | "
        f"{best_round_record['val_metrics']['f1']:.4f} |",
        "",
    ]
    fedavg_val_f1 = best_round_record["val_metrics"]["f1"]
    if fedavg_val_f1 > LOCAL_ONLY_VALIDATION_F1["max"]:
        local_comparison_sentence = (
            f"FedAvg's validation F1 ({fedavg_val_f1:.4f}) exceeds even the BEST "
            f"individual local-only client ({LOCAL_ONLY_VALIDATION_F1['max']:.4f})."
        )
    elif fedavg_val_f1 > LOCAL_ONLY_VALIDATION_F1["mean"]:
        local_comparison_sentence = (
            f"FedAvg's validation F1 ({fedavg_val_f1:.4f}) exceeds the mean local-only "
            f"client performance ({LOCAL_ONLY_VALIDATION_F1['mean']:.4f}), though not the "
            f"single best local client ({LOCAL_ONLY_VALIDATION_F1['max']:.4f})."
        )
    else:
        local_comparison_sentence = (
            f"FedAvg's validation F1 ({fedavg_val_f1:.4f}) does not exceed the mean "
            f"local-only client performance ({LOCAL_ONLY_VALIDATION_F1['mean']:.4f}) measured in Phase 5."
        )
    lines += [
        local_comparison_sentence + " This is a validation-set comparison only — "
        "see section 13 for the KDDTest+-only comparison against Phase 2.",
        "",
        "## 15. Communication-cost estimate",
        "",
        f"- Model parameters: {comm['num_model_parameters']:,}",
        f"- Bytes per model transfer (float32): {comm['bytes_per_model_transfer']:,} bytes "
        f"(~{comm['bytes_per_model_transfer'] / 1024:.1f} KB)",
        f"- Upload (clients -> server) per round: {comm['upload_bytes_per_round']:,} bytes "
        f"(~{comm['upload_bytes_per_round'] / 1024:.1f} KB)",
        f"- Download (server -> clients) per round: {comm['download_bytes_per_round']:,} bytes "
        f"(~{comm['download_bytes_per_round'] / 1024:.1f} KB)",
        f"- Total estimated communication, all {federation_config['num_rounds']} rounds: "
        f"{comm['total_communication_bytes_all_rounds']:,} bytes "
        f"(~{comm['total_communication_bytes_all_rounds'] / (1024 * 1024):.2f} MB)",
        "",
        "**This is a simulation estimate of parameter transfer size only** "
        "(raw float32 parameter count × 4 bytes) — it is NOT measured "
        "real network traffic and does NOT account for the simulated "
        "per-client bandwidth/latency from Phase 4 (those are preserved "
        "in round logs for later phases but do not affect this phase's "
        "training or this estimate).",
        "",
        "## 16. Limitations",
        "",
        "- Standard synchronous FedAvg only: no adaptive client "
        "selection, no staleness-aware aggregation, no FedProx, no "
        "personalization, no secure aggregation, no differential "
        "privacy, no resource-based client weighting.",
        "- ALL 8 clients participate every round — no partial "
        "participation or dropout modeling yet.",
        "- The satellite/client environment is simulated (Phase 4); "
        "only the underlying NSL-KDD records are real.",
        "- NSL-KDD remains a terrestrial dataset — not real satellite telemetry.",
        "- No DRL/DQN/reinforcement learning of any kind is used to "
        "guide this phase's client selection or aggregation.",
        "- Simulated resource metadata (bandwidth/latency/compute/"
        "availability/connectivity) is recorded per round but does not "
        "influence training in this phase.",
        "- Round count and local-epoch count were chosen for a clean, "
        "interpretable baseline, not tuned via a large sweep.",
        "",
        "## 17. What this enables next",
        "",
        "Phase 7 (Adaptive Federated Learning) can now build on a real, "
        "honestly-measured FedAvg baseline: does adapting client "
        "selection or aggregation (e.g. using the simulated resource "
        "metadata already being logged) improve on this baseline's "
        "KDDTest+ result, its communication cost, or both? No adaptive "
        "selection, staleness handling, or DRL has been implemented in "
        "Phase 6 — standard FedAvg only.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

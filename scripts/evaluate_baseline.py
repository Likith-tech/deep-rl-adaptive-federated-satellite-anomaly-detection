"""Final, one-time evaluation of the baseline MLP on KDDTest+.

Run from the repository root, after scripts/train_baseline.py has produced
a checkpoint:

    python scripts/evaluate_baseline.py

Reads:
    results/models/baseline/best_model.pt
    results/models/baseline/training_history.json
    data/processed/test.parquet   (KDDTest+ — never used for training/tuning)

Writes:
    results/plots/baseline/confusion_matrix.png
    results/plots/baseline/training_curves.png
    results/reports/baseline_results.md

This script is meant to be run ONCE per model version to report the
final test metrics. Do not use it in a tuning loop against the test set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.baseline_mlp import BaselineMLP  # noqa: E402
from src.training.baseline_trainer import load_features_and_labels  # noqa: E402

CHECKPOINT_PATH = REPO_ROOT / "results" / "models" / "baseline" / "best_model.pt"
HISTORY_PATH = REPO_ROOT / "results" / "models" / "baseline" / "training_history.json"
TEST_PARQUET = REPO_ROOT / "data" / "processed" / "test.parquet"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "baseline"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "baseline_results.md"


def plot_confusion_matrix(cm: list[list[int]], out_path: Path) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Normal (0)", "Anomaly (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Baseline MLP on KDDTest+", fontsize=12, pad=15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: dict, out_path: Path) -> None:
    epochs_data = history["epochs"]
    epochs = [e["epoch"] for e in epochs_data]
    train_loss = [e["train_loss"] for e in epochs_data]
    val_loss = [e["val_loss"] for e in epochs_data]
    val_f1 = [e["val_f1"] for e in epochs_data]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="Train loss", color="#3b82f6")
    axes[0].plot(epochs, val_loss, label="Validation loss", color="#ef4444")
    axes[0].axvline(history["best_epoch"], color="gray", linestyle="--", alpha=0.6, label="Best epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="Validation F1", color="#22c55e")
    axes[1].axvline(history["best_epoch"], color="gray", linestyle="--", alpha=0.6, label="Best epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 score")
    axes[1].set_title("Validation F1 per Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: {CHECKPOINT_PATH} not found. Run scripts/train_baseline.py first.", file=sys.stderr)
        sys.exit(1)
    if not TEST_PARQUET.exists():
        print(f"ERROR: {TEST_PARQUET} not found. Run the Phase 1 pipeline first.", file=sys.stderr)
        sys.exit(1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    config = checkpoint["config"]
    feature_columns = checkpoint["feature_columns"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineMLP(
        input_dim=config["input_dim"],
        hidden_dimensions=config["hidden_dimensions"],
        dropout=config["dropout"],
        output_dim=config["output_dim"],
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

    plot_confusion_matrix(test_metrics.confusion_matrix, PLOTS_DIR / "confusion_matrix.png")

    history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else None
    if history:
        plot_training_curves(history, PLOTS_DIR / "training_curves.png")

    print("\n=== Final test metrics (KDDTest+) ===")
    for key, value in test_metrics.to_dict().items():
        print(f"{key}: {value}")

    # --- Results report (real numbers only) ---
    best_val_metrics = checkpoint.get("val_metrics", {})
    lines = [
        "# Baseline Results — MLP Anomaly Detector",
        "",
        "## Dataset",
        "",
        "NSL-KDD (terrestrial network intrusion dataset — see "
        "`docs/datasets/dataset_selection.md`). Processed via the Phase 1 "
        "pipeline: cleaned, labeled, split (train/validation/KDDTest+), "
        "one-hot encoded, standard-scaled.",
        "",
        "## Model architecture",
        "",
        f"Feed-forward MLP: input({config['input_dim']}) -> "
        + " -> ".join(f"Dense({h}) -> ReLU -> Dropout({config['dropout']})" for h in config["hidden_dimensions"])
        + f" -> Dense({config['output_dim']}) -> logit (sigmoid at inference)",
        "",
        "## Hyperparameters",
        "",
        f"- Batch size: {config['training']['batch_size']}",
        f"- Learning rate: {config['training']['learning_rate']}",
        f"- Optimizer: {config['training']['optimizer']} (Adam)",
        f"- Loss: {config['training']['loss']} (BCEWithLogitsLoss)",
        f"- Max epochs: {config['training']['epochs']}",
        f"- Early stopping patience: {config['training']['early_stopping_patience']} epochs (on validation loss)",
        f"- Seed: {config['training']['seed']}",
        "",
        "## Training",
        "",
    ]
    if history:
        lines += [
            f"- Epochs run: {len(history['epochs'])} "
            f"({'stopped early' if history['stopped_early'] else 'completed max epochs'})",
            f"- Best epoch (by validation loss): {history['best_epoch']}",
            f"- Best validation loss: {history['best_val_loss']:.4f}",
            f"- Training time: {history['training_seconds']:.1f}s",
        ]
    else:
        lines.append("- Training history not found — could not report epoch-by-epoch detail.")

    lines += [
        "",
        "## Best validation performance (checkpoint selection)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in ("accuracy", "precision", "recall", "f1"):
        val = best_val_metrics.get(key)
        lines.append(f"| {key} | {val:.4f} |" if val is not None else f"| {key} | not available |")

    lines += [
        "",
        "## Final test performance (KDDTest+, evaluated once)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {test_metrics.accuracy:.4f} |",
        f"| Precision | {test_metrics.precision:.4f} |",
        f"| Recall | {test_metrics.recall:.4f} |",
        f"| F1 | {test_metrics.f1:.4f} |",
        f"| ROC-AUC | {test_metrics.roc_auc:.4f} |" if test_metrics.roc_auc is not None else "| ROC-AUC | not available |",
        f"| False Positive Rate | {test_metrics.false_positive_rate:.4f} |",
        "",
        "## Confusion matrix (KDDTest+)",
        "",
        f"```\n"
        f"                Predicted Normal   Predicted Anomaly\n"
        f"Actual Normal   {test_metrics.confusion_matrix[0][0]:>16}   {test_metrics.confusion_matrix[0][1]:>18}\n"
        f"Actual Anomaly  {test_metrics.confusion_matrix[1][0]:>16}   {test_metrics.confusion_matrix[1][1]:>18}\n"
        f"```",
        "",
        "See `results/plots/baseline/confusion_matrix.png` for the visual version.",
        "",
        "## Training curves",
        "",
        "See `results/plots/baseline/training_curves.png` "
        "(training/validation loss per epoch, validation F1 per epoch)."
        if history else "Not generated — training history unavailable.",
        "",
        "## Notes",
        "",
        "- KDDTest+ was used for this final evaluation only — never for "
        "training or checkpoint selection (validation loss on the "
        "held-out validation split was used for that).",
        "- KDDTest+ contains attack types absent from training (by NSL-KDD's "
        "own design), which affects generalization and should be kept in "
        "mind when interpreting recall/precision here.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

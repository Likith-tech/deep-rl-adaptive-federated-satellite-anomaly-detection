"""Phase 9A — final, ONE-TIME evaluation of the OPSSAT-AD temporal model
on the author's held-out test segments (train=0, 529 segments).

Run from the repository root, after scripts/train_opssat_temporal.py:

    python scripts/evaluate_opssat_temporal.py

Reads:
    results/models/opssat_temporal/best_model.pt
    results/models/opssat_preprocessing/{scaler,encoder}.joblib
    data/raw/opssat_ad/dataset.csv (only the train=0 segments are used)

Writes:
    results/plots/opssat_ad/{confusion_matrix,training_curves}.png
    results/reports/opssat_ad_results.md

This script is meant to be run ONCE per model version. The 529 test
segments are not used anywhere else in this phase — not for training,
not for validation, not for model/config selection.

CRITICAL LABELING REMINDER (enforced in the report text below, not just
this comment): these are REAL spacecraft OPERATIONAL anomaly detection
results. They are NOT cyberattack detection results, and OPSSAT-AD
contains no cyberattack labels of any kind.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import pandas/sklearn before torch/matplotlib — see the note in
# scripts/evaluate_local_models.py (Phase 5) about avoiding an
# intermittent access-violation crash on this Windows/Python 3.14 env.
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.satellite.loader import load_opssat_dataset, split_by_author_train_flag  # noqa: E402
from src.satellite.preprocessing import OpsSatPreprocessingArtifacts, extract_labels, transform  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sklearn.metrics import matthews_corrcoef, average_precision_score  # noqa: E402

from src.models.temporal_gru_attention import TemporalGRUAttention  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "opssat_ad.yaml"
CHECKPOINT_PATH = REPO_ROOT / "results" / "models" / "opssat_temporal" / "best_model.pt"
HISTORY_PATH = REPO_ROOT / "results" / "models" / "opssat_temporal" / "training_history.json"
ARTIFACTS_DIR = REPO_ROOT / "results" / "models" / "opssat_preprocessing"
RAW_DIR = REPO_ROOT / "data" / "raw" / "opssat_ad"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "opssat_ad"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "opssat_ad_results.md"
RUN_CONFIG_PATH = REPO_ROOT / "experiments" / "opssat_ad" / "run_config.json"

# Phase 3 NSL-KDD reference (FINAL KDDTest+ result) — fixed, not re-derived,
# shown side by side ONLY as a dataset-labeled comparison, never merged.
PHASE_3_NSL_KDD_REFERENCE = {
    "dataset": "NSL-KDD (KDDTest+, terrestrial cyberattack detection)",
    "accuracy": 0.7580, "precision": 0.9220, "recall": 0.6400, "f1": 0.7558, "roc_auc": 0.8923,
}


def plot_confusion_matrix(cm: list[list[int]], out_path: Path) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(cm_arr, cmap="Blues")
    labels = ["Nominal (0)", "Anomalous (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("OPSSAT-AD — Confusion Matrix (real spacecraft OPERATIONAL\nanomalies, NOT cyberattacks) — held-out test segments", fontsize=10, pad=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                     color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black", fontsize=13)
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
    axes[0].set_title("OPSSAT-AD — Training / Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="Validation F1", color="#22c55e")
    axes[1].axvline(history["best_epoch"], color="gray", linestyle="--", alpha=0.6, label="Best epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 score")
    axes[1].set_title("OPSSAT-AD — Validation F1 per Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: {CHECKPOINT_PATH} not found. Run scripts/train_opssat_temporal.py first.", file=sys.stderr)
        sys.exit(1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    config = checkpoint["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalGRUAttention(
        input_dim=config["model"]["input_dim"],
        projection_dim=config["model"]["projection_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_gru_layers=config["model"]["num_gru_layers"],
        dropout=config["model"]["dropout"],
        output_dim=config["model"]["output_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    artifacts = OpsSatPreprocessingArtifacts.load(ARTIFACTS_DIR)

    df = load_opssat_dataset(RAW_DIR)
    _, author_test_df = split_by_author_train_flag(df)
    print(f"Evaluating on {len(author_test_df)} held-out OPSSAT-AD test segments "
          "(train=0 — never used for training or model selection).")

    X_test = transform(author_test_df, artifacts)
    y_test = extract_labels(author_test_df)
    X_test_seq = X_test[:, np.newaxis, :]

    with torch.no_grad():
        logits, _ = model(torch.from_numpy(X_test_seq).to(device))
        y_prob = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    y_true = y_test.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    test_metrics = compute_metrics(y_true, y_pred, y_prob)
    mcc = float(matthews_corrcoef(y_true, y_pred))
    aucpr = float(average_precision_score(y_true, y_prob))

    print("\n=== Final OPSSAT-AD test metrics (real spacecraft OPERATIONAL anomalies) ===")
    for key, value in test_metrics.to_dict().items():
        print(f"{key}: {value}")
    print(f"mcc: {mcc}")
    print(f"aucpr: {aucpr}")

    plot_confusion_matrix(test_metrics.confusion_matrix, PLOTS_DIR / "confusion_matrix.png")

    history = json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else None
    if history:
        plot_training_curves(history, PLOTS_DIR / "training_curves.png")

    run_config = json.loads(RUN_CONFIG_PATH.read_text()) if RUN_CONFIG_PATH.exists() else {}

    # --- Results report (real numbers only) ---
    lines = [
        "# Phase 9A — OPSSAT-AD Real Spacecraft Telemetry Validation Results",
        "",
        "## What this is, and what it is NOT",
        "",
        "**REAL:** the telemetry (from ESA's OPS-SAT CubeSat, a real, flown, "
        "now-decommissioned mission) and the anomaly labels (expert-annotated "
        "operational anomalies) below are genuine.",
        "",
        "**NOT PROVIDED by this dataset, and NOT claimed here:** cyberattack "
        "labels of any kind, and real multi-satellite federated data. The "
        "anomalies detected below are operational/physical telemetry "
        "anomalies — sensor/subsystem behavior outside its normal envelope "
        "— NOT malicious network traffic or intrusions. This is a completely "
        "separate track from the NSL-KDD-based cybersecurity/FL work in "
        "Phases 1-8, and the two are never merged into a single metric.",
        "",
        "## 1. Dataset",
        "",
        "OPSSAT-AD (ESA OPS-SAT CubeSat telemetry anomaly benchmark), "
        "CC-BY-4.0 (per Zenodo's structured license metadata; the bundled "
        "`LICENSE` file's own text says MIT — both permissive, both recorded "
        "in `results/reports/opssat_ad_inspection.md` rather than silently "
        "resolved). Source: https://zenodo.org/records/12588358. "
        f"{run_config.get('author_train_segments', '?')} author-training "
        f"segments, {run_config.get('author_test_segments', '?')} author-test "
        "segments (author-defined split, used exactly as provided).",
        "",
        "## 2. Model architecture",
        "",
        "The SAME `TemporalGRUAttention` class Phase 3 uses on NSL-KDD "
        "(`src/models/temporal_gru_attention.py`), completely UNCHANGED — no "
        "fork, no modification. **Important honesty note:** OPSSAT-AD's "
        "`dataset.csv` gives one fixed-width feature vector per segment "
        "with no timestep dimension (confirmed by direct inspection — see "
        "`results/reports/opssat_ad_inspection.md`), so every sample is fed "
        "as a sequence of length 1. This means the GRU's recurrence and the "
        "temporal attention mechanism are NOT meaningfully exercised in this "
        "run — this validates the model's classification head and general "
        "architecture on real spacecraft-derived features, but is NOT a "
        "genuine test of sequential/temporal modeling on OPS-SAT data. A "
        "true temporal test would require the raw `segments.csv` file, "
        "explicitly out of scope for this phase (see the Phase 9A "
        "pre-integration audit).",
        "",
        f"- Input dimension: {config['model']['input_dim']} (19 numeric + 9 one-hot channel features)",
        f"- Projection dim: {config['model']['projection_dim']}, hidden dim: {config['model']['hidden_dim']}",
        f"- GRU layers: {config['model']['num_gru_layers']}, dropout: {config['model']['dropout']}",
        "",
        "## 3. Preprocessing",
        "",
        "Entirely independent of the NSL-KDD pipeline — no file, scaler, or "
        "encoder from `src/preprocessing/` or "
        "`results/models/preprocessing/` is read or modified. A new "
        "`StandardScaler` (19 numeric columns) and `OneHotEncoder` (the "
        "`channel` column, 9 categories) were fit **only on the training "
        "portion** (see section 4) and reused unchanged for validation and "
        "test.",
        "",
        "## 4. Train/validation/test split",
        "",
        f"- Author-provided split (via `dataset.csv`'s own `train` column): "
        f"{run_config.get('author_train_segments', '?')} training segments, "
        f"{run_config.get('author_test_segments', '?')} test segments — used exactly as given.",
        f"- This project's own validation carve-out (from the author's training "
        f"segments ONLY, stratified by label, seed={config['validation']['seed']}, "
        f"validation_size={config['validation']['validation_size']}): "
        f"{run_config.get('train_segments', '?')} train / {run_config.get('validation_segments', '?')} validation.",
        "- The 529 author-test segments were never touched until this evaluation.",
        "",
        "## 5. Training configuration",
        "",
        f"- Optimizer: {config['training']['optimizer']} (Adam)",
        f"- Learning rate: {config['training']['learning_rate']}",
        f"- Loss: {config['training']['loss']} (BCEWithLogitsLoss)",
        f"- Batch size: {config['training']['batch_size']}",
        f"- Max epochs: {config['training']['epochs']}",
        f"- Early stopping patience: {config['training']['early_stopping_patience']} epochs (on validation loss)",
        f"- Seed: {config['training']['seed']}",
        "",
        "## 6. Validation results (best epoch, checkpoint selection)",
        "",
    ]
    if history:
        lines += [
            f"- Epochs run: {len(history['epochs'])} "
            f"({'stopped early' if history['stopped_early'] else 'completed max epochs'})",
            f"- Best epoch (by validation loss): {history['best_epoch']}",
            f"- Best validation loss: {history['best_val_loss']:.4f}",
            f"- Training time: {history['training_seconds']:.1f}s",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        best_epoch_record = next(e for e in history["epochs"] if e["epoch"] == history["best_epoch"])
        for key in ("val_accuracy", "val_precision", "val_recall", "val_f1"):
            lines.append(f"| {key} | {best_epoch_record[key]:.4f} |")
    lines += [
        "",
        "See `results/plots/opssat_ad/training_curves.png`.",
        "",
        "## 7. Final KDDTest+-style evaluation — OPSSAT-AD held-out test segments (evaluated ONCE)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Accuracy | {test_metrics.accuracy:.4f} |",
        f"| Precision | {test_metrics.precision:.4f} |",
        f"| Recall | {test_metrics.recall:.4f} |",
        f"| F1 | {test_metrics.f1:.4f} |",
        f"| ROC-AUC | {test_metrics.roc_auc:.4f} |" if test_metrics.roc_auc is not None else "| ROC-AUC | not available |",
        f"| False Positive Rate | {test_metrics.false_positive_rate:.4f} |",
        f"| MCC (Matthews Correlation Coefficient) | {mcc:.4f} |",
        f"| AUCPR (Area Under Precision-Recall Curve) | {aucpr:.4f} |",
        "",
        "**These are real spacecraft OPERATIONAL anomaly detection results, "
        "NOT cyberattack detection results.**",
        "",
        "## 8. Confusion matrix (OPSSAT-AD test segments)",
        "",
        "```\n"
        "                 Predicted Nominal   Predicted Anomalous\n"
        f"Actual Nominal   {test_metrics.confusion_matrix[0][0]:>16}   {test_metrics.confusion_matrix[0][1]:>19}\n"
        f"Actual Anomalous {test_metrics.confusion_matrix[1][0]:>16}   {test_metrics.confusion_matrix[1][1]:>19}\n"
        "```",
        "",
        "See `results/plots/opssat_ad/confusion_matrix.png`.",
        "",
        "## 9. Comparison with Phase 3 (dataset-labeled, never merged)",
        "",
        "**These two rows are NOT a like-for-like benchmark.** They are shown "
        "side by side only to give a rough sense of scale, each explicitly "
        "labeled with its own dataset and task — never combined into one score.",
        "",
        "| Track | Dataset | Task | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        f"| Phase 3 | {PHASE_3_NSL_KDD_REFERENCE['dataset']} | Cyberattack detection | "
        f"{PHASE_3_NSL_KDD_REFERENCE['accuracy']:.4f} | {PHASE_3_NSL_KDD_REFERENCE['precision']:.4f} | "
        f"{PHASE_3_NSL_KDD_REFERENCE['recall']:.4f} | {PHASE_3_NSL_KDD_REFERENCE['f1']:.4f} | "
        f"{PHASE_3_NSL_KDD_REFERENCE['roc_auc']:.4f} |",
        f"| Phase 9A | OPSSAT-AD (real ESA CubeSat telemetry) | Operational anomaly detection | "
        f"{test_metrics.accuracy:.4f} | {test_metrics.precision:.4f} | {test_metrics.recall:.4f} | "
        f"{test_metrics.f1:.4f} | {test_metrics.roc_auc:.4f} |",
        "",
        "The OPSSAT-AD numbers are higher, but this reflects a smaller, "
        "cleaner, less ambiguous benchmark (2,123 total segments vs. "
        "NSL-KDD's 148,517 records, and OPSSAT-AD's test set does not "
        "contain the deliberately novel/harder attack types NSL-KDD's "
        "KDDTest+ does) — not evidence that either task is intrinsically "
        "\"easier\" to solve in general.",
        "",
        "## 10. Limitations",
        "",
        "- **Single spacecraft.** No genuine multi-satellite claim is "
        "supportable from this dataset.",
        "- **No cyberattack labels.** Every anomaly here is an operational/"
        "physical telemetry anomaly, not malicious activity.",
        "- **Sequence length of 1.** As discussed in section 2, this run "
        "does not genuinely exercise temporal/sequential modeling — it "
        "validates the model's architecture and classification head on "
        "real spacecraft-derived tabular features.",
        "- **Small dataset.** 1,594 author-training segments (1,354 after "
        "this project's own validation carve-out) is far smaller than "
        "NSL-KDD's 107,077 training records — results should be read with "
        "that sample size in mind.",
        "- **No federated/multi-client structure.** Any FL use of this data "
        "would require artificial, explicitly-simulated partitioning — none "
        "was performed in this phase.",
        "- Channel identity (`CADC08xx` codes) is opaque without ESA's "
        "internal documentation — this project does not know, and does not "
        "claim to know, the exact physical subsystem each code maps to "
        "beyond the paper's general description (magnetometer/photodiode).",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

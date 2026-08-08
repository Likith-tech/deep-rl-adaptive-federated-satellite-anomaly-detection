"""Phase 5 — cross-client evaluation, plots, and results report.

Run from the repository root, after scripts/train_local_models.py has
produced all 8 clients' checkpoints:

    python scripts/evaluate_local_models.py

Reads:
    data/partitions/manifest.json
    results/models/local/<client>/{best_model.pt,training_history.json,metadata.json}
    data/processed/validation.parquet   (GLOBAL validation set — used for all cross-client comparison)

Writes:
    results/plots/local_training/confusion_matrices/<client>_confusion_matrix.png
    results/plots/local_training/training_curves/<client>_training_curves.png
    results/plots/local_training/client_comparison.png
    results/reports/local_training_results.md

KDDTest+ is never touched here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import pandas/sklearn (pulls in pyarrow) before torch/matplotlib —
# on this Windows/Python 3.14 environment, importing torch first causes
# an intermittent access-violation crash during pyarrow's parquet
# reader initialization. Order matters here; do not reorder casually.
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.training.baseline_trainer import load_features_and_labels  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.evaluation.local_evaluation import (  # noqa: E402
    build_client_summary,
    compute_aggregate_statistics,
    plot_confusion_matrix,
    plot_training_curves,
    rank_clients_by_metric,
)
from src.models.baseline_mlp import BaselineMLP  # noqa: E402

PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
LOCAL_MODELS_DIR = REPO_ROOT / "results" / "models" / "local"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "local_training"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "local_training_results.md"

# Phase 2 centralized baseline, FINAL KDDTest+ results — fixed reference
# values, not re-derived here. See results/reports/baseline_results.md.
CENTRALIZED_BASELINE_TESTSET = {
    "accuracy": 0.7832,
    "precision": 0.9278,
    "recall": 0.6713,
    "f1": 0.7790,
    "roc_auc": 0.8972,
    "false_positive_rate": 0.0690,
}


def plot_client_comparison(client_ids: list[str], f1_scores: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#3b82f6"] * len(client_ids)
    ax.bar(client_ids, f1_scores, color=colors)
    ax.set_ylabel("Global validation F1")
    ax.set_title("Per-Client Global Validation F1 (Phase 5 — local training only)")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    manifest = json.loads((PARTITIONS_DIR / "manifest.json").read_text())
    client_ids = sorted(manifest["client_sample_counts"].keys())

    validation_parquet = PROCESSED_DIR / "validation.parquet"

    cm_dir = PLOTS_DIR / "confusion_matrices"
    curves_dir = PLOTS_DIR / "training_curves"
    cm_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summaries = []
    for client_id in client_ids:
        client_dir = LOCAL_MODELS_DIR / client_id
        metadata_path = client_dir / "metadata.json"
        history_path = client_dir / "training_history.json"
        checkpoint_path = client_dir / "best_model.pt"
        if not (metadata_path.exists() and checkpoint_path.exists()):
            print(f"WARNING: missing results for {client_id} — skipping. Run scripts/train_local_models.py first.",
                  file=sys.stderr)
            continue

        metadata = json.loads(metadata_path.read_text())
        history = json.loads(history_path.read_text())

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        config = checkpoint["config"]
        feature_columns = checkpoint["feature_columns"]

        model = BaselineMLP(
            input_dim=config["model"]["input_dim"] if "model" in config else config["input_dim"],
            hidden_dimensions=config["model"]["hidden_dimensions"] if "model" in config else config["hidden_dimensions"],
            dropout=config["model"]["dropout"] if "model" in config else config["dropout"],
            output_dim=config["model"]["output_dim"] if "model" in config else config["output_dim"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        X_val, y_val = load_features_and_labels(validation_parquet, feature_columns)
        with torch.no_grad():
            logits = model(torch.from_numpy(X_val).to(device)).squeeze(-1)
            y_prob = torch.sigmoid(logits).cpu().numpy()
        y_true = y_val.astype(int)
        y_pred = (y_prob >= 0.5).astype(int)
        val_metrics = compute_metrics(y_true, y_pred, y_prob)

        plot_confusion_matrix(
            val_metrics.confusion_matrix,
            f"{client_id} — Confusion Matrix (Global Validation)",
            cm_dir / f"{client_id}_confusion_matrix.png",
        )
        plot_training_curves(client_id, history["epochs"], history["best_epoch"],
                              curves_dir / f"{client_id}_training_curves.png")

        client_stats = manifest["client_stats"][client_id]
        summary = build_client_summary(
            client_id=client_id,
            client_stats=client_stats,
            local_train_metrics=metadata["local_train_metrics"],
            global_val_metrics=val_metrics.to_dict(),
        )
        summaries.append(summary)
        print(f"{client_id}: global val f1={val_metrics.f1:.4f} accuracy={val_metrics.accuracy:.4f}")

    if not summaries:
        print("ERROR: no client results found. Run scripts/train_local_models.py first.", file=sys.stderr)
        sys.exit(1)

    plot_client_comparison(
        [s.client_id for s in summaries],
        [s.global_val_metrics["f1"] for s in summaries],
        PLOTS_DIR / "client_comparison.png",
    )

    aggregates = compute_aggregate_statistics(summaries)
    ranked_best_to_worst = rank_clients_by_metric(summaries, "f1")

    # --- Results report (real numbers only) ---
    lines = [
        "# Phase 5 — Local Satellite Training Results",
        "",
        "## 1. Purpose",
        "",
        "Establish how well each simulated satellite client can detect "
        "anomalies using ONLY its own local, non-IID training data, with "
        "no communication or aggregation between clients. This is the "
        "reference point Federated Learning (Phase 6+) will be compared "
        "against.",
        "",
        "## 2. Local satellite setup",
        "",
        "Satellite clients and their local data partitions come from "
        "Phase 4 (`data/partitions/`, Dirichlet non-IID partitioning, "
        "alpha=0.5, seed=42). Every local data record is a REAL NSL-KDD "
        "training record; the satellite/client framing itself is "
        "SIMULATED. See `docs/project-progress/05-phase-4-satellite-simulation.md`.",
        "",
        f"## 3. Number of clients",
        "",
        f"{len(summaries)} clients trained: {', '.join(s.client_id for s in summaries)}.",
        "",
        "## 4. Local data sizes",
        "",
        "| Client | Total samples | Anomaly % | Categories present | Categories absent |",
        "|---|---:|---:|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.client_id} | {s.total_samples:,} | {s.anomaly_percentage:.2f}% | "
            f"{', '.join(s.categories_present)} | {', '.join(s.categories_absent) or 'none'} |"
        )

    lines += [
        "",
        "## 5. Model architecture",
        "",
        "Same architecture as the Phase 2 baseline MLP: "
        "input(121) -> Dense(128) -> ReLU -> Dropout(0.3) -> Dense(64) -> "
        "ReLU -> Dropout(0.3) -> Dense(1) -> logit (sigmoid at inference). "
        "Every client trains an independent copy, all starting from the "
        "SAME shared initial weights (see section 7 below).",
        "",
        "## 6. Preprocessing",
        "",
        "No preprocessing is refit per client. Every client's parquet "
        "file already contains the same one-hot-encoded, standard-scaled "
        "feature representation produced once by the Phase 1 pipeline "
        "(`results/models/preprocessing/{scaler,encoder}.joblib`) before "
        "Phase 4 partitioned the training data. Feature columns are "
        "identical and in the same order across all clients "
        "(`results/models/preprocessing/feature_columns.json`).",
        "",
        "## 7. Training configuration",
        "",
    ]
    example_meta = json.loads((LOCAL_MODELS_DIR / summaries[0].client_id / "metadata.json").read_text())
    tc = example_meta["training_config"]
    lines += [
        f"- Optimizer: {tc['optimizer']} (Adam)",
        f"- Learning rate: {tc['learning_rate']}",
        f"- Loss: {tc['loss']} (BCEWithLogitsLoss)",
        f"- Batch size: {tc['batch_size']}",
        f"- Max epochs: {tc['epochs']}",
        f"- Early stopping patience: {tc['early_stopping_patience']} epochs (on global validation loss)",
        f"- Training seed: {tc['seed']}",
        f"- Shared initial-weights seed: {example_meta['shared_init_seed']} "
        "(every client starts from an identical initial model state, so "
        "differences in outcomes reflect local data, not initialization)",
        "- Validation strategy: global validation set "
        "(`data/processed/validation.parquet`) evaluated after every "
        "epoch, for checkpoint selection AND as the common cross-client "
        "reference. KDDTest+ untouched.",
        "",
        "## 8. Local training metrics",
        "",
        "Metrics computed on each client's OWN local training data "
        "(the data it was trained on) — reflects what the model learned "
        "about its own local distribution, not generalization.",
        "",
        "| Client | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        m = s.local_train_metrics
        roc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "not available"
        lines.append(
            f"| {s.client_id} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {roc} | {m['false_positive_rate']:.4f} |"
        )

    lines += [
        "",
        "## 9. Global validation metrics",
        "",
        "Metrics computed on the SAME held-out global validation set for "
        "every client — this is the fair, common basis for comparing "
        "clients against each other and later against Federated Learning.",
        "",
        "| Client | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        m = s.global_val_metrics
        roc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "not available"
        lines.append(
            f"| {s.client_id} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {roc} | {m['false_positive_rate']:.4f} |"
        )

    lines += [
        "",
        "## 10. Per-client comparison",
        "",
        "| Satellite | Samples | Anomaly % | Val Accuracy | Val Precision | Val Recall | Val F1 | Val ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        m = s.global_val_metrics
        roc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "not available"
        lines.append(
            f"| {s.client_id} | {s.total_samples:,} | {s.anomaly_percentage:.2f}% | "
            f"{m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {roc} |"
        )
    lines += ["", "See `results/plots/local_training/client_comparison.png` for the visual version."]

    lines += [
        "",
        "## 11. Aggregate statistics (across all clients, global validation)",
        "",
        "| Metric | Mean | Median | Min | Max | Std dev |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"):
        agg = aggregates.get(key)
        if agg is None:
            lines.append(f"| {key} | not available | not available | not available | not available | not available |")
        else:
            lines.append(
                f"| {key} | {agg['mean']:.4f} | {agg['median']:.4f} | {agg['min']:.4f} | "
                f"{agg['max']:.4f} | {agg['stdev']:.4f} |"
            )

    best = ranked_best_to_worst[0]
    worst = ranked_best_to_worst[-1]
    lines += [
        "",
        "## 12. Non-IID observations",
        "",
        f"Best global-validation F1: **{best.client_id}** ({best.global_val_metrics['f1']:.4f}), "
        f"local anomaly rate {best.anomaly_percentage:.2f}%, "
        f"{best.total_samples:,} samples, "
        f"missing categories: {', '.join(best.categories_absent) or 'none'}.",
        "",
        f"Worst global-validation F1: **{worst.client_id}** ({worst.global_val_metrics['f1']:.4f}), "
        f"local anomaly rate {worst.anomaly_percentage:.2f}%, "
        f"{worst.total_samples:,} samples, "
        f"missing categories: {', '.join(worst.categories_absent) or 'none'}.",
        "",
    ]
    missing_cat_clients = [s for s in summaries if s.categories_absent]
    if missing_cat_clients:
        lines.append(
            "Clients missing at least one attack category in their local "
            "training data: " + ", ".join(
                f"{s.client_id} (missing: {', '.join(s.categories_absent)})" for s in missing_cat_clients
            ) + "."
        )
    else:
        lines.append("All clients had at least one sample of every attack category represented locally.")
    lines += [
        "",
        "Observed pattern (association, not proven causation — see caveat "
        "below): clients with extreme local anomaly rates (very high or "
        "very low) and/or missing attack categories tend to sit at the "
        "extremes of the global-validation F1 ranking above. This is "
        "consistent with, but does not by itself prove, that non-IID "
        "local data limits what a purely local model can learn.",
        "",
        "**Caveat:** with only 8 clients, this is a small, observational "
        "sample — good for describing what happened in this run, not for "
        "establishing statistical causation. Language above is "
        "deliberately correlational, not causal.",
        "",
        "## 13. Comparison to centralized baseline",
        "",
        "**Important dataset caveat:** the Phase 2 numbers below are "
        "FINAL, one-time KDDTest+ results. The Phase 5 numbers are "
        "global-VALIDATION results (used for development/checkpoint "
        "selection, per the project's rule that KDDTest+ stays untouched "
        "until final federated evaluation). These are NOT a like-for-like "
        "test-set comparison — shown side by side only to give a rough "
        "sense of scale, not a rigorous benchmark.",
        "",
        "| Metric | Phase 2 centralized (KDDTest+, final) | Phase 5 aggregate mean (global validation) |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
        ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("false_positive_rate", "FPR"),
    ):
        centralized_val = CENTRALIZED_BASELINE_TESTSET[key]
        agg = aggregates.get(key)
        agg_str = f"{agg['mean']:.4f}" if agg else "not available"
        lines.append(f"| {label} | {centralized_val:.4f} | {agg_str} |")

    lines += [
        "",
        "## 14. Limitations",
        "",
        "- NSL-KDD is a terrestrial network intrusion dataset — it is not, "
        "and has never been claimed to be, real satellite telemetry.",
        "- The satellite/client environment (client IDs, and the fact "
        "that this is 'satellites' at all) is simulated; only the "
        "underlying network records are real.",
        "- Local models never communicate with each other in this phase "
        "— no aggregation, no FedAvg, no global model. Any similarity "
        "between clients' models is coincidental (same architecture, "
        "same initial weights), not the result of information sharing.",
        "- Phase 5 metrics use the global VALIDATION set, not KDDTest+ "
        "— see the explicit caveat in section 13.",
        "- With only 8 clients, aggregate statistics (mean/std/etc.) are "
        "based on a small sample; treat spread numbers as descriptive, "
        "not as a statistically powered study.",
        "- Some clients have very few or zero samples of rare attack "
        "categories (R2L, U2R) purely because those categories are rare "
        "in the underlying real dataset overall — not a partitioning bug.",
        "",
        "## 15. What this enables next",
        "",
        "Phase 6 (Federated Learning Baseline) can now be evaluated "
        "against a real, honestly-measured local-only reference: does "
        "aggregating clients' updates (e.g. via FedAvg) actually improve "
        "on what any single satellite could achieve alone, and does it "
        "close the gap to the Phase 2 centralized result? No Federated "
        "Learning, DRL, or model aggregation has been implemented in "
        "Phase 5 — environment (Phase 4) and local-only training (Phase "
        "5) only.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

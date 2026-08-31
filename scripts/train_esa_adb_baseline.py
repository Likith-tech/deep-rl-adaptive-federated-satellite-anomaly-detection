"""ESA-ADB Experiment 1 — centralized (non-federated) baseline.

Establishes a reference point BEFORE any FL/DQN work. Trains
TemporalGRUAttention(input_dim=1) on the centralized (all 58 channels
pooled together, ignoring the 8-client partition) train split, selects
the decision threshold on validation only, evaluates once on the
untouched test split.

Run from the repository root, after the ESA-ADB Option B pipeline has
been built (scripts/run_esa_adb_pipeline.py):

    python scripts/train_esa_adb_baseline.py

Writes everything to experiments/esa_adb/baseline_01/ — does not touch
any other experiment directory, any protected Phase 6-9B/OPSSAT-AD
artifact, or any existing checkpoint.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.preprocessing.sequences import SequenceDataset  # noqa: E402
from src.satellite.esa_adb import clients as clients_mod  # noqa: E402
from src.satellite.esa_adb import loader  # noqa: E402
from src.satellite.esa_adb import pipeline  # noqa: E402
from src.satellite.esa_adb import preprocessing as prep  # noqa: E402
from src.training.esa_adb_trainer import (  # noqa: E402
    compute_extended_metrics,
    curve_data,
    make_dataloader,
    run_inference,
    select_threshold_by_f1,
    train_esa_adb_baseline,
)

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "esa_adb" / "baseline_01"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "esa_adb" / "manifest.json"
CLIENT_METADATA_PATH = REPO_ROOT / "data" / "processed" / "esa_adb" / "client_metadata.json"

EXPECTED_APPROX = {
    "train": {"num_sequences": 121510, "anomaly": 3888},
    "validation": {"num_sequences": 61364, "anomaly": 104},
    "test": {"num_sequences": 61364, "anomaly": 1413},
}

CONFIG = {
    "input_dim": 1,
    "projection_dim": 64,
    "hidden_dim": 64,
    "num_gru_layers": 1,
    "dropout": 0.3,
    "output_dim": 1,
    "sequence_length": prep.SEQUENCE_LENGTH,
    "stride": prep.STRIDE,
    "label_strategy": prep.LABEL_STRATEGY,
    "training": {
        "batch_size": 128,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "bce_with_logits",
        "epochs": 25,
        "early_stopping_patience": 4,
        "seed": 42,
    },
}


def pool_all_channels(channel_results: dict, split: str) -> SequenceDataset:
    return pipeline.pool_sequence_datasets(
        [channel_results[c].sequences[split] for c in channel_results], prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY
    )


def sanity_checks(train_ds: SequenceDataset, val_ds: SequenceDataset, test_ds: SequenceDataset, channel_results: dict) -> dict:
    results = {}

    # 1. Sequence dimensions.
    shapes_ok = all(ds.sequences.shape[1:] == (prep.SEQUENCE_LENGTH, 1) for ds in (train_ds, val_ds, test_ds))
    results["sequence_dimensions"] = {"passed": shapes_ok, "train_shape": list(train_ds.sequences.shape),
                                       "validation_shape": list(val_ds.sequences.shape), "test_shape": list(test_ds.sequences.shape)}

    # 2. Train/validation/test counts match the approved candidate analysis (approximately, real counts must match manifest exactly).
    counts_ok = True
    count_detail = {}
    for split, ds in (("train", train_ds), ("validation", val_ds), ("test", test_ds)):
        expected = EXPECTED_APPROX[split]
        actual_seq, actual_anom = ds.num_sequences, ds.anomaly_count
        matches = actual_seq == expected["num_sequences"] and actual_anom == expected["anomaly"]
        counts_ok = counts_ok and matches
        count_detail[split] = {"expected": expected, "actual": {"num_sequences": actual_seq, "anomaly": actual_anom}, "matches": matches}
    results["expected_counts"] = {"passed": counts_ok, "detail": count_detail}

    # 3. Labels are binary only.
    labels_ok = all(set(np.unique(ds.labels).tolist()).issubset({0, 1}) for ds in (train_ds, val_ds, test_ds))
    results["binary_labels"] = {"passed": labels_ok}

    # 4. No test leakage: re-verify scalers were fit on train only (re-derive and compare, same check as the pipeline's own).
    leakage_violations = pipeline.check_scalers_fit_train_only(channel_results)
    results["no_test_leakage_in_scalers"] = {"passed": len(leakage_violations) == 0, "violations": leakage_violations}

    # 5. Class distribution: every split must have both classes present (else ROC-AUC/PR-AUC are undefined).
    class_dist = {}
    dist_ok = True
    for split, ds in (("train", train_ds), ("validation", val_ds), ("test", test_ds)):
        has_both = len(np.unique(ds.labels)) == 2
        dist_ok = dist_ok and has_both
        class_dist[split] = {"normal": ds.normal_count, "anomaly": ds.anomaly_count, "anomaly_rate": ds.anomaly_rate, "has_both_classes": has_both}
    results["class_distribution"] = {"passed": dist_ok, "detail": class_dist}

    # 6. Model input dimension matches the data's feature dimension.
    input_dim_ok = train_ds.sequences.shape[-1] == CONFIG["input_dim"] == val_ds.sequences.shape[-1] == test_ds.sequences.shape[-1]
    results["model_input_dimension"] = {"passed": input_dim_ok, "configured_input_dim": CONFIG["input_dim"], "data_feature_dim": int(train_ds.sequences.shape[-1])}

    results["all_passed"] = all(v["passed"] for v in results.values())
    return results


def main() -> None:
    t_start = time.time()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Load ESA-ADB Option B pipeline data (centralized, all 58 channels pooled) ===")
    t0 = time.time()
    channel_results = pipeline.build_all_channels()
    load_time = time.time() - t0
    print(f"  Loaded + resampled + sequenced 58 channels in {load_time:.1f}s")

    train_ds = pool_all_channels(channel_results, "train")
    val_ds = pool_all_channels(channel_results, "validation")
    test_ds = pool_all_channels(channel_results, "test")
    print(f"  Centralized sequences — train: {train_ds.num_sequences:,} ({train_ds.anomaly_count:,} anomaly), "
          f"validation: {val_ds.num_sequences:,} ({val_ds.anomaly_count:,} anomaly), "
          f"test: {test_ds.num_sequences:,} ({test_ds.anomaly_count:,} anomaly)")

    print("\n=== Step 2: Sanity checks (STOP before training if any fail) ===")
    checks = sanity_checks(train_ds, val_ds, test_ds, channel_results)
    for name, result in checks.items():
        if name == "all_passed":
            continue
        print(f"  [{'PASS' if result['passed'] else 'FAIL'}] {name}")
    (EXPERIMENT_DIR / "sanity_checks.json").write_text(json.dumps(checks, indent=2, default=str))

    if not checks["all_passed"]:
        print("\nERROR: sanity checks failed. STOPPING before training.", file=sys.stderr)
        sys.exit(1)
    print("  All sanity checks passed.")

    print("\n=== Step 3: Per-client distribution (informational only, no rebalancing) ===")
    client_metadata = json.loads(CLIENT_METADATA_PATH.read_text())
    client_distribution = {}
    for client, info in client_metadata.items():
        client_distribution[client] = {
            "channels": info["channels"],
            "train": info["splits"]["train"],
            "validation": info["splits"]["validation"],
            "test": info["splits"]["test"],
        }
        s = info["splits"]
        print(f"  {client}: train={s['train']['num_sequences']:,} ({s['train']['anomaly']:,} anom) | "
              f"val={s['validation']['num_sequences']:,} ({s['validation']['anomaly']:,} anom) | "
              f"test={s['test']['num_sequences']:,} ({s['test']['anomaly']:,} anom)")
    (EXPERIMENT_DIR / "client_distribution.json").write_text(json.dumps(client_distribution, indent=2))

    print("\n=== Step 4: Train centralized baseline ===")
    config = json.loads(json.dumps(CONFIG))  # deep copy, mutated in-place by train_esa_adb_baseline (adds class_weighting)
    history = train_esa_adb_baseline(train_ds, val_ds, config, EXPERIMENT_DIR)
    print(f"  Training complete in {history.training_seconds:.1f}s. Best epoch: {history.best_epoch} "
          f"({'stopped early' if history.stopped_early else 'completed max epochs'})")

    print("\n=== Step 5: Threshold selection (VALIDATION ONLY) using the frozen best checkpoint ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(EXPERIMENT_DIR / "best_model.pt", weights_only=False)
    from src.models.temporal_gru_attention import TemporalGRUAttention

    model = TemporalGRUAttention(
        input_dim=config["input_dim"], projection_dim=config["projection_dim"], hidden_dim=config["hidden_dim"],
        num_gru_layers=config["num_gru_layers"], dropout=config["dropout"], output_dim=config["output_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loader = make_dataloader(val_ds, config["training"]["batch_size"], shuffle=False)
    val_prob, val_true = run_inference(model, val_loader, device)
    threshold_selection = select_threshold_by_f1(val_true, val_prob)
    threshold = threshold_selection["selected_threshold"]
    print(f"  Selected threshold: {threshold} (validation F1 at this threshold: {threshold_selection['selected_validation_f1']:.4f})")

    val_pred = (val_prob >= threshold).astype(int)
    val_metrics = compute_extended_metrics(val_true, val_pred, val_prob)
    (EXPERIMENT_DIR / "validation_metrics.json").write_text(json.dumps({
        "threshold": threshold, "metrics": val_metrics.to_dict(),
    }, indent=2))
    (EXPERIMENT_DIR / "threshold_selection.json").write_text(json.dumps(threshold_selection, indent=2))

    print("\n=== Step 6: Final test evaluation (ONCE, at the frozen threshold) ===")
    test_loader = make_dataloader(test_ds, config["training"]["batch_size"], shuffle=False)
    test_prob, test_true = run_inference(model, test_loader, device)
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = compute_extended_metrics(test_true, test_pred, test_prob)
    print(f"  Test accuracy={test_metrics.classification.accuracy:.4f} precision={test_metrics.classification.precision:.4f} "
          f"recall={test_metrics.classification.recall:.4f} f1={test_metrics.classification.f1:.4f} "
          f"roc_auc={test_metrics.classification.roc_auc:.4f} pr_auc={test_metrics.pr_auc:.4f}")

    test_curves = curve_data(test_true, test_prob)
    (EXPERIMENT_DIR / "test_metrics.json").write_text(json.dumps({
        "threshold": threshold, "metrics": test_metrics.to_dict(),
        "normal_count": int((test_true == 0).sum()), "anomaly_count": int((test_true == 1).sum()),
    }, indent=2))
    (EXPERIMENT_DIR / "roc_curve.json").write_text(json.dumps(test_curves["roc_curve"]))
    (EXPERIMENT_DIR / "pr_curve.json").write_text(json.dumps(test_curves["pr_curve"]))

    print("\n=== Step 7: Save reproducibility config ===")
    manifest_bytes = MANIFEST_PATH.read_bytes()
    import hashlib

    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    full_config = {
        "experiment": "esa_adb/baseline_01",
        "model_architecture": "TemporalGRUAttention (src/models/temporal_gru_attention.py, unmodified)",
        "config": config,
        "dataset_manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "dataset_manifest_sha256": manifest_hash,
        "preprocessing_reference": "src/satellite/esa_adb/preprocessing.py (StandardScaler per channel, train-split only)",
        "threshold_selection_method": threshold_selection["method"],
        "selected_threshold": threshold,
        "timing": {"data_load_seconds": load_time, "training_seconds": history.training_seconds},
    }
    (EXPERIMENT_DIR / "config.json").write_text(json.dumps(full_config, indent=2))

    total_time = time.time() - t_start
    (EXPERIMENT_DIR / "timing.json").write_text(json.dumps({
        "data_load_seconds": load_time, "training_seconds": history.training_seconds, "total_seconds": total_time,
    }, indent=2))

    print(f"\nAll artifacts written to {EXPERIMENT_DIR}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()

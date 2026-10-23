"""ESA-ADB Experiment 1A — multivariate (cross-channel) baseline.

Isolates the effect of input_dim: 1 -> 58 against Experiment 1
(baseline_01, single-channel). Everything else is held identical: same
Option B temporal split, same 58 channels, same sequence_length=50/
stride=25/"last" label strategy, same per-channel train-only
normalization, same pos_weight philosophy, same architecture family
(TemporalGRUAttention, unmodified, reused with input_dim=58), same
training hyperparameters (seed=42, Adam lr=0.001, batch=128, epochs=25,
patience=4).

Run from the repository root, after Mission 1 has been downloaded:

    python scripts/train_esa_adb_baseline_multivariate.py

Writes to experiments/esa_adb/baseline_02_multivariate/ — does not
touch baseline_01 or any other experiment/protected artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.satellite.esa_adb import loader  # noqa: E402
from src.satellite.esa_adb import multivariate as mv  # noqa: E402
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

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "esa_adb" / "baseline_02_multivariate"
BASELINE_01_DIR = REPO_ROOT / "experiments" / "esa_adb" / "baseline_01"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "esa_adb" / "manifest.json"

# NOTE: these are NOT the univariate baseline's pooled counts
# (121,510/61,364/61,364 — that total sums each of the 58 channels'
# OWN independent windowing, i.e. 58x more sequences than one aligned
# multivariate window sequence would produce). The correct multivariate
# expectation is exactly the univariate total divided by 58 (one
# sequence per aligned cross-channel time window, not one per channel):
# 121510/58 = 2095.0, 61364/58 = 1058.0 exactly. A first version of
# this check wrongly used the pooled 121,510/61,364/61,364 figures and
# was caught failing by sanity_checks() before any training ran —
# preserved here as the corrected, verified expectation.
EXPECTED_APPROX = {
    "train": {"num_sequences": 2095},
    "validation": {"num_sequences": 1058},
    "test": {"num_sequences": 1058},
}

CONFIG = {
    "input_dim": 58,
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


def sanity_checks(splits: dict, channel_columns: list[str], alignment_report: mv.AlignmentReport) -> dict:
    results = {}

    train_ds, val_ds, test_ds = splits["train"], splits["validation"], splits["test"]

    shapes_ok = all(ds.sequences.shape[1:] == (prep.SEQUENCE_LENGTH, 58) for ds in (train_ds, val_ds, test_ds))
    results["sequence_shape_N_50_58"] = {"passed": shapes_ok, "train_shape": list(train_ds.sequences.shape),
                                          "validation_shape": list(val_ds.sequences.shape), "test_shape": list(test_ds.sequences.shape)}

    results["timestamp_alignment"] = {
        "passed": True,  # informational — misalignment is EXPECTED and handled via intersection join, not a failure
        "detail": alignment_report.to_dict(),
        "resolution": "inner join on the intersection of all 58 channels' real timestamps; "
                       "no fabricated/interpolated values introduced",
    }

    all_58_present = len(channel_columns) == 58
    results["all_58_channels_present"] = {"passed": all_58_present, "count": len(channel_columns)}

    labels_ok = all(set(np.unique(ds.labels).tolist()).issubset({0, 1}) for ds in (train_ds, val_ds, test_ds))
    results["binary_labels"] = {"passed": labels_ok}

    counts_ok = True
    count_detail = {}
    for split, ds in (("train", train_ds), ("validation", val_ds), ("test", test_ds)):
        expected_n = EXPECTED_APPROX[split]["num_sequences"]
        # Multivariate sequencing loses at most 1 shared row (the dropped leading bin)
        # relative to the univariate pooled totals, so allow a tiny, documented tolerance
        # rather than requiring bit-for-bit equality with the univariate counts.
        matches = abs(ds.num_sequences - expected_n) <= 2
        counts_ok = counts_ok and matches
        count_detail[split] = {"expected_univariate_baseline": expected_n, "actual": ds.num_sequences, "within_tolerance": matches}
    results["counts_close_to_univariate_baseline"] = {"passed": counts_ok, "detail": count_detail}

    class_dist = {}
    dist_ok = True
    for split, ds in (("train", train_ds), ("validation", val_ds), ("test", test_ds)):
        has_both = len(np.unique(ds.labels)) == 2
        dist_ok = dist_ok and has_both
        class_dist[split] = {"normal": ds.normal_count, "anomaly": ds.anomaly_count, "anomaly_rate": ds.anomaly_rate, "has_both_classes": has_both}
    results["class_distribution"] = {"passed": dist_ok, "detail": class_dist}

    input_dim_ok = train_ds.sequences.shape[-1] == CONFIG["input_dim"] == val_ds.sequences.shape[-1] == test_ds.sequences.shape[-1]
    results["model_input_dimension"] = {"passed": input_dim_ok, "configured_input_dim": CONFIG["input_dim"], "data_feature_dim": int(train_ds.sequences.shape[-1])}

    results["all_passed"] = all(v["passed"] for v in results.values())
    return results


def main() -> None:
    t_start = time.time()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Load + resample all 58 channels, verify timestamp alignment ===")
    t0 = time.time()
    channel_frames = mv.load_all_resampled_channels()
    alignment_report = mv.verify_alignment(channel_frames)
    load_time = time.time() - t0
    print(f"  Loaded + resampled 58 channels in {load_time:.1f}s")
    print(f"  Common (intersection) row count: {alignment_report.common_row_count:,}")
    print(f"  Channels with missing timestamps: {len(alignment_report.misaligned_channels)} -> {list(alignment_report.misaligned_channels.keys())}")
    for c, detail in alignment_report.misaligned_channels.items():
        print(f"    {c}: missing {detail['num_missing_timestamps']} timestamp(s), e.g. {detail['missing_timestamps_sample']}")

    print("\n=== Step 2: Build aligned multivariate frame (inner join, real data only) ===")
    wide = mv.build_aligned_multivariate_frame(channel_frames)
    channel_columns = sorted([c for c in wide.columns if c != mv.JOINT_LABEL_COLUMN], key=lambda c: int(c.split("_")[1]))
    print(f"  Aligned frame: {len(wide):,} rows x {len(channel_columns)} channels + joint label")
    print(f"  Joint label anomaly rate (whole window): {wide[mv.JOINT_LABEL_COLUMN].mean():.4f}")

    print("\n=== Step 3: Split, fit train-only per-channel scalers, normalize ===")
    frame_splits = mv.split_multivariate_frame(wide)
    scalers = mv.fit_multivariate_scalers(frame_splits["train"], channel_columns)
    normalized_splits = {name: mv.transform_multivariate_frame(df, scalers, channel_columns) for name, df in frame_splits.items()}

    print("\n=== Step 4: Build multivariate sequences per split ===")
    sequence_splits = {name: mv.build_multivariate_sequences(df, channel_columns) for name, df in normalized_splits.items()}
    for name, ds in sequence_splits.items():
        print(f"  {name}: {ds.num_sequences:,} sequences, shape={ds.sequences.shape}, {ds.anomaly_count:,} anomaly ({100 * ds.anomaly_rate:.2f}%)")

    print("\n=== Step 5: Sanity checks (STOP before training if any fail) ===")
    checks = sanity_checks(sequence_splits, channel_columns, alignment_report)
    for name, result in checks.items():
        if name == "all_passed":
            continue
        print(f"  [{'PASS' if result['passed'] else 'FAIL'}] {name}")
    (EXPERIMENT_DIR / "sanity_checks.json").write_text(json.dumps(checks, indent=2, default=str))

    if not checks["all_passed"]:
        print("\nERROR: sanity checks failed. STOPPING before training.", file=sys.stderr)
        sys.exit(1)
    print("  All sanity checks passed.")

    # Also re-verify no-leakage: scalers reproducible from train split alone.
    leak_violations = []
    for c in channel_columns:
        recomputed = prep.fit_channel_scaler(c, frame_splits["train"][c].to_numpy(dtype="float64"))
        if not np.isclose(recomputed.mean, scalers[c].mean) or not np.isclose(recomputed.scale, scalers[c].scale):
            leak_violations.append(c)
    print(f"  [{'PASS' if not leak_violations else 'FAIL'}] scalers_fit_train_only_reverified ({len(leak_violations)} violations)")
    (EXPERIMENT_DIR / "scaler_leakage_check.json").write_text(json.dumps({"violations": leak_violations}, indent=2))
    if leak_violations:
        print("ERROR: scaler leakage detected. STOPPING.", file=sys.stderr)
        sys.exit(1)

    print("\n=== Step 6: Train multivariate baseline (input_dim=58) ===")
    config = json.loads(json.dumps(CONFIG))
    history = train_esa_adb_baseline(sequence_splits["train"], sequence_splits["validation"], config, EXPERIMENT_DIR)
    print(f"  Training complete in {history.training_seconds:.1f}s. Best epoch: {history.best_epoch} "
          f"({'stopped early' if history.stopped_early else 'completed max epochs'})")

    print("\n=== Step 7: Threshold selection (VALIDATION ONLY) using the frozen best checkpoint ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(EXPERIMENT_DIR / "best_model.pt", weights_only=False)
    from src.models.temporal_gru_attention import TemporalGRUAttention

    model = TemporalGRUAttention(
        input_dim=config["input_dim"], projection_dim=config["projection_dim"], hidden_dim=config["hidden_dim"],
        num_gru_layers=config["num_gru_layers"], dropout=config["dropout"], output_dim=config["output_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loader = make_dataloader(sequence_splits["validation"], config["training"]["batch_size"], shuffle=False)
    val_prob, val_true = run_inference(model, val_loader, device)
    threshold_selection = select_threshold_by_f1(val_true, val_prob)
    threshold = threshold_selection["selected_threshold"]
    print(f"  Selected threshold: {threshold} (validation F1 at this threshold: {threshold_selection['selected_validation_f1']:.4f})")

    val_pred = (val_prob >= threshold).astype(int)
    val_metrics = compute_extended_metrics(val_true, val_pred, val_prob)
    (EXPERIMENT_DIR / "validation_metrics.json").write_text(json.dumps({"threshold": threshold, "metrics": val_metrics.to_dict()}, indent=2))
    (EXPERIMENT_DIR / "threshold_selection.json").write_text(json.dumps(threshold_selection, indent=2))

    print("\n=== Step 8: Final test evaluation (ONCE, at the frozen threshold) ===")
    test_loader = make_dataloader(sequence_splits["test"], config["training"]["batch_size"], shuffle=False)
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

    print("\n=== Step 9: Comparison table vs baseline_01 (single-channel) ===")
    comparison = {"input_dimension": {"single_channel": 1, "multivariate": 58}}
    if (BASELINE_01_DIR / "validation_metrics.json").exists() and (BASELINE_01_DIR / "test_metrics.json").exists():
        b1_val = json.loads((BASELINE_01_DIR / "validation_metrics.json").read_text())["metrics"]
        b1_test = json.loads((BASELINE_01_DIR / "test_metrics.json").read_text())["metrics"]
        rows = [
            ("val_roc_auc", b1_val.get("roc_auc"), val_metrics.classification.roc_auc),
            ("val_pr_auc", b1_val.get("pr_auc"), val_metrics.pr_auc),
            ("val_f1", b1_val.get("f1"), val_metrics.classification.f1),
            ("test_roc_auc", b1_test.get("roc_auc"), test_metrics.classification.roc_auc),
            ("test_pr_auc", b1_test.get("pr_auc"), test_metrics.pr_auc),
            ("test_f1", b1_test.get("f1"), test_metrics.classification.f1),
            ("test_recall", b1_test.get("recall"), test_metrics.classification.recall),
            ("test_fpr", b1_test.get("false_positive_rate"), test_metrics.classification.false_positive_rate),
        ]
        for name, sc, mc in rows:
            comparison[name] = {"single_channel": sc, "multivariate": mc, "delta": (mc - sc) if (sc is not None and mc is not None) else None}
            print(f"  {name:15s} single={sc if sc is not None else 'N/A'}  multivariate={mc}")
    else:
        print("  baseline_01 metrics not found — comparison table skipped.")
    (EXPERIMENT_DIR / "comparison_vs_baseline_01.json").write_text(json.dumps(comparison, indent=2))

    print("\n=== Step 10: Save reproducibility config ===")
    manifest_hash = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    full_config = {
        "experiment": "esa_adb/baseline_02_multivariate",
        "model_architecture": "TemporalGRUAttention (src/models/temporal_gru_attention.py, unmodified, input_dim=58)",
        "config": config,
        "dataset_manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "dataset_manifest_sha256": manifest_hash,
        "alignment_resolution": "inner join on intersection of all 58 channels' real timestamps",
        "joint_label_rule": "OR across all 58 channels' own official per-timestamp labels; sequence label = joint label of final timestep ('last' strategy, unchanged)",
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

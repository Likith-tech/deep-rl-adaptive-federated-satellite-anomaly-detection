"""ESA-ADB rule-based adaptive Federated Learning (Step B of this
phase's plan) — Phase-8-style deterministic client scoring
(`src.federated.adaptive`), adapted to ESA-ADB's real 8-client Option A
([X;M], input_dim=116) representation via
`src.training.esa_adb_adaptive.run_esa_adb_adaptive_fl`.

Mirrors `scripts/train_esa_adb_fedavg_baseline.py`'s structure/config/
output format exactly (same data build, same sanity checks, same
threshold-selection/final-evaluation/fairness/comparison steps) so
results are directly comparable — the ONLY difference is Step 5, which
calls the rule-based aggregation trainer instead of plain FedAvg.

Uses the "combined" rule-weight config from Phase 8's
configs/adaptive_fl.yaml (performance=0.40, data=0.25, resource=0.15,
fairness=0.20) as the primary rule — same rationale as Phase 8: this is
the rule where NO single signal dominates, so it is the fairest
single-config choice for a direct FedAvg-vs-adaptive comparison here
(this phase does not budget a full rule-weight ablation grid for
ESA-ADB — that remains future work, documented in the experiment
README). ESA-ADB's own resource/fairness signals
(`src.satellite.esa_adb.adaptive_signals`) are real, structural signals,
not Phase 4's synthetic simulation — see that module's docstring.

Run from the repository root, after fedavg_baseline_01 has been run:

    python scripts/train_esa_adb_adaptive_rule_baseline.py

Writes to experiments/esa_adb/adaptive_rule_baseline_01/ — does not
touch baseline_01/, baseline_02_multivariate/, fedavg_baseline_01/, or
option_a_federated_design/.
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

from src.satellite.esa_adb import clients as clients_mod  # noqa: E402
from src.satellite.esa_adb import federated_representation as fr  # noqa: E402
from src.satellite.esa_adb import multivariate as mv  # noqa: E402
from src.satellite.esa_adb import pipeline as esa_pipeline  # noqa: E402
from src.satellite.esa_adb import preprocessing as prep  # noqa: E402
from src.satellite.esa_adb.adaptive_signals import build_esa_adb_fairness_signals, build_esa_adb_resource_signals  # noqa: E402
from src.training.esa_adb_adaptive import run_esa_adb_adaptive_fl  # noqa: E402
from src.training.esa_adb_federated import build_initial_state_dict  # noqa: E402
from src.training.esa_adb_trainer import (  # noqa: E402
    compute_extended_metrics,
    compute_pos_weight,
    curve_data,
    make_dataloader,
    run_inference,
    select_threshold_by_f1,
)

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "esa_adb" / "adaptive_rule_baseline_01"
BASELINE_01_DIR = REPO_ROOT / "experiments" / "esa_adb" / "baseline_01"
BASELINE_02_DIR = REPO_ROOT / "experiments" / "esa_adb" / "baseline_02_multivariate"
FEDAVG_DIR = REPO_ROOT / "experiments" / "esa_adb" / "fedavg_baseline_01"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "esa_adb" / "manifest.json"

MODEL_CONFIG = {"input_dim": 116, "projection_dim": 64, "hidden_dim": 64, "num_gru_layers": 1, "dropout": 0.3, "output_dim": 1}
TRAINING_CONFIG = {"batch_size": 128, "learning_rate": 0.001, "local_epochs": 1, "seed": 42}
NUM_ROUNDS = 10
SHARED_INIT_SEED = 42
RULE_WEIGHTS = {"performance": 0.40, "data": 0.25, "resource": 0.15, "fairness": 0.20}  # Phase 8 "combined" config, reused verbatim


def fairness_gap(per_client_metric: dict[str, float]) -> float:
    values = list(per_client_metric.values())
    return max(values) - min(values)


def main() -> None:
    t_start = time.time()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Build the aligned, normalized global 58-channel frame ===")
    channel_frames = mv.load_all_resampled_channels()
    aligned = mv.build_aligned_multivariate_frame(channel_frames)
    channel_columns = list(fr.GLOBAL_CHANNEL_ORDER)
    splits = mv.split_multivariate_frame(aligned)
    scalers = mv.fit_multivariate_scalers(splits["train"], channel_columns)
    normalized_splits = {name: mv.transform_multivariate_frame(df, scalers, channel_columns) for name, df in splits.items()}
    print(f"  rows: train={len(normalized_splits['train']):,} val={len(normalized_splits['validation']):,} test={len(normalized_splits['test']):,}")

    print("\n=== Step 2: Build per-client [X;M] sequences (train/validation/test) — real ownership only ===")
    client_train_ds, client_val_ds, client_test_ds = {}, {}, {}
    for client, owned in clients_mod.CLIENT_CHANNELS.items():
        client_train_ds[client] = fr.build_client_sequences(fr.build_masked_client_frame(normalized_splits["train"], owned))
        client_val_ds[client] = fr.build_client_sequences(fr.build_masked_client_frame(normalized_splits["validation"], owned))
        client_test_ds[client] = fr.build_client_sequences(fr.build_masked_client_frame(normalized_splits["test"], owned))
        print(f"  {client} ({len(owned)} ch): train={client_train_ds[client].num_sequences:,} "
              f"val={client_val_ds[client].num_sequences:,} test={client_test_ds[client].num_sequences:,}")

    pooled_val = esa_pipeline.pool_sequence_datasets(list(client_val_ds.values()), prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY)
    pooled_test = esa_pipeline.pool_sequence_datasets(list(client_test_ds.values()), prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY)
    print(f"  Pooled (per-client-shaped, APPROVED) validation: {pooled_val.num_sequences:,} seq, {pooled_val.anomaly_count:,} anomaly")
    print(f"  Pooled (per-client-shaped, same design) test:       {pooled_test.num_sequences:,} seq, {pooled_test.anomaly_count:,} anomaly")

    print("\n=== Step 3: Pre-training sanity / leakage checks (STOP before training if any fail) ===")
    checks = {}
    checks["validation_representation_matches_approved_design"] = {
        "passed": pooled_val.num_sequences == 8464 and pooled_val.anomaly_count == 120,
        "expected": {"num_sequences": 8464, "anomaly_count": 120},
        "actual": {"num_sequences": pooled_val.num_sequences, "anomaly_count": pooled_val.anomaly_count},
    }
    scaler_violations = []
    for c in channel_columns:
        recomputed = prep.fit_channel_scaler(c, splits["train"][c].to_numpy(dtype="float64"))
        if not np.isclose(recomputed.mean, scalers[c].mean) or not np.isclose(recomputed.scale, scalers[c].scale):
            scaler_violations.append(c)
    checks["scalers_fit_train_only"] = {"passed": not scaler_violations, "violations": scaler_violations}
    ownership_violations = []
    for client, owned in clients_mod.CLIENT_CHANNELS.items():
        owned_set = set(owned)
        masked = fr.build_masked_client_frame(normalized_splits["train"], owned)
        for c in fr.GLOBAL_CHANNEL_ORDER:
            if c not in owned_set and (not (masked[c] == 0.0).all() or not (masked[fr.mask_column(c)] == 0.0).all()):
                ownership_violations.append(f"{client}: unowned {c} leaked or mis-masked")
    checks["each_client_receives_only_its_own_channels"] = {"passed": not ownership_violations, "violations": ownership_violations}
    init_a = build_initial_state_dict(MODEL_CONFIG, SHARED_INIT_SEED)
    init_b = build_initial_state_dict(MODEL_CONFIG, SHARED_INIT_SEED)
    identical_init = all(torch.equal(init_a[k], init_b[k]) for k in init_a)
    checks["global_model_initialized_identically"] = {"passed": identical_init}
    checks["no_test_data_used_in_training"] = {"passed": True, "note": "structural: run_esa_adb_adaptive_fl only ever receives client_train_ds and validation datasets; test datasets are constructed but never passed to it"}
    checks["no_test_labels_used_for_threshold_selection"] = {"passed": True, "note": "structural: select_threshold_by_f1 is called on pooled_val only, before pooled_test is ever touched"}
    checks["rule_weights_sum_to_one"] = {"passed": abs(sum(RULE_WEIGHTS.values()) - 1.0) < 1e-9, "rule_weights": RULE_WEIGHTS}

    all_passed = all(v["passed"] for v in checks.values())
    (EXPERIMENT_DIR / "sanity_checks.json").write_text(json.dumps(checks, indent=2, default=str))
    for name, v in checks.items():
        print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {name}")
    if not all_passed:
        print("\nERROR: sanity checks failed. STOPPING before training.", file=sys.stderr)
        sys.exit(1)
    print("  All sanity checks passed.")

    print("\n=== Step 4: Static resource/fairness signals (real, non-simulated — see adaptive_signals.py) ===")
    resource_signals = build_esa_adb_resource_signals(list(client_train_ds.keys()))
    fairness_signals = build_esa_adb_fairness_signals({c: ds.labels for c, ds in client_train_ds.items()})
    for c in client_train_ds:
        print(f"  {c}: resource_raw={resource_signals[c]:.4f} fairness_entropy_bits={fairness_signals[c]:.4f}")

    print(f"\n=== Step 5: Run rule-based adaptive FL — {NUM_ROUNDS} rounds, all 8 clients every round, rule={RULE_WEIGHTS} ===")
    result = run_esa_adb_adaptive_fl(
        client_train_datasets=client_train_ds, client_validation_datasets=client_val_ds, pooled_validation_dataset=pooled_val,
        model_config=MODEL_CONFIG, training_config=TRAINING_CONFIG, num_rounds=NUM_ROUNDS,
        shared_init_seed=SHARED_INIT_SEED, checkpoint_dir=EXPERIMENT_DIR, rule_weights=RULE_WEIGHTS,
    )
    print(f"  Best round: {result.best_round} (val_loss={result.best_val_loss:.4f}). Total FL time: {result.total_seconds:.1f}s")

    print("\n=== Step 6: Threshold selection (VALIDATION ONLY, frozen best-round global model) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_checkpoint = torch.load(EXPERIMENT_DIR / "best_global_model.pt", weights_only=False)
    from src.models.temporal_gru_attention import TemporalGRUAttention

    model = TemporalGRUAttention(**{k: MODEL_CONFIG[k] for k in ("input_dim", "projection_dim", "hidden_dim", "num_gru_layers", "dropout", "output_dim")}).to(device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    val_loader = make_dataloader(pooled_val, TRAINING_CONFIG["batch_size"], shuffle=False)
    val_prob, val_true = run_inference(model, val_loader, device)
    threshold_selection = select_threshold_by_f1(val_true, val_prob)
    threshold = threshold_selection["selected_threshold"]
    print(f"  Selected threshold: {threshold} (pooled validation F1 at this threshold: {threshold_selection['selected_validation_f1']:.4f})")

    val_pred = (val_prob >= threshold).astype(int)
    val_metrics = compute_extended_metrics(val_true, val_pred, val_prob)
    (EXPERIMENT_DIR / "validation_metrics.json").write_text(json.dumps({"threshold": threshold, "metrics": val_metrics.to_dict()}, indent=2))
    (EXPERIMENT_DIR / "threshold_selection.json").write_text(json.dumps(threshold_selection, indent=2))

    print("\n=== Step 7: Final test evaluation (ONCE, frozen threshold) — pooled + per-client ===")
    test_loader = make_dataloader(pooled_test, TRAINING_CONFIG["batch_size"], shuffle=False)
    test_prob, test_true = run_inference(model, test_loader, device)
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = compute_extended_metrics(test_true, test_pred, test_prob)
    print(f"  Pooled test: accuracy={test_metrics.classification.accuracy:.4f} precision={test_metrics.classification.precision:.4f} "
          f"recall={test_metrics.classification.recall:.4f} f1={test_metrics.classification.f1:.4f} "
          f"roc_auc={test_metrics.classification.roc_auc:.4f} pr_auc={test_metrics.pr_auc:.4f}")

    test_curves = curve_data(test_true, test_prob)
    (EXPERIMENT_DIR / "test_metrics.json").write_text(json.dumps({
        "threshold": threshold, "metrics": test_metrics.to_dict(),
        "normal_count": int((test_true == 0).sum()), "anomaly_count": int((test_true == 1).sum()),
    }, indent=2))
    (EXPERIMENT_DIR / "roc_curve.json").write_text(json.dumps(test_curves["roc_curve"]))
    (EXPERIMENT_DIR / "pr_curve.json").write_text(json.dumps(test_curves["pr_curve"]))

    per_client_test = {}
    for client, ds in client_test_ds.items():
        loader = make_dataloader(ds, TRAINING_CONFIG["batch_size"], shuffle=False)
        prob, true = run_inference(model, loader, device)
        pred = (prob >= threshold).astype(int)
        m = compute_extended_metrics(true, pred, prob)
        per_client_test[client] = {
            "num_sequences": ds.num_sequences, "anomaly_count": ds.anomaly_count,
            "accuracy": m.classification.accuracy, "precision": m.classification.precision, "recall": m.classification.recall,
            "f1": m.classification.f1, "roc_auc": m.classification.roc_auc, "pr_auc": m.pr_auc,
            "false_positive_rate": m.classification.false_positive_rate, "confusion_matrix": m.classification.confusion_matrix,
        }
        print(f"  {client}: f1={m.classification.f1:.4f} recall={m.classification.recall:.4f} "
              f"precision={m.classification.precision:.4f} roc_auc={m.classification.roc_auc}")
    (EXPERIMENT_DIR / "per_client_test_metrics.json").write_text(json.dumps(per_client_test, indent=2))

    print("\n=== Step 8: Fairness (test-time, same max-min F1 formula as Phase 8/9B) ===")
    test_fairness_gap_f1 = fairness_gap({c: m["f1"] for c, m in per_client_test.items()})
    final_round_val_fairness = result.round_history[-1].fairness_gap_f1
    best_round_val_fairness = result.round_history[result.best_round - 1].fairness_gap_f1
    print(f"  Test fairness gap (max-min client F1): {test_fairness_gap_f1:.4f}")
    print(f"  Best-round validation fairness gap: {best_round_val_fairness:.4f}")
    (EXPERIMENT_DIR / "fairness_summary.json").write_text(json.dumps({
        "test_fairness_gap_f1": test_fairness_gap_f1,
        "best_round_validation_fairness_gap_f1": best_round_val_fairness,
        "final_round_validation_fairness_gap_f1": final_round_val_fairness,
        "per_client_test_f1": {c: m["f1"] for c, m in per_client_test.items()},
    }, indent=2))

    print("\n=== Step 9: Comparison with centralized/FedAvg baselines ===")
    comparison = {"adaptive_rule_baseline_01": {"roc_auc": test_metrics.classification.roc_auc, "pr_auc": test_metrics.pr_auc,
                                                 "f1": test_metrics.classification.f1, "recall": test_metrics.classification.recall,
                                                 "fpr": test_metrics.classification.false_positive_rate}}
    for name, d in (("baseline_01_single_channel", BASELINE_01_DIR), ("baseline_02_multivariate", BASELINE_02_DIR), ("fedavg_baseline_01", FEDAVG_DIR)):
        p = d / "test_metrics.json"
        if p.exists():
            m = json.loads(p.read_text())["metrics"]
            comparison[name] = {"roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"), "f1": m.get("f1"),
                                 "recall": m.get("recall"), "fpr": m.get("false_positive_rate")}
            print(f"  {name}: roc_auc={m.get('roc_auc')} pr_auc={m.get('pr_auc')} f1={m.get('f1')} recall={m.get('recall')} fpr={m.get('false_positive_rate')}")
    print(f"  adaptive_rule_baseline_01: roc_auc={test_metrics.classification.roc_auc:.4f} pr_auc={test_metrics.pr_auc:.4f} "
          f"f1={test_metrics.classification.f1:.4f} recall={test_metrics.classification.recall:.4f} fpr={test_metrics.classification.false_positive_rate:.4f}")
    (EXPERIMENT_DIR / "comparison_with_centralized_baselines.json").write_text(json.dumps(comparison, indent=2))

    print("\n=== Step 10: Save reproducibility config/manifest ===")
    manifest_hash = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    full_config = {
        "experiment": "esa_adb/adaptive_rule_baseline_01",
        "model_architecture": "TemporalGRUAttention (unmodified), input_dim=116",
        "model_config": MODEL_CONFIG,
        "training_config": TRAINING_CONFIG,
        "num_rounds": NUM_ROUNDS,
        "shared_init_seed": SHARED_INIT_SEED,
        "rule_weights": RULE_WEIGHTS,
        "rule_weights_source": "Phase 8 'combined' config (configs/adaptive_fl.yaml), reused verbatim, not re-tuned for ESA-ADB",
        "resource_fairness_signal_source": "src.satellite.esa_adb.adaptive_signals — real structural/statistical signals, not Phase 4's synthetic satellite simulation",
        "client_participation": "all 8 real clients every round (no sampling, no dropout); aggregation WEIGHTING via Phase 8 rule, not selection",
        "aggregation": "Phase-8-style deterministic rule-based weighted average (src.federated.adaptive + FederatedServer.aggregate_with_weights, unmodified)",
        "class_imbalance_handling": "BCEWithLogitsLoss(pos_weight), computed PER CLIENT from that client's own train labels only",
        "dataset_manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "dataset_manifest_sha256": manifest_hash,
        "threshold_selection_method": threshold_selection["method"],
        "selected_threshold": threshold,
        "best_round": result.best_round,
        "best_round_val_loss": result.best_val_loss,
        "timing": {"adaptive_fl_total_seconds": result.total_seconds, "script_total_seconds": time.time() - t_start},
    }
    (EXPERIMENT_DIR / "config.json").write_text(json.dumps(full_config, indent=2))

    print(f"\nAll artifacts written to {EXPERIMENT_DIR}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()

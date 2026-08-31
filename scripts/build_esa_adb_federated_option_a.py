"""ESA-ADB Option A federated input representation — implementation +
validation ONLY. No FedAvg training, no DQN, no centralized training,
no hyperparameter tuning, no test evaluation.

Builds the real [X; M] (input_dim=116) per-client data representation
for all 8 real clients, verifies model shape/FedAvg compatibility with
a synthetic (untrained) aggregation call, and documents the global
validation representation decision — writes everything needed for
review to experiments/esa_adb/option_a_federated_design/ (a new,
isolated directory; does not touch baseline_01/ or
baseline_02_multivariate/).

Run from the repository root, after Mission 1 has been downloaded:

    python scripts/build_esa_adb_federated_option_a.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.satellite.esa_adb import clients as clients_mod  # noqa: E402
from src.satellite.esa_adb import federated_representation as fr  # noqa: E402
from src.satellite.esa_adb import multivariate as mv  # noqa: E402
from src.satellite.esa_adb import preprocessing as prep  # noqa: E402

EXPERIMENT_DIR = REPO_ROOT / "experiments" / "esa_adb" / "option_a_federated_design"

MODEL_CONFIG = {
    "input_dim": 116,
    "projection_dim": 64,
    "hidden_dim": 64,
    "num_gru_layers": 1,
    "dropout": 0.3,
    "output_dim": 1,
}


def main() -> None:
    t0 = time.time()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Load + align + normalize the global 58-channel frame (reused from Experiment 1A) ===")
    channel_frames = mv.load_all_resampled_channels()
    alignment_report = mv.verify_alignment(channel_frames)
    aligned = mv.build_aligned_multivariate_frame(channel_frames)
    channel_columns = list(fr.GLOBAL_CHANNEL_ORDER)
    print(f"  Aligned frame: {len(aligned):,} rows x 58 channels. Misaligned channels: {list(alignment_report.misaligned_channels.keys())}")

    splits = mv.split_multivariate_frame(aligned)
    scalers = mv.fit_multivariate_scalers(splits["train"], channel_columns)
    normalized_splits = {name: mv.transform_multivariate_frame(df, scalers, channel_columns) for name, df in splits.items()}
    print(f"  Split sizes (rows): train={len(normalized_splits['train']):,} val={len(normalized_splits['validation']):,} test={len(normalized_splits['test']):,}")

    print("\n=== Step 2: Verify real client ownership against clients.py (not assumed) ===")
    client_counts = {c: len(chans) for c, chans in clients_mod.CLIENT_CHANNELS.items()}
    for c, n in client_counts.items():
        print(f"  {c}: {n} channels -> {clients_mod.CLIENT_SUBSYSTEM_GROUPS[c]}")
    assert sum(client_counts.values()) == 58

    print("\n=== Step 3: Build per-client [X; M] masked frames + sequences (train/validation/test) ===")
    per_client_sequences: dict[str, dict[str, object]] = {}
    for client, owned in clients_mod.CLIENT_CHANNELS.items():
        per_client_sequences[client] = {}
        for split_name, df in normalized_splits.items():
            masked = fr.build_masked_client_frame(df, owned)
            seq = fr.build_client_sequences(masked)
            per_client_sequences[client][split_name] = seq
        s = per_client_sequences[client]
        print(f"  {client} ({client_counts[client]} ch): train={s['train'].num_sequences:,} ({s['train'].anomaly_count:,} anom) | "
              f"val={s['validation'].num_sequences:,} ({s['validation'].anomaly_count:,} anom) | "
              f"test={s['test'].num_sequences:,} ({s['test'].anomaly_count:,} anom)")

    print("\n=== Step 4: Global validation representation — decision + comparison ===")
    client_val_frames = {c: fr.build_masked_client_frame(normalized_splits["validation"], owned) for c, owned in clients_mod.CLIENT_CHANNELS.items()}
    pooled_val_approved = fr.build_pooled_validation_sequences(client_val_frames)
    pooled_val_hypothetical = fr.build_global_validation_all_available_HYPOTHETICAL(normalized_splits["validation"])
    print(f"  APPROVED (pooled per-client-shaped): {pooled_val_approved.num_sequences:,} sequences, "
          f"{pooled_val_approved.anomaly_count:,} anomaly ({100 * pooled_val_approved.anomaly_rate:.2f}%)")
    print(f"  REJECTED (all-58-available hypothetical, for comparison only): {pooled_val_hypothetical.num_sequences:,} sequences, "
          f"{pooled_val_hypothetical.anomaly_count:,} anomaly ({100 * pooled_val_hypothetical.anomaly_rate:.2f}%)")

    print("\n=== Step 5: Model shape / FedAvg compatibility (untrained, synthetic) ===")
    models = fr.instantiate_client_models(MODEL_CONFIG, num_clients=8)
    compat_report = fr.verify_state_dict_compatibility(models)
    print(f"  State_dict compatibility: {'PASS' if compat_report['all_identical'] else 'FAIL'} "
          f"({compat_report['num_pairs_checked']} pairs checked, parameter count={compat_report['parameter_count_per_model'][0]:,})")

    real_sample_counts = [per_client_sequences[c]["train"].num_sequences for c in clients_mod.CLIENT_CHANNELS]
    fedavg_report = fr.verify_fedavg_compatibility(models, real_sample_counts)
    print(f"  FedAvg synthetic aggregation: {'PASS' if fedavg_report['succeeded'] else 'FAIL'} (error={fedavg_report['error']})")

    print("\n=== Step 6: Leakage / integrity checks ===")
    leakage_violations = []
    for client, owned in clients_mod.CLIENT_CHANNELS.items():
        owned_set = set(owned)
        masked = fr.build_masked_client_frame(normalized_splits["train"], owned)
        for c in fr.GLOBAL_CHANNEL_ORDER:
            if c not in owned_set:
                if not (masked[c] == 0.0).all() or not (masked[fr.mask_column(c)] == 0.0).all():
                    leakage_violations.append(f"{client}: unowned channel {c} not correctly zero/masked")
    print(f"  [{'PASS' if not leakage_violations else 'FAIL'}] no_cross_client_telemetry_leakage ({len(leakage_violations)} violations)")

    scaler_violations = []
    for c in channel_columns:
        recomputed = prep.fit_channel_scaler(c, splits["train"][c].to_numpy(dtype="float64"))
        if not np.isclose(recomputed.mean, scalers[c].mean) or not np.isclose(recomputed.scale, scalers[c].scale):
            scaler_violations.append(c)
    print(f"  [{'PASS' if not scaler_violations else 'FAIL'}] scalers_fit_train_only ({len(scaler_violations)} violations)")

    all_checks_passed = compat_report["all_identical"] and fedavg_report["succeeded"] and not leakage_violations and not scaler_violations

    print("\n=== Step 7: Save manifest and reports ===")
    manifest = {
        "design": "Option A: global 58-channel input + explicit 58-channel availability mask, input_dim=116",
        "approved_by": "user, this turn",
        "global_channel_order": list(fr.GLOBAL_CHANNEL_ORDER),
        "feature_columns_116": fr.feature_columns(),
        "real_client_channel_ownership": {c: chans for c, chans in clients_mod.CLIENT_CHANNELS.items()},
        "real_client_channel_counts": client_counts,
        "client_counts_note": "Real counts verified directly from clients.py: "
                               "client_1=6, client_2=7, client_3=3, client_4=8, client_5=9, client_6=9, client_7=8, client_8=8. "
                               "This differs from the per-client-label assignment stated in the originating instruction "
                               "(which used the same {3,6,7,8,8,8,9,9} multiset but attached to different client labels) — "
                               "the real clients.py mapping was used, per the explicit 'verify, do not assume' instruction.",
        "mask_semantics": {
            "owned_channel": "X = real normalized telemetry, M = 1",
            "unowned_channel": "X = 0.0 (fixed placeholder, never fabricated/interpolated/borrowed), M = 0",
            "rejected_alternative": "gate-only X*M — rejected because a real normalized-zero reading from an owned "
                                     "channel would be indistinguishable from an unowned channel under that scheme",
        },
        "model_architecture": "TemporalGRUAttention (src/models/temporal_gru_attention.py, unmodified)",
        "model_config": MODEL_CONFIG,
        "input_tensor_shape": [None, prep.SEQUENCE_LENGTH, 116],
        "state_dict_compatibility": compat_report,
        "fedavg_compatibility": fedavg_report,
        "global_validation_representation": {
            "decision": "pooled per-client-shaped sequences (each client's own real ownership mask)",
            "approved_num_sequences": pooled_val_approved.num_sequences,
            "approved_anomaly_count": pooled_val_approved.anomaly_count,
            "rejected_alternative": "all-58-channels-available (M=all-1s) hypothetical",
            "rejected_num_sequences": pooled_val_hypothetical.num_sequences,
            "rejected_anomaly_count": pooled_val_hypothetical.anomaly_count,
            "reasoning": "The all-available variant is never a scenario any real client or real deployment of the "
                         "federated model would encounter — it is exactly what Experiment 1A's centralized baseline "
                         "already measures. Using it as the DQN's validation signal would silently re-test the "
                         "centralized ceiling instead of genuinely evaluating the federated model.",
        },
        "leakage_checks": {"no_cross_client_telemetry_leakage": {"passed": not leakage_violations, "violations": leakage_violations},
                            "scalers_fit_train_only": {"passed": not scaler_violations, "violations": scaler_violations}},
        "all_checks_passed": all_checks_passed,
        "alignment_report": alignment_report.to_dict(),
    }
    (EXPERIMENT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    per_client_summary = {}
    for client in clients_mod.CLIENT_CHANNELS:
        entry = {"channels": clients_mod.CLIENT_CHANNELS[client], "num_channels": client_counts[client]}
        for split, seq in per_client_sequences[client].items():
            entry[split] = {"num_sequences": seq.num_sequences, "anomaly": seq.anomaly_count, "normal": seq.normal_count}
        per_client_summary[client] = entry
    (EXPERIMENT_DIR / "per_client_sequence_summary.json").write_text(json.dumps(per_client_summary, indent=2))

    print(f"\nAll checks passed: {all_checks_passed}")
    print(f"Manifest written to {EXPERIMENT_DIR}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

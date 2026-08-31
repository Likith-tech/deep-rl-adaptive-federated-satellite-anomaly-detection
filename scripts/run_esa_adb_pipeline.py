"""Run the ESA-ADB Mission 1 data-preparation pipeline end-to-end on the
real, downloaded archive and write the reproducible manifest artifacts.

DATA PREPARATION ONLY — loads, labels, preprocesses, sequences, and
partitions the approved 2000 window into the 8-client structure. Does
NOT train any model.

Run from the repository root, after Mission 1 has been downloaded and
extracted:

    python scripts/run_esa_adb_pipeline.py

Writes:
    data/processed/esa_adb/{manifest.json,client_metadata.json,preprocessing.json}
    results/models/esa_adb_preprocessing/{channel_scalers.joblib,channel_scalers_summary.json}
    results/reports/esa_adb_pipeline_inspection.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.satellite.esa_adb import clients as clients_mod  # noqa: E402
from src.satellite.esa_adb import loader  # noqa: E402
from src.satellite.esa_adb import pipeline  # noqa: E402
from src.satellite.esa_adb import preprocessing as prep  # noqa: E402
from src.satellite.esa_adb.labeling import group_label_rows_by_channel  # noqa: E402

SEED = 42
REPORT_PATH = REPO_ROOT / "results" / "reports" / "esa_adb_pipeline_inspection.md"


def main() -> None:
    t0 = time.time()
    print("Loading metadata and building per-channel pipeline for all 58 Target=YES channels...")
    channel_results = pipeline.build_all_channels()
    print(f"  Done in {time.time() - t0:.1f}s — {len(channel_results)} channels processed.")

    label_rows = loader.load_label_rows()
    label_rows_by_channel = group_label_rows_by_channel(label_rows)

    print("Building 8-client pooled sequence datasets...")
    client_datasets = pipeline.build_client_datasets(channel_results)

    print("Running leakage/integrity checks...")
    checks = pipeline.run_all_checks(channel_results, client_datasets, label_rows_by_channel)
    for name, violations in checks.items():
        status = "PASS" if not violations else f"FAIL ({len(violations)} violations)"
        print(f"  [{status}] {name}")
        for v in violations[:5]:
            print(f"      - {v}")

    any_failed = any(v for v in checks.values())
    if any_failed:
        print("\nERROR: one or more leakage/integrity checks failed. Manifests NOT written.", file=sys.stderr)
        sys.exit(1)

    print("Saving per-channel scalers...")
    prep.save_scalers({c: r.scaler for c, r in channel_results.items()})

    print("Writing manifests...")
    manifest = pipeline.build_manifest(channel_results, client_datasets, checks, seed=SEED)
    client_metadata = pipeline.build_client_metadata(channel_results, client_datasets)
    preprocessing_manifest = pipeline.build_preprocessing_manifest(channel_results)
    pipeline.write_manifests(manifest, client_metadata, preprocessing_manifest)

    print("Writing lightweight inspection artifacts (results/reports/esa_adb/)...")
    pipeline.write_inspection_artifacts(channel_results, client_datasets)

    # Lightweight inspection report (counts/distributions only, no raw data, no large plots).
    lines = [
        "# ESA-ADB Mission 1 Pipeline — Inspection Summary",
        "",
        "**DATA PREPARATION ONLY. No model trained.**",
        "",
        f"Window: {loader.WINDOW_START} to {loader.WINDOW_END}. Channels: {len(channel_results)} (Target=YES).",
        f"Resample freq: {prep.RESAMPLE_FREQ}. Sequence length: {prep.SEQUENCE_LENGTH}. Stride: {prep.STRIDE}. Label strategy: {prep.LABEL_STRATEGY}.",
        "",
        "## Per-channel resampled row counts and label balance (whole window, all splits pooled)",
        "",
        "| Channel | Subsystem | Group | Resampled rows | Constant? |",
        "|---|---|---:|---:|---|",
    ]
    for c in sorted(channel_results, key=lambda c: int(c.split("_")[1])):
        r = channel_results[c]
        lines.append(f"| {c} | {r.subsystem} | {r.group} | {r.resampled_rows:,} | {r.scaler.is_constant} |")

    lines += ["", "## Per-split sequence counts and anomaly rate (all channels combined)", "", "| Split | Total sequences | Normal | Anomaly | Anomaly rate |", "|---|---:|---:|---:|---:|"]
    for split in pipeline.SPLITS:
        all_seq = pipeline.pool_sequence_datasets(
            [channel_results[c].sequences[split] for c in channel_results], prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY
        )
        lines.append(f"| {split} | {all_seq.num_sequences:,} | {all_seq.normal_count:,} | {all_seq.anomaly_count:,} | {all_seq.anomaly_rate:.4f} |")

    lines += ["", "## Per-client sequence counts", "", "| Client | Channels | Train seq | Val seq | Test seq | Train anomaly rate | Test anomaly rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for client in clients_mod.CLIENT_CHANNELS:
        d = client_datasets[client]
        lines.append(
            f"| {client} | {len(clients_mod.CLIENT_CHANNELS[client])} | {d['train'].num_sequences:,} | "
            f"{d['validation'].num_sequences:,} | {d['test'].num_sequences:,} | {d['train'].anomaly_rate:.4f} | {d['test'].anomaly_rate:.4f} |"
        )

    lines += ["", "## Leakage/integrity checks", ""]
    for name, violations in checks.items():
        lines.append(f"- **{name}**: {'PASS' if not violations else f'FAIL ({len(violations)})'}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

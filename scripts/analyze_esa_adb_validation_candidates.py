"""ESA-ADB validation-split candidate analysis (Track C).

DESIGN/ANALYSIS ONLY — no training, no pipeline artifacts overwritten.

The current, already-implemented split (train Jan-Aug, validation Sep,
test Oct-Dec) produces ZERO anomaly sequences in validation, even
though 8 real anomalous resampled records exist in that month (all
`Length=Point`, single-channel, subsystem_3 events that never happen to
land on the LAST timestep of a stride-aligned window under the "last"
sequence-labeling rule). This script evaluates alternative validation
windows using the REAL pipeline machinery (loader, labeling,
preprocessing) already built and tested in src/satellite/esa_adb/ — it
computes ACTUAL sequence-level anomaly counts per candidate, not
estimates from event metadata alone, because event-level richness does
not guarantee sequence-level positives (as the current split already
demonstrates).

TEST is held fixed at 2000-10-01 to 2000-12-31 in every candidate (per
the "test set must remain untouched" requirement) — only the
train/validation boundary moves. Each channel's raw window is loaded
and resampled ONCE and reused across all candidates (the resampling
step does not depend on where the train/validation boundary falls), so
this stays cheap despite testing multiple candidates.

Run from the repository root, after Mission 1 has been downloaded:

    python scripts/analyze_esa_adb_validation_candidates.py

Writes:
    results/reports/esa_adb_validation_split_candidates.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.satellite.esa_adb import clients as clients_mod  # noqa: E402
from src.satellite.esa_adb import loader  # noqa: E402
from src.satellite.esa_adb import pipeline  # noqa: E402
from src.satellite.esa_adb import preprocessing as prep  # noqa: E402
from src.satellite.esa_adb.labeling import channel_anomaly_intervals, group_label_rows_by_channel  # noqa: E402

REPORT_PATH = REPO_ROOT / "results" / "reports" / "esa_adb_validation_split_candidates.md"

TEST_START = loader.TEST_START  # fixed across all candidates: 2000-10-01
TEST_END = loader.TEST_END      # fixed across all candidates: 2000-12-31 23:59:59.999999

CANDIDATES = {
    "A": {
        "description": "Train Jan-Jul, Validation Aug-Sep, Test Oct-Dec",
        "train_end": pd.Timestamp("2000-07-31 23:59:59.999999"),
        "val_start": pd.Timestamp("2000-08-01 00:00:00"),
        "val_end": pd.Timestamp("2000-09-30 23:59:59.999999"),
    },
    "B": {
        "description": "Train Jan-Jun, Validation Jul-Sep, Test Oct-Dec",
        "train_end": pd.Timestamp("2000-06-30 23:59:59.999999"),
        "val_start": pd.Timestamp("2000-07-01 00:00:00"),
        "val_end": pd.Timestamp("2000-09-30 23:59:59.999999"),
    },
    "C_current": {
        "description": "Train Jan-Aug, Validation Sep (CURRENT, already implemented)",
        "train_end": pd.Timestamp("2000-08-31 23:59:59.999999"),
        "val_start": pd.Timestamp("2000-09-01 00:00:00"),
        "val_end": pd.Timestamp("2000-09-30 23:59:59.999999"),
    },
    "D": {
        "description": "Train Jan-May, Validation Jun-Sep, Test Oct-Dec",
        "train_end": pd.Timestamp("2000-05-31 23:59:59.999999"),
        "val_start": pd.Timestamp("2000-06-01 00:00:00"),
        "val_end": pd.Timestamp("2000-09-30 23:59:59.999999"),
    },
}
TRAIN_START = loader.TRAIN_START  # fixed: 2000-01-01


def load_and_resample_all_channels() -> dict[str, dict]:
    """Load + resample every one of the 58 channels ONCE (whole-window,
    not split-dependent), returning per-channel {resampled_df, meta}."""
    meta_by_channel = loader.load_channel_metadata()
    label_rows_by_channel = group_label_rows_by_channel(loader.load_label_rows())

    out = {}
    for channel in loader.target_yes_channels():
        raw = loader.load_channel_window(channel)
        intervals = channel_anomaly_intervals(channel, label_rows_by_channel.get(channel, []))
        resampled = prep.resample_channel(raw, intervals)
        out[channel] = {"resampled": resampled, "meta": meta_by_channel[channel]}
    return out


def build_candidate_result(channel_data: dict[str, dict], train_end: pd.Timestamp, val_start: pd.Timestamp, val_end: pd.Timestamp) -> dict:
    """Given cached resampled per-channel data, slice by this
    candidate's boundaries, fit train-only scalers, build sequences,
    and pool per-client + overall results."""
    bounds = {
        "train": (TRAIN_START, train_end),
        "validation": (val_start, val_end),
        "test": (TEST_START, TEST_END),
    }

    per_channel_sequences: dict[str, dict] = {}
    for channel, data in channel_data.items():
        df = data["resampled"]
        split_frames = {name: df[(df.index >= s) & (df.index <= e)] for name, (s, e) in bounds.items()}
        train_values = split_frames["train"][prep.VALUE_COLUMN].to_numpy(dtype="float64")
        scaler = prep.fit_channel_scaler(channel, train_values)

        seqs = {}
        for split, frame in split_frames.items():
            seqs[split] = prep.build_channel_sequences(frame, scaler) if len(frame) >= prep.SEQUENCE_LENGTH else _empty_seq()
        per_channel_sequences[channel] = seqs

    overall = {
        split: pipeline.pool_sequence_datasets(
            [per_channel_sequences[c][split] for c in per_channel_sequences], prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY
        )
        for split in ("train", "validation", "test")
    }

    per_client = {}
    for client, chans in clients_mod.CLIENT_CHANNELS.items():
        per_client[client] = {
            split: pipeline.pool_sequence_datasets(
                [per_channel_sequences[c][split] for c in chans], prep.SEQUENCE_LENGTH, prep.STRIDE, prep.LABEL_STRATEGY
            )
            for split in ("train", "validation", "test")
        }

    return {"bounds": bounds, "overall": overall, "per_client": per_client}


def _empty_seq():
    import numpy as np

    from src.preprocessing.sequences import SequenceDataset

    return SequenceDataset(
        sequences=np.empty((0, prep.SEQUENCE_LENGTH, 1), dtype="float32"),
        labels=np.empty((0,), dtype="int64"), record_labels=[],
        sequence_length=prep.SEQUENCE_LENGTH, stride=prep.STRIDE, label_strategy=prep.LABEL_STRATEGY,
    )


def check_no_straddle(channel_data: dict[str, dict], label_rows_by_channel: dict, train_end: pd.Timestamp, val_start: pd.Timestamp, val_end: pd.Timestamp) -> list[str]:
    violations = []
    boundaries = [("train/validation", val_start), ("validation/test", TEST_START)]
    for channel in channel_data:
        intervals = channel_anomaly_intervals(channel, label_rows_by_channel.get(channel, []))
        for start, end in intervals:
            for name, boundary in boundaries:
                if start < boundary <= end:
                    violations.append(f"{channel}: interval [{start}, {end}] straddles {name} ({boundary})")
    return violations


def main() -> None:
    t0 = time.time()
    print("Loading + resampling all 58 channels once (shared across all candidates)...")
    channel_data = load_and_resample_all_channels()
    label_rows_by_channel = group_label_rows_by_channel(loader.load_label_rows())
    print(f"  Done in {time.time() - t0:.1f}s")

    lines = [
        "# ESA-ADB Validation-Split Candidate Analysis",
        "",
        "**DESIGN/ANALYSIS ONLY — no training, no production manifest overwritten.**",
        "",
        "TEST is fixed at 2000-10-01 to 2000-12-31 in every candidate. Only the",
        "train/validation boundary moves. All numbers are REAL, computed via the",
        "same tested pipeline functions used in production (src/satellite/esa_adb/).",
        "",
    ]

    results = {}
    for name, cfg in CANDIDATES.items():
        print(f"\nCandidate {name}: {cfg['description']}")
        result = build_candidate_result(channel_data, cfg["train_end"], cfg["val_start"], cfg["val_end"])
        results[name] = result
        violations = check_no_straddle(channel_data, label_rows_by_channel, cfg["train_end"], cfg["val_start"], cfg["val_end"])

        lines += [f"## Candidate {name}: {cfg['description']}", ""]
        lines += [f"- Train: {TRAIN_START.date()} to {cfg['train_end'].date()}",
                   f"- Validation: {cfg['val_start'].date()} to {cfg['val_end'].date()}",
                   f"- Test: {TEST_START.date()} to {TEST_END.date()}", ""]
        lines += ["| Split | Sequences | Normal | Anomaly | Anomaly % |", "|---|---:|---:|---:|---:|"]
        for split in ("train", "validation", "test"):
            ds = result["overall"][split]
            lines.append(f"| {split} | {ds.num_sequences:,} | {ds.normal_count:,} | {ds.anomaly_count:,} | {ds.anomaly_rate * 100:.2f}% |")
            print(f"  {split}: {ds.num_sequences:,} seq, {ds.anomaly_count:,} anomaly ({ds.anomaly_rate * 100:.2f}%)")

        val_clients_with_anomalies = [c for c in clients_mod.CLIENT_CHANNELS if result["per_client"][c]["validation"].anomaly_count > 0]
        lines += ["", f"**Validation: clients with >=1 anomaly sequence: {len(val_clients_with_anomalies)}/8** ({', '.join(val_clients_with_anomalies) or 'none'})", ""]
        lines += ["| Client | Train seq (anom) | Val seq (anom) | Test seq (anom) |", "|---|---:|---:|---:|"]
        for client in clients_mod.CLIENT_CHANNELS:
            tr, va, te = (result["per_client"][client][s] for s in ("train", "validation", "test"))
            lines.append(
                f"| {client} | {tr.num_sequences:,} ({tr.anomaly_count:,}) | {va.num_sequences:,} ({va.anomaly_count:,}) | {te.num_sequences:,} ({te.anomaly_count:,}) |"
            )

        lines += ["", f"**Leakage check (anomaly interval straddling train/val or val/test boundary): {'PASS' if not violations else f'FAIL ({len(violations)})'}**", ""]
        for v in violations[:10]:
            lines.append(f"- {v}")
        lines.append("")
        print(f"  leakage straddle check: {'PASS' if not violations else f'FAIL ({len(violations)})'}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

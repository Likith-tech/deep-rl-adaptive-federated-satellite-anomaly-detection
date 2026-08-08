"""Investigate candidate sequence-labeling strategies before committing
to one for the Phase 3 temporal model.

Background: an initial experiment used "sequence = anomaly if ANY
record in the window is anomalous" and found it produced a
near-single-class sequence dataset on NSL-KDD (see
results/reports/temporal_results.md, "Initial sequence-labeling
experiment"). This script measures the ACTUAL class distribution
produced by several alternative strategies, across sequence lengths and
splits, so the replacement strategy is chosen from evidence rather than
guessed.

Run from the repository root (after the Phase 1 pipeline has been run):

    python scripts/analyze_sequence_labeling.py

Writes: results/reports/sequence_labeling_analysis.md

Only label distributions are computed here — no model features/scaler/
encoder are needed, since this only depends on the record-level
label_binary column and row order (order is exactly what Phase 1's
interim/processed files already preserve for train/test; validation
uses the same order-preserving split used for real training).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.sequences import create_sequences, split_ordered_train_validation  # noqa: E402

INTERIM_TRAIN_PARQUET = REPO_ROOT / "data" / "interim" / "kdd_train_cleaned.parquet"
PROCESSED_TEST_PARQUET = REPO_ROOT / "data" / "processed" / "test.parquet"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "sequence_labeling_analysis.md"

SEQUENCE_LENGTHS = [8, 16, 32]
STRATEGIES: list[tuple[str, float | None]] = [
    ("any", None),
    ("majority", None),
    ("last", None),
    ("ratio", 0.25),
    ("ratio", 0.5),
    ("ratio", 0.75),
]


def strategy_label(strategy: str, threshold: float | None) -> str:
    return f"{strategy}@{threshold}" if strategy == "ratio" else strategy


def distribution_row(name: str, ds) -> dict:
    return {
        "split": name,
        "num_sequences": ds.num_sequences,
        "normal": ds.normal_count,
        "anomaly": ds.anomaly_count,
        "anomaly_pct": 100 * ds.anomaly_rate,
    }


def main() -> None:
    for path in (INTERIM_TRAIN_PARQUET, PROCESSED_TEST_PARQUET):
        if not path.exists():
            print(f"ERROR: {path} not found. Run the Phase 1 pipeline first.", file=sys.stderr)
            sys.exit(1)

    temporal_config = yaml.safe_load((REPO_ROOT / "configs" / "temporal.yaml").read_text())["temporal"]
    validation_size = temporal_config["validation_size"]

    full_train_df = pd.read_parquet(INTERIM_TRAIN_PARQUET)
    train_df, validation_df = split_ordered_train_validation(full_train_df, validation_size)
    test_df = pd.read_parquet(PROCESSED_TEST_PARQUET)

    print(f"Order-preserving split: {len(train_df)} train rows, {len(validation_df)} validation rows")
    print(f"Test rows (KDDTest+): {len(test_df)}")

    record_level_rates = {
        "train": float(train_df["label_binary"].mean()),
        "validation": float(validation_df["label_binary"].mean()),
        "test": float(test_df["label_binary"].mean()),
    }
    print(f"Record-level anomaly rates: {record_level_rates}")

    lines = [
        "# Sequence-Labeling Strategy Analysis",
        "",
        "Measured class distributions for candidate sequence-labeling "
        "strategies, computed directly from the real (order-preserving) "
        "train/validation splits and the real KDDTest+ split — no "
        "estimates. This analysis exists because an initial experiment "
        "using the \"any\" strategy at sequence_length=16 produced a "
        "100%-anomaly sequence dataset (see "
        "`results/reports/temporal_results.md`, \"Initial "
        "sequence-labeling experiment\"), which was rejected as an "
        "invalid benchmark.",
        "",
        "## Record-level anomaly rate (for reference)",
        "",
        f"- Train: {100 * record_level_rates['train']:.2f}%",
        f"- Validation: {100 * record_level_rates['validation']:.2f}%",
        f"- Test (KDDTest+): {100 * record_level_rates['test']:.2f}%",
        "",
    ]

    all_results = {}  # (seq_len, strategy_label) -> {split: row}

    for seq_len in SEQUENCE_LENGTHS:
        lines.append(f"## Sequence length {seq_len}")
        lines.append("")
        lines.append("| Strategy | Split | Sequences | Normal | Anomaly | Anomaly % |")
        lines.append("|---|---|---:|---:|---:|---:|")

        for strategy, threshold in STRATEGIES:
            label = strategy_label(strategy, threshold)
            splits = {
                "train": create_sequences(train_df, ["label_binary"], "label_binary", seq_len, None, strategy, threshold),
                "validation": create_sequences(validation_df, ["label_binary"], "label_binary", seq_len, None, strategy, threshold),
                "test": create_sequences(test_df, ["label_binary"], "label_binary", seq_len, None, strategy, threshold),
            }
            all_results[(seq_len, label)] = splits
            for split_name, ds in splits.items():
                row = distribution_row(split_name, ds)
                lines.append(
                    f"| {label} | {row['split']} | {row['num_sequences']} | {row['normal']} | "
                    f"{row['anomaly']} | {row['anomaly_pct']:.2f}% |"
                )
        lines.append("")

    # --- Recommendation, derived from the measured numbers above ---
    lines += [
        "## Recommendation",
        "",
        "A defensible strategy for this dataset must avoid a near-single-class "
        "sequence set (the flaw in the rejected \"any\" experiment) on ALL "
        "three splits, at the sequence length actually used, while staying "
        "simple to explain and semantically sensible for anomaly detection.",
        "",
    ]

    # Programmatically evaluate which strategies keep both classes reasonably
    # represented (defined here as: neither class below 15% of sequences)
    # across all three splits, for every candidate sequence length.
    MIN_MINORITY_PCT = 15.0
    balanced_candidates = []
    for seq_len in SEQUENCE_LENGTHS:
        for strategy, threshold in STRATEGIES:
            label = strategy_label(strategy, threshold)
            splits = all_results[(seq_len, label)]
            pct_anomaly = {name: 100 * ds.anomaly_rate for name, ds in splits.items()}
            minority_pct = {name: min(p, 100 - p) for name, p in pct_anomaly.items()}
            if all(m >= MIN_MINORITY_PCT for m in minority_pct.values()):
                balanced_candidates.append((seq_len, label, pct_anomaly))

    if balanced_candidates:
        lines.append(
            f"Strategy/length combinations where BOTH classes have at least "
            f"{MIN_MINORITY_PCT:.0f}% representation on every split "
            "(train, validation, and test) — i.e. not degenerate:"
        )
        lines.append("")
        lines.append("| Sequence length | Strategy | Train anomaly % | Validation anomaly % | Test anomaly % |")
        lines.append("|---|---|---:|---:|---:|")
        for seq_len, label, pct in balanced_candidates:
            lines.append(f"| {seq_len} | {label} | {pct['train']:.2f}% | {pct['validation']:.2f}% | {pct['test']:.2f}% |")
        lines.append("")
    else:
        lines.append(
            f"No strategy/length combination kept both classes above "
            f"{MIN_MINORITY_PCT:.0f}% on every split. The least imbalanced "
            "options should be used with this explicitly documented."
        )
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

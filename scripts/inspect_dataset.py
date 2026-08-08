"""Inspect the raw NSL-KDD dataset and print a human-readable summary.

Run from the repository root:

    python scripts/inspect_dataset.py

Expects `data/raw/KDDTrain+.txt` and `data/raw/KDDTest+.txt` to already
be present (see docs/datasets/dataset_selection.md for how to obtain
them). Fails with a clear message if they are missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import DatasetNotFoundError, load_nsl_kdd, profile_dataframe  # noqa: E402


def _print_profile(name: str, profile) -> None:
    print(f"\n=== {name} ===")
    print(f"Rows:    {profile.num_rows}")
    print(f"Columns: {profile.num_columns}")
    print(f"Numerical features:   {len(profile.numerical_columns)}")
    print(f"Categorical features: {len(profile.categorical_columns)} {profile.categorical_columns}")
    print(f"Duplicate rows:  {profile.duplicate_rows}")
    print(f"Infinite values: {profile.infinite_value_count}")
    missing_cols = {c.name: c.missing_count for c in profile.columns if c.missing_count > 0}
    print(f"Columns with missing values: {missing_cols if missing_cols else 'none'}")
    print(f"Memory usage: {profile.memory_usage_mb:.2f} MB")
    print(f"Label column: {profile.label_column}")
    print(f"Number of classes: {len(profile.class_distribution)}")
    print("Class distribution:")
    for label, count in sorted(profile.class_distribution.items(), key=lambda kv: -kv[1]):
        pct = 100 * count / profile.num_rows
        print(f"  {label:<16} {count:>7}  ({pct:5.2f}%)")


def main() -> None:
    try:
        dataset = load_nsl_kdd()
    except DatasetNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    train_profile = profile_dataframe(dataset.train, source_files=["KDDTrain+.txt"])
    test_profile = profile_dataframe(dataset.test, source_files=["KDDTest+.txt"])

    _print_profile("KDDTrain+", train_profile)
    _print_profile("KDDTest+", test_profile)

    normal_train = train_profile.class_distribution.get("normal", 0)
    anomaly_train = train_profile.num_rows - normal_train
    print("\n=== Binary label summary (KDDTrain+) ===")
    print(f"Normal:  {normal_train} ({100 * normal_train / train_profile.num_rows:.2f}%)")
    print(f"Anomaly: {anomaly_train} ({100 * anomaly_train / train_profile.num_rows:.2f}%)")


if __name__ == "__main__":
    main()

"""Generate dataset visualizations and the data-quality report from the
real NSL-KDD dataset.

Run from the repository root, after `data/raw/KDDTrain+.txt` and
`data/raw/KDDTest+.txt` are present:

    python scripts/generate_dataset_report.py

Writes:
    results/plots/dataset/*.png
    results/reports/dataset_quality.md

Every number in the generated report is computed from the actual raw
files present at run time — nothing here is fabricated or hardcoded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — this is a script, not a notebook
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import DatasetNotFoundError, load_nsl_kdd, profile_dataframe  # noqa: E402
from src.data.schema import ATTACK_CATEGORY_MAP, NUMERICAL_COLUMNS  # noqa: E402
from src.preprocessing.labels import add_binary_label  # noqa: E402

PLOTS_DIR = REPO_ROOT / "results" / "plots" / "dataset"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "dataset_quality.md"


def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["label_binary"].map({0: "Normal", 1: "Anomaly"}).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#3b82f6", "#ef4444"])
    ax.set_title("Binary Class Distribution (Normal vs Anomaly)", pad=20)
    ax.set_ylabel("Record count")
    ax.set_ylim(0, counts.max() * 1.2)
    for i, v in enumerate(counts):
        ax.text(i, v, f"{v}\n({100 * v / len(df):.1f}%)", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_attack_category_distribution(df: pd.DataFrame, out_path: Path) -> None:
    categories = df["label_original"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
    counts = categories.value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="#60a5fa")
    ax.set_title("Attack Category Distribution")
    ax.set_ylabel("Record count")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, features: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, feature in zip(axes.flat, features):
        df[feature].clip(upper=df[feature].quantile(0.99)).hist(ax=ax, bins=40, color="#3b82f6")
        ax.set_title(feature)
    fig.suptitle("Selected Numerical Feature Distributions (clipped at 99th percentile)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, features: list[str], out_path: Path) -> None:
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90, fontsize=6)
    ax.set_yticklabels(features, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Numerical Feature Correlation (KDDTrain+)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    try:
        dataset = load_nsl_kdd()
    except DatasetNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    train = add_binary_label(dataset.train, "attack")
    test = add_binary_label(dataset.test, "attack")

    train_profile = profile_dataframe(dataset.train, source_files=["KDDTrain+.txt"])
    test_profile = profile_dataframe(dataset.test, source_files=["KDDTest+.txt"])

    plot_class_distribution(train, PLOTS_DIR / "class_distribution_train.png")
    plot_class_distribution(test, PLOTS_DIR / "class_distribution_test.png")
    plot_attack_category_distribution(train, PLOTS_DIR / "attack_category_distribution_train.png")

    important_features = ["duration", "src_bytes", "dst_bytes", "count"]
    plot_feature_distributions(train, important_features, PLOTS_DIR / "feature_distributions.png")

    plot_correlation_heatmap(train, NUMERICAL_COLUMNS, PLOTS_DIR / "feature_correlation.png")

    # --- Data quality report (measured, not fabricated) ---
    normal_train = int((train["label_binary"] == 0).sum())
    anomaly_train = int((train["label_binary"] == 1).sum())
    normal_test = int((test["label_binary"] == 0).sum())
    anomaly_test = int((test["label_binary"] == 1).sum())

    lines = [
        "# NSL-KDD Data Quality Report",
        "",
        f"Generated from the actual raw files present in `data/raw/` at run time.",
        "",
        "## KDDTrain+",
        "",
        f"- Total rows: {train_profile.num_rows}",
        f"- Total features: {len(NUMERICAL_COLUMNS) + 3} (38 numerical + 3 categorical, excluding label/difficulty)",
        f"- Missing values: {sum(c.missing_count for c in train_profile.columns)}",
        f"- Duplicate rows: {train_profile.duplicate_rows}",
        f"- Infinite values: {train_profile.infinite_value_count}",
        f"- Normal: {normal_train} ({100 * normal_train / train_profile.num_rows:.2f}%)",
        f"- Anomaly: {anomaly_train} ({100 * anomaly_train / train_profile.num_rows:.2f}%)",
        f"- Number of distinct attack labels: {len(train_profile.class_distribution)}",
        "",
        "### Class distribution (KDDTrain+)",
        "",
        "| Label | Count | % |",
        "|---|---|---|",
    ]
    for label, count in sorted(train_profile.class_distribution.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {label} | {count} | {100 * count / train_profile.num_rows:.2f}% |")

    lines += [
        "",
        "## KDDTest+",
        "",
        f"- Total rows: {test_profile.num_rows}",
        f"- Missing values: {sum(c.missing_count for c in test_profile.columns)}",
        f"- Duplicate rows: {test_profile.duplicate_rows}",
        f"- Infinite values: {test_profile.infinite_value_count}",
        f"- Normal: {normal_test} ({100 * normal_test / test_profile.num_rows:.2f}%)",
        f"- Anomaly: {anomaly_test} ({100 * anomaly_test / test_profile.num_rows:.2f}%)",
        f"- Number of distinct attack labels: {len(test_profile.class_distribution)}",
        "",
        "### Class distribution (KDDTest+)",
        "",
        "| Label | Count | % |",
        "|---|---|---|",
    ]
    for label, count in sorted(test_profile.class_distribution.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {label} | {count} | {100 * count / test_profile.num_rows:.2f}% |")

    lines += [
        "",
        "## Class imbalance",
        "",
        f"KDDTrain+ normal:anomaly ratio is approximately "
        f"{normal_train / anomaly_train:.2f}:1." if anomaly_train else "N/A",
        "",
        "## Note on KDDTest+ label distribution",
        "",
        "KDDTest+ deliberately includes attack types absent from KDDTrain+ "
        "(this is intentional in the original NSL-KDD design, to test "
        "generalization to unseen attacks) — its class distribution should "
        "not be expected to match KDDTrain+.",
        "",
        "## Generated plots",
        "",
        "- `results/plots/dataset/class_distribution_train.png`",
        "- `results/plots/dataset/class_distribution_test.png`",
        "- `results/plots/dataset/attack_category_distribution_train.png`",
        "- `results/plots/dataset/feature_distributions.png`",
        "- `results/plots/dataset/feature_correlation.png`",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

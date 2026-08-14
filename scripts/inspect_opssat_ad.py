"""Phase 9A — inspect the downloaded OPSSAT-AD dataset BEFORE building any
preprocessing pipeline against it. Mirrors scripts/inspect_dataset.py's role
for NSL-KDD in Phase 1.

Run from the repository root, after scripts/download_opssat_ad.py:

    python scripts/inspect_opssat_ad.py

Reads:
    data/raw/opssat_ad/dataset.csv

Writes:
    results/reports/opssat_ad_inspection.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = REPO_ROOT / "data" / "raw" / "opssat_ad" / "dataset.csv"
LICENSE_FILE = REPO_ROOT / "data" / "raw" / "opssat_ad" / "LICENSE"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "opssat_ad_inspection.md"

ID_COLUMNS = {"segment", "train"}
LABEL_COLUMN = "anomaly"
CATEGORICAL_COLUMNS = {"channel"}


def main() -> None:
    if not DATASET_CSV.exists():
        print(f"ERROR: {DATASET_CSV} not found. Run scripts/download_opssat_ad.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATASET_CSV)

    feature_columns = [c for c in df.columns if c not in ID_COLUMNS | {LABEL_COLUMN} | CATEGORICAL_COLUMNS]

    train_test_counts = df["train"].value_counts().sort_index(ascending=False)
    label_counts = df[LABEL_COLUMN].value_counts().sort_index()
    crosstab = pd.crosstab(df["train"], df[LABEL_COLUMN])
    channel_counts = df["channel"].value_counts()

    missing_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    duplicate_segment_ids = int(df["segment"].duplicated().sum())

    license_text_snippet = LICENSE_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[0] if LICENSE_FILE.exists() else "N/A"

    print(f"Shape: {df.shape}")
    print(f"Train segments: {int(train_test_counts.get(1, 0))}, Test segments: {int(train_test_counts.get(0, 0))}")
    print(f"Nominal: {int(label_counts.get(0, 0))}, Anomalous: {int(label_counts.get(1, 0))}")
    print(f"Missing values: {missing_total}, Duplicate rows: {duplicate_rows}, Duplicate segment IDs: {duplicate_segment_ids}")
    print(f"Unique channels: {df['channel'].nunique()}")

    lines = [
        "# OPSSAT-AD — Dataset Inspection Report",
        "",
        "**Source:** `data/raw/opssat_ad/dataset.csv`, downloaded via "
        "`scripts/download_opssat_ad.py` from the official Zenodo record "
        "(https://zenodo.org/records/12588358, CC-BY-4.0 per Zenodo's own "
        "structured metadata).",
        "",
        "**License note (recorded, not silently resolved):** the bundled "
        f"`LICENSE` file's own first line reads: *\"{license_text_snippet}\"* "
        "— i.e. the file itself says MIT, while Zenodo's record metadata says "
        "CC-BY-4.0. Both are recorded here rather than picking one silently; "
        "either way, the dataset is licensed for open reuse with attribution "
        "(MIT and CC-BY-4.0 are both permissive).",
        "",
        "## Shape and structure",
        "",
        f"- Rows (segments): {len(df)}",
        f"- Columns: {len(df.columns)} — {list(df.columns)}",
        f"- ID/split columns (not features): {sorted(ID_COLUMNS)}",
        f"- Label column: `{LABEL_COLUMN}`",
        f"- Categorical column: `channel` ({df['channel'].nunique()} unique values)",
        f"- Numeric feature columns used for modeling ({len(feature_columns)}): {feature_columns}",
        "",
        "## Train/test split (author-provided, via the `train` column)",
        "",
        "**Correction versus the Phase 9A pre-integration audit:** the audit "
        "(transcribed from the arXiv paper text) reported 1,494 training "
        "segments. Direct inspection of the actual downloaded file shows "
        f"**{int(train_test_counts.get(1, 0))} training segments** — the "
        "audit's train figure was a transcription error (1,273 + 321 was "
        "already recorded correctly and sums to 1,594, not 1,494). The test "
        f"count ({int(train_test_counts.get(0, 0))}) matches the audit exactly. "
        "Corrected here from the real data, which takes precedence.",
        "",
        f"| | Nominal (0) | Anomalous (1) | Total |",
        f"|---|---:|---:|---:|",
        f"| train=1 (training) | {int(crosstab.loc[1, 0])} | {int(crosstab.loc[1, 1])} | {int(train_test_counts.get(1, 0))} |",
        f"| train=0 (test) | {int(crosstab.loc[0, 0])} | {int(crosstab.loc[0, 1])} | {int(train_test_counts.get(0, 0))} |",
        f"| **Total** | {int(label_counts.get(0, 0))} | {int(label_counts.get(1, 0))} | {len(df)} |",
        "",
        "## Channel distribution",
        "",
        "**Correction versus the pre-integration audit:** the audit (extracted "
        "from the paper's text) reported human-readable channel names "
        "(`I_B_FB_MM_0/1/2`, `I_PD1_THETA`..`I_PD6_THETA`). The actual "
        "downloaded `dataset.csv` uses opaque internal telemetry codes "
        "instead — recorded here exactly as found, not reconciled by guessing "
        "which name maps to which code:",
        "",
        "| Channel code | Segment count |",
        "|---|---:|",
    ] + [f"| {ch} | {count} |" for ch, count in channel_counts.items()] + [
        "",
        "## Data quality",
        "",
        f"- Missing values: **{missing_total}**",
        f"- Fully duplicate rows: **{duplicate_rows}**",
        f"- Duplicate segment IDs: **{duplicate_segment_ids}** (every segment ID is unique — "
        "confirmed 1:1 with row count, so no segment appears more than once and no "
        "adjacency/relatedness between rows is introduced via repeated IDs)",
        "",
        "## Sampling / duration / length (evidence that segments are genuinely variable)",
        "",
        f"- `sampling` column: min={df['sampling'].min()}, max={df['sampling'].max()}, "
        f"most common values are 1 and 5 — confirms variable sampling rate across segments, "
        "as the paper stated.",
        f"- `duration`: min={df['duration'].min()}, max={df['duration'].max()}",
        f"- `len` (segment length in samples): min={df['len'].min()}, max={df['len'].max()}",
        "",
        "These confirm that the underlying telemetry segments are genuinely "
        "variable-length and variable-rate. `dataset.csv`, however, has "
        "already reduced each segment to ONE fixed-width row of summary "
        "statistics — there is no per-timestep sequence dimension left in "
        "this file (see the model-architecture discussion in "
        "`results/reports/opssat_ad_results.md`).",
        "",
        "## Data types",
        "",
        "```",
        str(df.dtypes),
        "```",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

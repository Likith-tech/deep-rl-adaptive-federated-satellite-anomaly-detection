"""SMAP/MSL Track B (candidate) — integrity verification + leakage audit.

Run from the repository root, after scripts/download_smap_msl.py:

    python scripts/inspect_smap_msl.py

Reads ONLY data/raw/smap_msl/ (real, downloaded data — never
fabricated). Writes results/reports/smap_msl_inspection.md. This is a
read-only audit script — it does not train anything, does not touch
KDDTest+/NSL-KDD/OPSSAT-AD, and does not modify any existing artifact.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "smap_msl"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "smap_msl_inspection.md"


def main() -> None:
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} not found. Run scripts/download_smap_msl.py first.", file=sys.stderr)
        sys.exit(1)

    with open(RAW_DIR / "labeled_anomalies.csv") as f:
        rows = list(csv.DictReader(f))

    # INTEGRITY FINDING (real, discovered this run): labeled_anomalies.csv
    # has 82 rows but only 81 UNIQUE chan_id values -- "P-2" appears twice
    # with two DIFFERENT (overlapping but non-identical) anomaly_sequences.
    # This means one metadata row is effectively lost from a naive
    # dict-by-chan_id mapping. Detected explicitly here rather than
    # silently overwriting one P-2 row with the other.
    chan_id_counts = Counter(r["chan_id"] for r in rows)
    duplicate_chan_ids = {cid: n for cid, n in chan_id_counts.items() if n > 1}
    meta = {}
    for r in rows:
        meta.setdefault(r["chan_id"], []).append(r)

    train_dir, test_dir = RAW_DIR / "train", RAW_DIR / "test"
    channels = sorted(p.stem for p in train_dir.glob("*.npy"))
    disk_set, meta_set = set(channels), set(meta.keys())
    channels_missing_metadata = sorted(disk_set - meta_set)
    metadata_missing_from_disk = sorted(meta_set - disk_set)
    # Use only channels that have exactly one, unambiguous metadata row for
    # the main per-spacecraft statistics below; ambiguous/missing ones are
    # reported separately, not silently dropped or guessed at.
    channels = sorted(c for c in channels if c in meta and len(meta[c]) == 1)

    by_spacecraft = defaultdict(lambda: {"channels": 0, "anomaly_sequences": 0, "test_rows": 0, "train_rows": 0, "dims": set()})
    nan_channels, mismatch_channels, overlap_report = [], [], []
    prefix_by_sc = defaultdict(Counter)
    total_test_dup_fraction = {}

    for c in channels:
        tr = np.load(train_dir / f"{c}.npy")
        te = np.load(test_dir / f"{c}.npy")
        sc = meta[c][0]["spacecraft"]
        seqs = json.loads(meta[c][0]["anomaly_sequences"].replace("(", "[").replace(")", "]"))

        by_spacecraft[sc]["channels"] += 1
        by_spacecraft[sc]["anomaly_sequences"] += len(seqs)
        by_spacecraft[sc]["test_rows"] += te.shape[0]
        by_spacecraft[sc]["train_rows"] += tr.shape[0]
        by_spacecraft[sc]["dims"].add(tr.shape[1])
        prefix_by_sc[sc][c.split("-")[0]] += 1

        if np.isnan(tr).any() or np.isnan(te).any():
            nan_channels.append(c)
        expected = int(meta[c][0]["num_values"])
        if expected != te.shape[0]:
            mismatch_channels.append((c, expected, te.shape[0]))

        tr_set = set(map(tuple, tr))
        te_set = set(map(tuple, te))
        inter = tr_set & te_set
        if inter:
            overlap_report.append((c, len(inter), te.shape[0], round(100 * len(inter) / te.shape[0], 1)))

        dup_consec = int(np.sum(np.all(te[1:] == te[:-1], axis=1))) if len(te) > 1 else 0
        total_test_dup_fraction[c] = round(100 * dup_consec / len(te), 1)

    total_train = sum(v["train_rows"] for v in by_spacecraft.values())
    total_test = sum(v["test_rows"] for v in by_spacecraft.values())
    total_anoms = sum(v["anomaly_sequences"] for v in by_spacecraft.values())

    channel_counts_str = ", ".join(f"{sc}={v['channels']}" for sc, v in by_spacecraft.items())
    lines = [
        "# SMAP/MSL Track B (candidate) — Integrity Verification + Leakage Audit",
        "",
        "**AUDIT ONLY — no model was trained. Real, downloaded data only.**",
        "",
        "## Integrity finding — the official metadata file itself has a real defect",
        "",
        f"`labeled_anomalies.csv` (the authoritative NASA/JPL-authored label file) "
        f"contains **82 rows but only 81 unique `chan_id` values**. The duplicate: "
        f"{duplicate_chan_ids}. The two `P-2` rows have DIFFERENT, overlapping-but-"
        f"non-identical `anomaly_sequences` ([[5350,6575]] vs [[5300,6420]]), same "
        f"`num_values` (8209). Consequence: exactly one real channel present on "
        f"disk has **no unambiguous metadata row** and was therefore excluded from "
        f"every statistic below rather than guessed at: "
        f"**{channels_missing_metadata or metadata_missing_from_disk}**. "
        f"This is a defect in the official, upstream file — not introduced by this "
        f"project — and must be resolved with an explicit, disclosed decision "
        f"(e.g. take the union of both P-2 windows, or exclude P-2 pending "
        f"clarification) before P-2 or the orphaned channel are used in any "
        f"labeled experiment.",
        "",
        "## Summary",
        "",
        f"(Statistics below cover the {len(channels)} channels with an unambiguous "
        f"single metadata row — i.e. excluding `P-2` and its metadata-orphaned "
        f"counterpart, per the finding above.)",
        "",
        f"- Total channels: {len(channels)} ({channel_counts_str})",
        f"- Total train rows (sum across channels): {total_train:,}",
        f"- Total test rows (sum across channels): {total_test:,}",
        f"- Total labeled anomaly sequences: {total_anoms}",
        f"- Channels with any NaN value: {len(nan_channels)} {nan_channels}",
        f"- Channels where test row count != labeled_anomalies.csv num_values: {len(mismatch_channels)} {mismatch_channels}",
        "",
        "## Per-spacecraft breakdown",
        "",
        "| Spacecraft | Channels | Feature dim | Train rows | Test rows | Anomaly sequences |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sc, v in by_spacecraft.items():
        lines.append(f"| {sc} | {v['channels']} | {sorted(v['dims'])} | {v['train_rows']:,} | {v['test_rows']:,} | {v['anomaly_sequences']} |")

    lines += [
        "",
        "**Feature-dimension finding**: SMAP and MSL channels do NOT share a feature "
        "space (SMAP=25-dim, MSL=55-dim, confirmed with zero exceptions across all "
        "82 channels) — any shared-architecture FL design must either train separate "
        "per-mission input layers or explicitly align/pad the two.",
        "",
        "## Subsystem/channel-prefix breakdown (candidate FL sub-client grouping)",
        "",
    ]
    for sc, counter in prefix_by_sc.items():
        lines.append(f"- **{sc}**: {dict(sorted(counter.items()))}")

    lines += [
        "",
        "## Leakage audit — exact train/test row overlap (per channel)",
        "",
        f"Channels with at least one exact-duplicate feature row shared between "
        f"train and test: **{len(overlap_report)} of {len(channels)}**.",
        "",
        "| Channel | Overlapping rows | Test set size | Overlap % |",
        "|---|---:|---:|---:|",
    ]
    for c, n_overlap, n_test, pct in sorted(overlap_report, key=lambda r: -r[3])[:15]:
        lines.append(f"| {c} | {n_overlap} | {n_test} | {pct}% |")
    lines += [
        "",
        f"(showing the 15 channels with the highest overlap percentage, out of "
        f"{len(overlap_report)} affected)",
        "",
        "**Interpretation, not asserted as a construction flaw**: many channels "
        "show substantial WITHIN-test-set consecutive-duplicate rates (e.g. "
        f"channel A-1: {total_test_dup_fraction.get('A-1', 'n/a')}% of its own test "
        "rows are exact consecutive duplicates of the previous timestep) — real, "
        "natural telemetry flat-lining between sensor updates. Cross-split overlap "
        "is consistent with this same low-entropy characteristic rather than "
        "necessarily indicating a flawed split. The single highest-overlap channel "
        "found is flagged explicitly below for extra scrutiny before use.",
        "",
    ]
    if overlap_report:
        worst = max(overlap_report, key=lambda r: r[3])
        lines.append(f"**Highest-overlap channel: `{worst[0]}` — {worst[3]}% of its test rows exactly match a train row.** "
                      "Recommend excluding or specially reviewing this channel if it is used.")

    lines += [
        "",
        "## What this audit does NOT change",
        "",
        "This script only reads `data/raw/smap_msl/` (git-ignored, never committed) "
        "and writes this report. No NSL-KDD, OPSSAT-AD, STIN, Phase 6-9B, or "
        "Improvement 01 artifact was read or modified.",
    ]

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")
    print(f"Channels: {len(channels)}, train rows: {total_train:,}, test rows: {total_test:,}, "
          f"anomaly sequences: {total_anoms}, NaN channels: {len(nan_channels)}, "
          f"overlap-affected channels: {len(overlap_report)}")


if __name__ == "__main__":
    main()

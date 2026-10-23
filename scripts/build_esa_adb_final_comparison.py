"""ESA-ADB Step D — final honest comparison across all five ESA-ADB
experiments run in this phase (and the two prior centralized
baselines). Reads ONLY already-saved `test_metrics.json`/
`fairness_summary.json` files from each experiment directory — never
recomputes or fabricates a number.

Run from the repository root, after baseline_01, baseline_02_multivariate,
fedavg_baseline_01, adaptive_rule_baseline_01, and dqn_adaptive_fl_01 have
all been run:

    python scripts/build_esa_adb_final_comparison.py

Writes docs/research/experiments/esa-adb/final_comparison.json and
prints a markdown table to stdout for inclusion in the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS = {
    "Centralized single-channel (baseline_01)": REPO_ROOT / "experiments" / "esa_adb" / "baseline_01",
    "Centralized multivariate (baseline_02_multivariate)": REPO_ROOT / "experiments" / "esa_adb" / "baseline_02_multivariate",
    "FedAvg (fedavg_baseline_01)": REPO_ROOT / "experiments" / "esa_adb" / "fedavg_baseline_01",
    "Rule-based adaptive FL (adaptive_rule_baseline_01)": REPO_ROOT / "experiments" / "esa_adb" / "adaptive_rule_baseline_01",
    "DQN adaptive FL, 48-round pretrain budget (dqn_adaptive_fl_01)": REPO_ROOT / "experiments" / "esa_adb" / "dqn_adaptive_fl_01",
    "DQN adaptive FL, 240-round extended pretrain budget (dqn_adaptive_fl_02_extended)": REPO_ROOT / "experiments" / "esa_adb" / "dqn_adaptive_fl_02_extended",
}

OUT_PATH = REPO_ROOT / "docs" / "research" / "experiments" / "esa-adb" / "final_comparison.json"


def main() -> None:
    rows = {}
    for name, d in EXPERIMENTS.items():
        test_path = d / "test_metrics.json"
        fairness_path = d / "fairness_summary.json"
        if not test_path.exists():
            print(f"WARNING: {test_path} does not exist — skipping {name!r}", file=sys.stderr)
            continue
        m = json.loads(test_path.read_text())["metrics"]
        row = {
            "accuracy": m.get("accuracy"), "precision": m.get("precision"), "recall": m.get("recall"),
            "f1": m.get("f1"), "roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"),
            "false_positive_rate": m.get("false_positive_rate"),
        }
        if fairness_path.exists():
            fs = json.loads(fairness_path.read_text())
            row["fairness_gap_f1"] = fs.get("test_fairness_gap_f1")
        else:
            row["fairness_gap_f1"] = None
        rows[name] = row

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {OUT_PATH}\n")

    header = ["Experiment", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "FPR", "Fairness Gap (F1)"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name, row in rows.items():
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
        print(f"| {name} | {fmt(row['accuracy'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | "
              f"{fmt(row['f1'])} | {fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} | {fmt(row['false_positive_rate'])} | "
              f"{fmt(row['fairness_gap_f1'])} |")


if __name__ == "__main__":
    main()

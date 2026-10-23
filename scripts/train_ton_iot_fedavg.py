"""TON_IoT FedAvg baseline — reproduces the shape of FedSTAR's Table 3
(Accuracy/Precision/Recall/F1, mean +/- std over 5 seeds, IID and
Non-IID) using OUR OWN pipeline (standard sample-count-weighted FedAvg,
BaselineMLP, 8 simulated clients) — NOT a reproduction of FedSTAR's own
architecture. Run scripts/prepare_ton_iot.py first.

Writes to experiments/ton_iot/fedavg_baseline_01/ only.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.training.tabular_fl_experiment import run_fedavg_cell  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "ton_iot"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "ton_iot" / "fedavg_baseline_01"
SCRATCH_DIR = EXPERIMENT_DIR / "_scratch"

SEEDS = [42, 43, 44, 45, 46]
NUM_ROUNDS = 10
MODEL_CONFIG = {"input_dim": None, "hidden_dimensions": [128, 64], "dropout": 0.3, "output_dim": 1}
TRAINING_CONFIG = {"batch_size": 128, "learning_rate": 0.001, "local_epochs": 1}


def summarize(cells: list[dict]) -> dict:
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate"]
    summary = {}
    for m in metric_names:
        values = [c["test_metrics"][m] for c in cells if c["test_metrics"].get(m) is not None]
        summary[m] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values} if values else None
    return summary


def main() -> None:
    t_start = time.time()
    metadata = json.loads((PROCESSED_DIR / "metadata.json").read_text())
    feature_columns = metadata["feature_columns"]
    MODEL_CONFIG["input_dim"] = len(feature_columns)
    print(f"TON_IoT: {len(feature_columns)} features, {metadata['rows']}")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    all_cells = []

    for partition_type in ("iid", "non_iid"):
        cells = []
        for seed in SEEDS:
            print(f"\n=== TON_IoT | {partition_type} | seed={seed} ===")
            cell = run_fedavg_cell(
                dataset_name="ton_iot", processed_dir=PROCESSED_DIR, category_column="type",
                partition_type=partition_type, seed=seed, num_rounds=NUM_ROUNDS,
                model_config=MODEL_CONFIG, training_config=TRAINING_CONFIG,
                feature_columns=feature_columns, experiment_scratch_dir=SCRATCH_DIR,
            )
            tm = cell["test_metrics"]
            print(f"  acc={tm['accuracy']:.4f} prec={tm['precision']:.4f} rec={tm['recall']:.4f} "
                  f"f1={tm['f1']:.4f} roc_auc={tm['roc_auc']:.4f} pr_auc={tm['pr_auc']:.4f} fpr={tm['false_positive_rate']:.4f}")
            cells.append(cell)
            all_cells.append(cell)

        summary = summarize(cells)
        print(f"\n{partition_type} summary (mean +/- std over {len(SEEDS)} seeds):")
        for m, s in summary.items():
            if s:
                print(f"  {m}: {100*s['mean']:.2f} +/- {100*s['std']:.2f} (%)")
        (EXPERIMENT_DIR / f"summary_{partition_type}.json").write_text(json.dumps({
            "partition_type": partition_type, "seeds": SEEDS, "cells": cells, "summary": summary,
        }, indent=2))

    (EXPERIMENT_DIR / "all_results.json").write_text(json.dumps({
        "dataset": "ton_iot", "seeds": SEEDS, "num_rounds": NUM_ROUNDS,
        "model_config": MODEL_CONFIG, "training_config": TRAINING_CONFIG,
        "feature_columns": feature_columns, "cells": all_cells,
        "total_seconds": time.time() - t_start,
    }, indent=2))

    if SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR)  # per-cell partitions/checkpoints are large and reproducible; only summaries kept

    print(f"\nDone in {time.time() - t_start:.1f}s. Results in {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()

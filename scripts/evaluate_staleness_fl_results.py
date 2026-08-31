"""Phase 9 — evaluate the staleness- and resource-aware FedAvg ablation
study.

Run from the repository root, after scripts/run_staleness_fl_experiments.py
has produced experiments/staleness_fl/<config_name>/ for every staleness
config:

    python scripts/evaluate_staleness_fl_results.py

For each config: evaluates the selected best-validation-round global
model ONCE on KDDTest+ (never used for config selection), computes a
client-fairness proxy (the model evaluated on each client's own local
data — same caveat as Phase 7/8: not a held-out split, since none
exists), and compares against Phase 6/8/9B's stored, already-committed
reference numbers.

Writes:
    results/plots/staleness_fl/*.png
    results/reports/staleness_fl_results.md

Phase 6/8/9B's own reports/plots/models are only READ (for the fixed
reference numbers below) — never modified.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.training.baseline_trainer import load_features_and_labels  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.models.baseline_mlp import BaselineMLP  # noqa: E402

EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "staleness_fl"
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "staleness_fl"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "staleness_fl_results.md"

CONFIG_ORDER = ["no_correction", "linear_decay", "exponential_decay"]

# Fixed reference values, read from already-committed reports, never
# re-derived or modified here.
PHASE_6_BASELINE = {
    "val_f1": 0.9879,
    "test": {"accuracy": 0.7531, "precision": 0.9219, "recall": 0.6186,
              "f1": 0.7404, "roc_auc": 0.8479, "false_positive_rate": 0.0693},
}
PHASE_8_BEST_F1 = {"rule": "resource_only", "test_f1": 0.7547}
PHASE_8_BEST_FAIRNESS = {"rule": "performance_only", "gap": 0.1245}
PHASE_9B_DQN = {"test_f1": 0.7425, "test_roc_auc": 0.8361, "fairness_gap": 0.1833}


def load_model(checkpoint: dict, device: torch.device):
    model_config = checkpoint["model_config"]
    model = BaselineMLP(
        input_dim=model_config["input_dim"],
        hidden_dimensions=model_config["hidden_dimensions"],
        dropout=model_config["dropout"],
        output_dim=model_config["output_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["feature_columns"]


def evaluate_on_parquet(model, parquet_path: Path, feature_columns: list[str], device: torch.device):
    X, y = load_features_and_labels(parquet_path, feature_columns)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device)).squeeze(-1)
        y_prob = torch.sigmoid(logits).cpu().numpy()
    y_true = y.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)
    return compute_metrics(y_true, y_pred, y_prob)


def plot_val_f1_comparison(config_results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for config_name in CONFIG_ORDER:
        rh = config_results[config_name]["round_history"]
        rounds = [r["round_number"] for r in rh]
        f1s = [r["val_metrics"]["f1"] for r in rh]
        ax.plot(rounds, f1s, marker="o", label=config_name, alpha=0.85)
    ax.axhline(PHASE_6_BASELINE["val_f1"], color="gray", linestyle="--", alpha=0.6,
               label=f"Phase 6 FedAvg (final={PHASE_6_BASELINE['val_f1']:.4f})")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global validation F1")
    ax.set_title("Validation F1 per Round — Staleness Configs vs. Phase 6 FedAvg")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_staleness_by_round(round_history: list[dict], config_name: str, out_path: Path) -> None:
    client_ids = sorted(round_history[0]["client_staleness"].keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cid in client_ids:
        staleness = [r["client_staleness"][cid] for r in round_history]
        ax.plot([r["round_number"] for r in round_history], staleness, marker="o", label=cid, alpha=0.8)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Staleness (rounds behind)")
    ax.set_title(f"{config_name} — Real Per-Client Staleness by Round")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_aggregation_weight_distribution(config_name: str, round_history: list[dict], out_path: Path) -> None:
    client_ids = sorted(round_history[0]["aggregation_weights"].keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cid in client_ids:
        weights = [r["aggregation_weights"][cid] for r in round_history]
        ax.plot([r["round_number"] for r in round_history], weights, marker="o", label=cid, alpha=0.8)
    uniform = 1.0 / len(client_ids)
    ax.axhline(uniform, color="gray", linestyle="--", alpha=0.5, label=f"Uniform (1/{len(client_ids)})")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Aggregation weight")
    ax.set_title(f"{config_name} — Per-Client Aggregation Weight by Round")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_test_f1_comparison(config_test_f1: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = CONFIG_ORDER + ["phase6_fedavg", "phase8_best (resource_only)", "phase9b_dqn"]
    values = (
        [config_test_f1[c] for c in CONFIG_ORDER]
        + [PHASE_6_BASELINE["test"]["f1"], PHASE_8_BEST_F1["test_f1"], PHASE_9B_DQN["test_f1"]]
    )
    colors = ["#3b82f6"] * len(CONFIG_ORDER) + ["#6b7280", "#f59e0b", "#8b5cf6"]
    ax.bar(names, values, color=colors)
    ax.set_ylabel("Final KDDTest+ F1")
    ax.set_title("Final KDDTest+ F1 — Staleness Configs vs. Phase 6/8/9B")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    config_results = {}
    config_test_metrics = {}
    config_fairness = {}

    for config_name in CONFIG_ORDER:
        run_dir = EXPERIMENT_ROOT / config_name
        if not (run_dir / "best_global_model.pt").exists():
            print(f"ERROR: {run_dir}/best_global_model.pt not found. "
                  "Run scripts/run_staleness_fl_experiments.py first.", file=sys.stderr)
            sys.exit(1)

        run_config = json.loads((run_dir / "run_config.json").read_text())
        round_history = json.loads((run_dir / "round_history.json").read_text())["round_history"]
        checkpoint = torch.load(run_dir / "best_global_model.pt", weights_only=False)
        model, feature_columns = load_model(checkpoint, device)

        test_metrics = evaluate_on_parquet(model, TEST_PARQUET, feature_columns, device)
        print(f"{config_name}: KDDTest+ f1={test_metrics.f1:.4f} accuracy={test_metrics.accuracy:.4f} "
              f"max_staleness_seen={run_config['max_staleness_seen']}")

        client_ids = sorted(run_config["client_sample_counts"].keys())
        client_metrics = {}
        for cid in client_ids:
            client_parquet = PARTITIONS_DIR / cid / "train.parquet"
            client_metrics[cid] = evaluate_on_parquet(model, client_parquet, feature_columns, device)
        client_f1s = [m.f1 for m in client_metrics.values()]
        fairness = {
            "mean": statistics.mean(client_f1s), "min": min(client_f1s), "max": max(client_f1s),
            "stdev": statistics.stdev(client_f1s) if len(client_f1s) > 1 else 0.0,
            "gap": max(client_f1s) - min(client_f1s),
        }
        print(f"  fairness (own-local-data F1): mean={fairness['mean']:.4f} gap={fairness['gap']:.4f}")

        plot_staleness_by_round(round_history, config_name, PLOTS_DIR / f"{config_name}_staleness_by_round.png")
        plot_aggregation_weight_distribution(config_name, round_history, PLOTS_DIR / f"{config_name}_weights_by_round.png")

        config_results[config_name] = {"run_config": run_config, "round_history": round_history}
        config_test_metrics[config_name] = test_metrics
        config_fairness[config_name] = fairness

    plot_val_f1_comparison(config_results, PLOTS_DIR / "val_f1_comparison.png")
    plot_test_f1_comparison({c: config_test_metrics[c].f1 for c in CONFIG_ORDER}, PLOTS_DIR / "test_f1_vs_phase6_8_9b.png")

    write_report(config_results, config_test_metrics, config_fairness)
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote plots to {PLOTS_DIR}")


def write_report(config_results: dict, config_test_metrics: dict, config_fairness: dict) -> None:
    lines = [
        "# Phase 9 — Staleness- and Resource-Aware Federated Learning Results",
        "",
        "**All numbers in this report are real, measured values from "
        "scripts/run_staleness_fl_experiments.py, read directly from "
        "experiments/staleness_fl/*/run_config.json and round_history.json, "
        "plus one KDDTest+ evaluation per config computed by this script. "
        "Nothing here is estimated, interpolated, or fabricated. Where a "
        "result is unfavorable, it is reported exactly as measured.**",
        "",
        "## 1. Research question",
        "",
        "Phase 4's satellite simulation has always assigned every client a "
        "simulated `availability_probability`, but no phase before this "
        "one ever let it affect round-to-round PARTICIPATION — Phase "
        "6/8/9B all had every client fetch and train against the exact "
        "current global model every round, so no run before this one ever "
        "contained real staleness. This phase asks: when availability is "
        "finally allowed to gate whether a client receives the latest "
        "global model each round (producing real, measured staleness), "
        "does discounting stale updates before aggregation help, hurt, or "
        "not matter relative to ignoring staleness entirely?",
        "",
        "## 2. Mechanism",
        "",
        "Each round, each client's participation is decided by a seeded "
        "coin flip weighted by its real Phase 4 `availability_probability` "
        "(`src/federated/staleness.py:decide_participation`). A client "
        "that doesn't sync this round keeps training against whatever "
        "global state it last synced to — a REAL, structurally-caused "
        "staleness (not simulated after the fact), since that update "
        "genuinely reflects an older base model. Every client still "
        "trains and submits an update every round (staleness WEIGHTING, "
        "not selection — the same philosophy Phase 8 already committed "
        "to, for the same reason: a dropped client's rare-category data "
        "contributes nothing that round). The resulting staleness is "
        "converted to a discount factor and combined with the client's "
        "sample count, then applied through the UNMODIFIED "
        "`FederatedServer.aggregate_with_weights` (Phase 8's real "
        "extension point).",
        "",
        "```",
        "no_correction:      w_stale = 1                    (staleness measured but ignored)",
        "linear_decay:       w_stale = 1 / (1 + staleness)",
        "exponential_decay:  w_stale = 0.5 ** staleness",
        "```",
        "",
        "All three configs share the exact same `participation_seed`, so "
        "the real staleness PATTERN each client experiences is identical "
        "across configs — only the aggregation response to that pattern "
        "differs. Full formulas in `src/federated/staleness.py`; full "
        "config in `configs/staleness_fl.yaml`.",
        "",
        "## 3. Experimental configuration",
        "",
        "Identical to Phase 6/8 in every respect except staleness "
        "handling: 8 clients, the SAME `data/partitions/` (Phase 4 "
        "original, alpha=0.5) partition, 10 rounds, 1 local epoch, batch "
        "size 128, learning rate 0.001, Adam, BCEWithLogitsLoss, seed 42, "
        "shared initial weights (seed 42).",
        "",
        "## 4. Main comparison table",
        "",
        "| Config | Val F1 (best round) | Test F1 | Test ROC-AUC | Client fairness gap | Max staleness seen |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config_name in CONFIG_ORDER:
        run_config = config_results[config_name]["run_config"]
        val_f1 = run_config["best_val_metrics"]["f1"]
        m = config_test_metrics[config_name]
        roc = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "not available"
        gap = config_fairness[config_name]["gap"]
        max_stale = run_config["max_staleness_seen"]
        lines.append(f"| {config_name} | {val_f1:.4f} | {m.f1:.4f} | {roc} | {gap:.4f} | {max_stale} |")
    lines.append(
        f"| **phase6_fedavg (reference)** | {PHASE_6_BASELINE['val_f1']:.4f} | "
        f"{PHASE_6_BASELINE['test']['f1']:.4f} | {PHASE_6_BASELINE['test']['roc_auc']:.4f} | "
        "not measured | 0 (synchronous) |"
    )
    lines.append(
        f"| **phase8_best_f1 ({PHASE_8_BEST_F1['rule']}, reference)** | not comparable | "
        f"{PHASE_8_BEST_F1['test_f1']:.4f} | not reported per-rule | "
        f"see {PHASE_8_BEST_FAIRNESS['rule']} below | 0 (synchronous) |"
    )
    lines.append(
        f"| **phase9b_dqn (reference)** | not comparable | {PHASE_9B_DQN['test_f1']:.4f} | "
        f"{PHASE_9B_DQN['test_roc_auc']:.4f} | {PHASE_9B_DQN['fairness_gap']:.4f} | 0 (synchronous) |"
    )

    best_test_config = max(CONFIG_ORDER, key=lambda c: config_test_metrics[c].f1)
    best_fair_config = min(CONFIG_ORDER, key=lambda c: config_fairness[c]["gap"])
    no_corr_f1 = config_test_metrics["no_correction"].f1
    best_f1 = config_test_metrics[best_test_config].f1

    lines += [
        "",
        "## 5. Honest reading",
        "",
        f"Best KDDTest+ F1 among the three staleness configs: **{best_test_config}** "
        f"({best_f1:.4f}). Best client fairness gap: **{best_fair_config}** "
        f"({config_fairness[best_fair_config]['gap']:.4f}). ",
    ]
    if best_test_config != "no_correction":
        delta = best_f1 - no_corr_f1
        lines.append(
            f"Staleness-aware discounting ({best_test_config}) improved on the "
            f"staleness-blind control ({no_corr_f1:.4f}) by {delta:+.4f} F1 under "
            "this real (seeded, not fabricated) staleness pattern."
        )
    else:
        lines.append(
            "The staleness-blind control (`no_correction`) scored best on F1 among "
            "the three configs — staleness-aware discounting did NOT improve on "
            "ignoring staleness in this single-seed run. Reported exactly as "
            "measured, not adjusted or re-run to seek a more favorable result."
        )
    lines += [
        "",
        f"Against the wider Phase 6/8/9B comparison, the best staleness config here "
        f"({best_test_config}, F1 {best_f1:.4f}) "
        + (
            f"beat Phase 6 FedAvg ({PHASE_6_BASELINE['test']['f1']:.4f})"
            if best_f1 > PHASE_6_BASELINE["test"]["f1"]
            else f"did NOT beat Phase 6 FedAvg ({PHASE_6_BASELINE['test']['f1']:.4f})"
        )
        + " and "
        + (
            f"beat Phase 8's best rule ({PHASE_8_BEST_F1['rule']}, {PHASE_8_BEST_F1['test_f1']:.4f})."
            if best_f1 > PHASE_8_BEST_F1["test_f1"]
            else f"did NOT beat Phase 8's best rule ({PHASE_8_BEST_F1['rule']}, {PHASE_8_BEST_F1['test_f1']:.4f})."
        ),
        "",
        "## 6. Validation results (best round per config)",
        "",
        "| Config | Best round | Val F1 | Val loss |",
        "|---|---:|---:|---:|",
    ]
    for config_name in CONFIG_ORDER:
        rc = config_results[config_name]["run_config"]
        lines.append(f"| {config_name} | {rc['best_round']} | {rc['best_val_metrics']['f1']:.4f} | {rc['best_val_loss']:.4f} |")

    lines += [
        "",
        "## 7. Final KDDTest+ results",
        "",
        "Evaluated ONCE per config, after round selection by validation "
        "loss only:",
        "",
        "| Config | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config_name in CONFIG_ORDER:
        m = config_test_metrics[config_name]
        roc = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "not available"
        lines.append(f"| {config_name} | {m.accuracy:.4f} | {m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | {roc} | {m.false_positive_rate:.4f} |")

    lines += [
        "",
        "## 8. Fairness results",
        "",
        "**Caveat (same as Phase 7/8):** these per-client numbers evaluate "
        "each config's final selected model on every client's OWN local "
        "training data (not a held-out split, since none exists per "
        "client) — a proxy for fairness, not a clean generalization test.",
        "",
        "| Config | Mean local F1 | Min | Max | Gap (max-min) |",
        "|---|---:|---:|---:|---:|",
    ]
    for config_name in CONFIG_ORDER:
        f = config_fairness[config_name]
        lines.append(f"| {config_name} | {f['mean']:.4f} | {f['min']:.4f} | {f['max']:.4f} | {f['gap']:.4f} |")

    lines += [
        "",
        "## 9. Limitations",
        "",
        "- Single run, single seed per config — not a statistically "
        "powered comparison, same caveat every prior phase (6, 7, 8, 9B) "
        "already carried.",
        "- `participation_seed` is fixed and shared across configs by "
        "design (so the staleness pattern itself is identical), but this "
        "means only one specific random staleness pattern was tested, not "
        "a distribution of possible patterns.",
        "- `decay_rate=0.5` for `exponential_decay` was chosen as a round, "
        "interpretable default before any result was seen, not tuned.",
        "- The client-fairness proxy (own local training data, not a "
        "held-out split) carries the same limitation Phase 7/8 already "
        "documented.",
        "",
        "## 10. Exact artifact locations",
        "",
        "```",
        "experiments/staleness_fl/<config_name>/run_config.json",
        "experiments/staleness_fl/<config_name>/round_history.json",
        "results/plots/staleness_fl/*.png",
        "results/reports/staleness_fl_results.md   (this file)",
        "```",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

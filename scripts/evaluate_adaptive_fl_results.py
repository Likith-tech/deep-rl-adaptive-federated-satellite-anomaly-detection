"""Phase 8 — evaluate the rule-based adaptive FedAvg ablation study.

Run from the repository root, after scripts/run_adaptive_fl_experiments.py
has produced experiments/adaptive_fl/<rule_name>/ for every rule config:

    python scripts/evaluate_adaptive_fl_results.py

For each rule config: evaluates the selected best-validation-round
global model ONCE on KDDTest+ (never used for rule selection), computes
a client-fairness proxy (the model evaluated on each client's own local
data — same caveat as Phase 7: not a held-out split, since none
exists), and compares aggregation-weight behavior against Phase 6's
uniform sample-count weighting.

Writes:
    results/plots/adaptive_fl/*.png
    results/reports/adaptive_fl_results.md

Phase 6's own report/plots/models are only READ (for the fixed
reference numbers below) — never modified.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import pandas/sklearn before torch/matplotlib — see the note in
# scripts/evaluate_local_models.py (Phase 5) about avoiding an
# intermittent access-violation crash on this Windows/Python 3.14 env.
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.training.baseline_trainer import load_features_and_labels  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.baseline_mlp import BaselineMLP  # noqa: E402

EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "adaptive_fl"
PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "adaptive_fl"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "adaptive_fl_results.md"

RULE_ORDER = ["performance_only", "resource_only", "data_only", "combined"]

# Phase 6 baseline (standard FedAvg, data/partitions/, alpha=0.5) —
# fixed reference values, read from the committed report, not
# re-derived or modified here.
PHASE_6_BASELINE = {
    "val_f1": 0.9879,
    "test": {"accuracy": 0.7531, "precision": 0.9219, "recall": 0.6186,
              "f1": 0.7404, "roc_auc": 0.8479, "false_positive_rate": 0.0693},
    "num_model_parameters": 23937,
    "total_communication_bytes_all_rounds": 15319680,
}


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


def plot_val_f1_comparison(rule_results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rule_name in RULE_ORDER:
        rh = rule_results[rule_name]["round_history"]
        rounds = [r["round_number"] for r in rh]
        f1s = [r["val_metrics"]["f1"] for r in rh]
        ax.plot(rounds, f1s, marker="o", label=rule_name, alpha=0.85)
    ax.axhline(PHASE_6_BASELINE["val_f1"], color="gray", linestyle="--", alpha=0.6,
               label=f"Phase 6 FedAvg (final={PHASE_6_BASELINE['val_f1']:.4f})")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global validation F1")
    ax.set_title("Validation F1 per Round — Adaptive Rules vs. Phase 6 FedAvg")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_val_loss_comparison(rule_results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for rule_name in RULE_ORDER:
        rh = rule_results[rule_name]["round_history"]
        rounds = [r["round_number"] for r in rh]
        losses = [r["val_loss"] for r in rh]
        ax.plot(rounds, losses, marker="o", label=rule_name, alpha=0.85)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global validation loss")
    ax.set_title("Validation Loss per Round — Adaptive Rules")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_aggregation_weight_distribution(rule_name: str, round_history: list[dict], out_path: Path) -> None:
    client_ids = sorted(round_history[0]["aggregation_weights"].keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cid in client_ids:
        weights = [r["aggregation_weights"][cid] for r in round_history]
        ax.plot([r["round_number"] for r in round_history], weights, marker="o", label=cid, alpha=0.8)
    uniform = 1.0 / len(client_ids)
    ax.axhline(uniform, color="gray", linestyle="--", alpha=0.5, label=f"Uniform (1/{len(client_ids)})")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Aggregation weight")
    ax.set_title(f"{rule_name} — Per-Client Aggregation Weight by Round")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_client_contribution_range(rule_results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = []
    for rule_name in RULE_ORDER:
        rh = rule_results[rule_name]["round_history"]
        all_weights = [w for r in rh for w in r["aggregation_weights"].values()]
        data.append(all_weights)
    ax.boxplot(data, tick_labels=RULE_ORDER)
    n_clients = len(rule_results[RULE_ORDER[0]]["round_history"][0]["aggregation_weights"])
    ax.axhline(1.0 / n_clients, color="gray", linestyle="--", alpha=0.5, label="Uniform")
    ax.set_ylabel("Aggregation weight (all clients, all rounds)")
    ax.set_title("Client Contribution Spread by Rule")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_test_f1_comparison(rule_test_f1: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = RULE_ORDER + ["phase6_fedavg"]
    values = [rule_test_f1[r] for r in RULE_ORDER] + [PHASE_6_BASELINE["test"]["f1"]]
    colors = ["#3b82f6"] * len(RULE_ORDER) + ["#6b7280"]
    ax.bar(names, values, color=colors)
    ax.set_ylabel("Final KDDTest+ F1")
    ax.set_title("Final KDDTest+ F1 — Adaptive Rules vs. Phase 6 FedAvg")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rule_results = {}
    rule_test_metrics = {}
    rule_fairness = {}

    for rule_name in RULE_ORDER:
        rule_dir = EXPERIMENT_ROOT / rule_name
        if not (rule_dir / "best_global_model.pt").exists():
            print(f"ERROR: {rule_dir}/best_global_model.pt not found. "
                  "Run scripts/run_adaptive_fl_experiments.py first.", file=sys.stderr)
            sys.exit(1)

        run_config = json.loads((rule_dir / "run_config.json").read_text())
        round_history = json.loads((rule_dir / "round_history.json").read_text())["round_history"]
        checkpoint = torch.load(rule_dir / "best_global_model.pt", weights_only=False)
        model, feature_columns = load_model(checkpoint, device)

        test_metrics = evaluate_on_parquet(model, TEST_PARQUET, feature_columns, device)
        print(f"{rule_name}: KDDTest+ f1={test_metrics.f1:.4f} accuracy={test_metrics.accuracy:.4f}")

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

        plot_aggregation_weight_distribution(rule_name, round_history, PLOTS_DIR / f"{rule_name}_weights_by_round.png")

        rule_results[rule_name] = {"run_config": run_config, "round_history": round_history}
        rule_test_metrics[rule_name] = test_metrics
        rule_fairness[rule_name] = fairness

    plot_val_f1_comparison(rule_results, PLOTS_DIR / "val_f1_comparison.png")
    plot_val_loss_comparison(rule_results, PLOTS_DIR / "val_loss_comparison.png")
    plot_client_contribution_range(rule_results, PLOTS_DIR / "client_contribution_range.png")
    plot_test_f1_comparison({r: rule_test_metrics[r].f1 for r in RULE_ORDER}, PLOTS_DIR / "test_f1_vs_phase6.png")

    write_report(rule_results, rule_test_metrics, rule_fairness)
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote plots to {PLOTS_DIR}")


def write_report(rule_results: dict, rule_test_metrics: dict, rule_fairness: dict) -> None:
    lines = [
        "# Phase 8 — Rule-Based Adaptive Federated Learning Results",
        "",
        "## Main comparison table",
        "",
        "| Rule | Val F1 (best round) | Test F1 | Test ROC-AUC | Client fairness gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for rule_name in RULE_ORDER:
        val_f1 = rule_results[rule_name]["run_config"]["best_val_metrics"]["f1"]
        m = rule_test_metrics[rule_name]
        roc = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "not available"
        gap = rule_fairness[rule_name]["gap"]
        lines.append(f"| {rule_name} | {val_f1:.4f} | {m.f1:.4f} | {roc} | {gap:.4f} |")
    lines.append(
        f"| **phase6_fedavg (reference)** | {PHASE_6_BASELINE['val_f1']:.4f} | "
        f"{PHASE_6_BASELINE['test']['f1']:.4f} | {PHASE_6_BASELINE['test']['roc_auc']:.4f} | not measured in Phase 6 |"
    )

    lines += [
        "",
        "## 1. Purpose",
        "",
        "Establish a transparent, DETERMINISTIC (not reinforcement-learned) "
        "rule-based adaptive FedAvg baseline, sitting between Phase 6 "
        "(standard FedAvg) and the future Phase 9 (DRL-based adaptive FL). "
        "Every client still participates every round — this phase changes "
        "ONLY how much each client's update counts toward the aggregated "
        "global model, via a fixed, documented arithmetic rule with no "
        "learned parameters and no reward signal.",
        "",
        "## 2. Why Phase 6 was insufficient",
        "",
        "Phase 6 weights every client purely by local sample count. Phase 7 "
        "showed that non-IID heterogeneity barely moves the overall KDDTest+ "
        "score but devastates PER-CLIENT fairness (~41x gap between the "
        "best- and worst-served satellite across tested alphas). Sample-"
        "count weighting has no mechanism to respond to that — a large but "
        "unhelpful or narrow-category client gets the same automatic "
        "priority as a small, diverse, currently-useful one.",
        "",
        "## 3. Adaptive rule / formula",
        "",
        "```",
        "client_score_k = w_perf * performance_k + w_data * data_k",
        "               + w_resource * resource_k + w_fair * fairness_k",
        "```",
        "",
        "All four raw signals are min-max normalized to [0, 1] across the "
        "participating clients each round (fairness/data/resource are "
        "static per experiment; performance is recomputed every round from "
        "that round's local update, evaluated on global validation). Scores "
        "are renormalized to sum to 1 and used directly as FedAvg "
        "aggregation weights (`src/federated/server.py:aggregate_with_weights`, "
        "additive on top of the unmodified Phase 6 `aggregate` method). Full "
        "formula and component rationale in `src/federated/adaptive.py` and "
        "`configs/adaptive_fl.yaml`.",
        "",
        "**Component rationale:**",
        "- **performance**: this round's local update's F1 on global "
        "validation — rewards updates currently helping the shared model.",
        "- **data**: local training sample count — the classic FedAvg "
        "intuition, now one signal among several.",
        "- **resource**: simulated Phase 4 conditions (bandwidth, compute, "
        "availability, connectivity — higher better; latency inverted). "
        "SIMULATED, not real satellite telemetry.",
        "- **fairness**: Shannon entropy of the client's local "
        "attack-category distribution (same function as Phase 7) — "
        "explicitly counteracts high performers drowning out clients "
        "holding rare categories (r2l/u2r).",
        "",
        "## 4. Client-selection or weighting mechanism",
        "",
        "AGGREGATION WEIGHTING, not selection — all 8 clients participate "
        "every round for every rule config, chosen deliberately as the "
        "safer, more defensible mechanism (hard selection risks silently "
        "dropping a rare-category client for an entire round). A top-k "
        "selection utility is implemented and unit-tested "
        "(`select_top_k_clients`) for later reuse but not separately "
        "benchmarked in this phase's experiments.",
        "",
        "## 5. Experimental configuration",
        "",
        "Identical to Phase 6 in every respect except the aggregation "
        "rule: 8 clients, the SAME `data/partitions/` (Phase 4 original, "
        "alpha=0.5) partition — not a fresh one — 10 rounds, 1 local "
        "epoch, batch size 128, learning rate 0.001, Adam, "
        "BCEWithLogitsLoss, seed 42, shared initial weights (seed 42). "
        "Rule weights (section 3) were fixed BEFORE any KDDTest+ "
        "evaluation ran.",
        "",
        "## 6. Validation results (best round per rule)",
        "",
        "| Rule | Best round | Val F1 | Val loss |",
        "|---|---:|---:|---:|",
    ]
    for rule_name in RULE_ORDER:
        rc = rule_results[rule_name]["run_config"]
        lines.append(f"| {rule_name} | {rc['best_round']} | {rc['best_val_metrics']['f1']:.4f} | {rc['best_val_loss']:.4f} |")

    lines += [
        "",
        "## 7. Final KDDTest+ results",
        "",
        "Evaluated ONCE per rule config, after round selection by "
        "validation loss only:",
        "",
        "| Rule | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rule_name in RULE_ORDER:
        m = rule_test_metrics[rule_name]
        roc = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "not available"
        lines.append(f"| {rule_name} | {m.accuracy:.4f} | {m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | {roc} | {m.false_positive_rate:.4f} |")
    lines.append(
        f"| phase6_fedavg (reference) | {PHASE_6_BASELINE['test']['accuracy']:.4f} | "
        f"{PHASE_6_BASELINE['test']['precision']:.4f} | {PHASE_6_BASELINE['test']['recall']:.4f} | "
        f"{PHASE_6_BASELINE['test']['f1']:.4f} | {PHASE_6_BASELINE['test']['roc_auc']:.4f} | "
        f"{PHASE_6_BASELINE['test']['false_positive_rate']:.4f} |"
    )

    lines += [
        "",
        "## 8. FedAvg vs. Adaptive FL comparison",
        "",
    ]
    combined_f1 = rule_test_metrics["combined"].f1
    delta = combined_f1 - PHASE_6_BASELINE["test"]["f1"]
    if delta > 0:
        comparison_sentence = f"The 'combined' rule improved on Phase 6 FedAvg's KDDTest+ F1 by {delta:+.4f}."
    elif delta == 0:
        comparison_sentence = "The 'combined' rule exactly matched Phase 6 FedAvg's KDDTest+ F1."
    else:
        comparison_sentence = (
            f"The 'combined' rule did NOT beat Phase 6 FedAvg's KDDTest+ F1 — "
            f"{abs(delta):.4f} lower ({combined_f1:.4f} vs {PHASE_6_BASELINE['test']['f1']:.4f})."
        )
    lines.append(comparison_sentence + " Reported exactly as measured — the rule weights were not adjusted after seeing this number.")

    # Honest, explicit callout of the actual best/worst rules by each
    # metric — do not let a counterintuitive result hide in a table.
    best_test_rule = max(RULE_ORDER, key=lambda r: rule_test_metrics[r].f1)
    best_fair_rule = min(RULE_ORDER, key=lambda r: rule_fairness[r]["gap"])
    lines += [
        "",
        "**Observed pattern, reported exactly as measured:** the rule with "
        f"the highest KDDTest+ F1 was **{best_test_rule}** "
        f"({rule_test_metrics[best_test_rule].f1:.4f}), not `combined`. The "
        f"rule with the SMALLEST fairness gap was **{best_fair_rule}** "
        f"({rule_fairness[best_fair_rule]['gap']:.4f}) — "
        + (
            "which is the fairness-weighted `combined` rule, as intended."
            if best_fair_rule == "combined"
            else "NOT the fairness-weighted `combined` rule, which is a "
            "genuinely counterintuitive result we are not smoothing over: "
            "explicitly weighting for local category diversity did not "
            "produce the best-measured fairness outcome in this single-seed "
            "run. A plausible explanation is that `performance` (evaluated "
            "on the shared global validation set) already implicitly "
            "favors clients whose updates generalize well, which can "
            "correlate with diversity in ways the static entropy signal "
            "does not fully capture round-to-round — but with one seed "
            "and four rule configs we cannot separate a genuine effect "
            "from noise here."
        ),
        "",
        "## 9. Fairness results",
        "",
        "**Caveat (same as Phase 7):** these per-client numbers evaluate "
        "each rule's final selected model on every client's OWN local "
        "training data — a proxy for how well the model fits that "
        "client's distribution, not a held-out generalization test.",
        "",
        "| Rule | Mean F1 | Min F1 | Max F1 | Std Dev | Max-Min Gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for rule_name in RULE_ORDER:
        f = rule_fairness[rule_name]
        lines.append(f"| {rule_name} | {f['mean']:.4f} | {f['min']:.4f} | {f['max']:.4f} | {f['stdev']:.4f} | {f['gap']:.4f} |")

    best_fairness_rule = min(RULE_ORDER, key=lambda r: rule_fairness[r]["gap"])
    worst_fairness_rule = max(RULE_ORDER, key=lambda r: rule_fairness[r]["gap"])
    lines += [
        "",
        f"Smallest fairness gap: **{best_fairness_rule}** ({rule_fairness[best_fairness_rule]['gap']:.4f}). "
        f"Largest: **{worst_fairness_rule}** ({rule_fairness[worst_fairness_rule]['gap']:.4f}). "
        "Since all clients participate every round in every rule (weighting, "
        "not selection), category coverage in the aggregate is 100% for "
        "every rule — no client's categories are ever entirely excluded; "
        "the fairness question here is about DEGREE of influence, not "
        "presence/absence.",
        "",
        "## 10. Communication comparison",
        "",
        f"Identical to Phase 6 for every rule: {PHASE_6_BASELINE['num_model_parameters']:,} parameters, "
        f"all 8 clients participate every round, 10 rounds — same "
        f"~{PHASE_6_BASELINE['total_communication_bytes_all_rounds'] / (1024 * 1024):.2f} MB total estimated "
        "transfer as Phase 6. Aggregation weighting changes HOW MUCH each "
        "update counts, not whether it is transmitted — communication cost "
        "would only drop if selection (not weighting) were used instead.",
        "",
        "## 11. Ablation results",
        "",
        "Four rule configs were run: performance_only, resource_only, "
        "data_only (each isolating one signal), and combined (the "
        "intended primary rule). See the main comparison table above and "
        "`results/plots/adaptive_fl/val_f1_comparison.png` / "
        "`client_contribution_range.png` for the visual comparison. Client "
        "SELECTION (as opposed to weighting) was implemented and "
        "unit-tested but not separately benchmarked — see section 4.",
        "",
        "## 12. Limitations",
        "",
        "- Rule-based only — NOT reinforcement learning; no learned "
        "parameters, no reward signal.",
        "- Single seed (42) per rule config — not a statistically powered "
        "comparison.",
        "- Fairness metric uses each client's own local training data as a "
        "proxy (no separate held-out per-client split exists).",
        "- Resource signals are simulated (Phase 4), not real satellite "
        "telemetry.",
        "- All clients participate every round in every rule — this phase "
        "did not benchmark hard client selection.",
        "- NSL-KDD remains terrestrial network data; satellite framing is "
        "simulated throughout.",
        "",
        "## 13. Why these results motivate adaptive Federated Learning",
        "",
        "Whatever pattern emerges above (reported honestly, not "
        "cherry-picked), this phase's main contribution is the "
        "infrastructure and evidence needed before Phase 9: a working, "
        "tested, non-RL scoring/weighting mechanism plumbed through the "
        "exact same client/server/FedAvg code Phase 6 and Phase 9 both "
        "use, with real measured numbers for what simple, transparent "
        "rules can and cannot achieve. Phase 9's DRL controller will be "
        "compared against THIS baseline, not just against Phase 6.",
        "",
        "## 14. What Phase 9 will do",
        "",
        "Replace the fixed rule weights with a DRL (DQN) controller that "
        "learns client-weighting/selection decisions from a reward signal, "
        "using the same signals (performance, data, resource, fairness) "
        "this phase established as inputs — but LEARNED rather than "
        "hand-specified.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

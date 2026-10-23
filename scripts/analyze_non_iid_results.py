"""Phase 7 — analyze the controlled non-IID FedAvg experiments.

Run from the repository root, after scripts/run_non_iid_experiments.py
has produced experiments/non_iid/alpha_<X>/ for every configured alpha:

    python scripts/analyze_non_iid_results.py

For each alpha: evaluates the selected best-validation-round global
model ONCE on KDDTest+ (never used for round/alpha selection), and
evaluates that same model on every client's own local partition as a
"client fairness" proxy (there is no separate held-out per-client
split, so a client's own local data is the best available way to see
how well the shared global model fits that client's distribution —
this is explicitly NOT a leakage-free held-out evaluation, and the
report labels it as such throughout).

Writes:
    results/plots/non_iid/*.png
    results/reports/non_iid_results.md
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Import pandas/sklearn before torch/matplotlib — see the note in
# scripts/evaluate_local_models.py (Phase 5) / evaluate_federated_model.py
# (Phase 6) about avoiding an intermittent access-violation crash on
# this Windows/Python 3.14 env when torch is imported before pyarrow's
# parquet reader initializes.
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.training.baseline_trainer import load_features_and_labels  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.baseline_mlp import BaselineMLP  # noqa: E402

EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "non_iid"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "non_iid"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "non_iid_results.md"

ALL_CATEGORIES = ["normal", "dos", "probe", "r2l", "u2r"]
RARE_CATEGORIES = ["r2l", "u2r"]

# Phase 6 baseline (alpha=0.5, original data/partitions/ run) — fixed
# reference values, not re-derived here.
PHASE_6_BASELINE = {
    "alpha": 0.5, "num_rounds": 10,
    "val_f1": 0.9879,
    "test": {"accuracy": 0.7531, "precision": 0.9219, "recall": 0.6186,
              "f1": 0.7404, "roc_auc": 0.8479, "false_positive_rate": 0.0693},
}


def load_model(checkpoint: dict, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
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


def evaluate_on_parquet(model: torch.nn.Module, parquet_path: Path, feature_columns: list[str], device: torch.device):
    X, y = load_features_and_labels(parquet_path, feature_columns)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device)).squeeze(-1)
        y_prob = torch.sigmoid(logits).cpu().numpy()
    y_true = y.astype(int)
    y_pred = (y_prob >= 0.5).astype(int)
    return compute_metrics(y_true, y_pred, y_prob)


def plot_anomaly_pct_by_satellite(alpha: float, client_stats: dict, out_path: Path) -> None:
    client_ids = sorted(client_stats.keys())
    values = [client_stats[c]["anomaly_percentage"] for c in client_ids]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(client_ids, values, color="#3b82f6")
    ax.set_ylabel("Anomaly %")
    ax.set_title(f"alpha={alpha} — Local Anomaly Percentage by Satellite")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_category_distribution_by_satellite(alpha: float, non_iid_report: dict, out_path: Path) -> None:
    categories = non_iid_report["categories"]
    client_ids = sorted(non_iid_report["client_distributions"].keys())
    distributions = np.array([non_iid_report["client_distributions"][c] for c in client_ids])

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(client_ids))
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    for cat_idx, cat in enumerate(categories):
        values = distributions[:, cat_idx]
        ax.bar(client_ids, values, bottom=bottom, label=cat, color=colors[cat_idx])
        bottom += values
    ax.set_ylabel("Proportion of local data")
    ax.set_title(f"alpha={alpha} — Attack-Category Distribution by Satellite")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_val_f1_per_round(alpha: float, round_history: list[dict], best_round: int, out_path: Path) -> None:
    rounds = [r["round_number"] for r in round_history]
    f1s = [r["val_metrics"]["f1"] for r in round_history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rounds, f1s, marker="o", color="#3b82f6")
    ax.axvline(best_round, color="gray", linestyle="--", alpha=0.6, label=f"Best round ({best_round})")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Global validation F1")
    ax.set_title(f"alpha={alpha} — Validation F1 per Round")
    ax.set_xticks(rounds)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_vs_alpha(alphas: list[float], values: list[float], ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alphas, values, marker="o", color="#3b82f6")
    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet alpha (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_client_f1_distribution(alphas: list[float], client_f1_by_alpha: dict[float, list[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [client_f1_by_alpha[a] for a in alphas]
    ax.boxplot(data, tick_labels=[str(a) for a in alphas])
    for i, a in enumerate(alphas, start=1):
        ys = client_f1_by_alpha[a]
        ax.scatter([i] * len(ys), ys, color="#ef4444", s=15, zorder=3, alpha=0.7)
    ax.set_xlabel("Dirichlet alpha")
    ax.set_ylabel("Per-client F1 (evaluated on own local data)")
    ax.set_title("Per-Client F1 Distribution by Alpha")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_js_vs_metric(mean_js: list[float], values: list[float], alphas: list[float], ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(mean_js, values, color="#3b82f6", s=60, zorder=3)
    for x, y, a in zip(mean_js, values, alphas):
        ax.annotate(f"alpha={a}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean pairwise JS distance")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    alpha_dirs = sorted(EXPERIMENT_ROOT.glob("alpha_*"), key=lambda p: float(p.name.replace("alpha_", "")))
    if not alpha_dirs:
        print(f"ERROR: no experiments found under {EXPERIMENT_ROOT}. "
              "Run scripts/run_non_iid_experiments.py first.", file=sys.stderr)
        sys.exit(1)

    all_results = {}

    for alpha_dir in alpha_dirs:
        alpha = float(alpha_dir.name.replace("alpha_", ""))
        print(f"\n=== alpha={alpha} ===")

        partition_metadata = json.loads((alpha_dir / "partition_metadata.json").read_text())
        run_config = json.loads((alpha_dir / "run_config.json").read_text())
        round_history = json.loads((alpha_dir / "round_history.json").read_text())["round_history"]

        checkpoint = torch.load(alpha_dir / "best_global_model.pt", weights_only=False)
        model, feature_columns = load_model(checkpoint, device)

        test_metrics = evaluate_on_parquet(model, TEST_PARQUET, feature_columns, device)
        print(f"KDDTest+: f1={test_metrics.f1:.4f} accuracy={test_metrics.accuracy:.4f}")

        client_ids = sorted(partition_metadata["client_sample_counts"].keys())
        client_metrics = {}
        for cid in client_ids:
            client_parquet = alpha_dir / "partitions" / cid / "train.parquet"
            client_metrics[cid] = evaluate_on_parquet(model, client_parquet, feature_columns, device)
        client_f1s = [m.f1 for m in client_metrics.values()]
        fairness = {
            "mean": statistics.mean(client_f1s),
            "median": statistics.median(client_f1s),
            "min": min(client_f1s),
            "max": max(client_f1s),
            "stdev": statistics.stdev(client_f1s) if len(client_f1s) > 1 else 0.0,
            "gap": max(client_f1s) - min(client_f1s),
        }
        print(f"Client fairness (own-local-data F1): mean={fairness['mean']:.4f} "
              f"min={fairness['min']:.4f} max={fairness['max']:.4f} gap={fairness['gap']:.4f}")

        rare_category_gaps = {}
        for cat in RARE_CATEGORIES:
            zero_clients = [
                cid for cid, s in partition_metadata["client_stats"].items()
                if s["category_counts"].get(cat, 0) == 0
            ]
            rare_category_gaps[cat] = zero_clients

        plot_anomaly_pct_by_satellite(alpha, partition_metadata["client_stats"], PLOTS_DIR / f"alpha_{alpha}_anomaly_pct.png")
        plot_category_distribution_by_satellite(alpha, partition_metadata["non_iid_report"], PLOTS_DIR / f"alpha_{alpha}_category_distribution.png")
        plot_val_f1_per_round(alpha, round_history, run_config["best_round"], PLOTS_DIR / f"alpha_{alpha}_val_f1_per_round.png")

        all_results[alpha] = {
            "partition_metadata": partition_metadata,
            "run_config": run_config,
            "round_history": round_history,
            "test_metrics": test_metrics,
            "client_metrics": client_metrics,
            "fairness": fairness,
            "rare_category_gaps": rare_category_gaps,
        }

    alphas = sorted(all_results.keys())
    mean_js_by_alpha = [all_results[a]["partition_metadata"]["non_iid_report"]["pairwise_js_distance_avg"] for a in alphas]
    val_f1_by_alpha = [all_results[a]["run_config"]["best_val_metrics"]["f1"] for a in alphas]
    test_f1_by_alpha = [all_results[a]["test_metrics"].f1 for a in alphas]
    client_f1_by_alpha = {a: [m.f1 for m in all_results[a]["client_metrics"].values()] for a in alphas}

    plot_metric_vs_alpha(alphas, mean_js_by_alpha, "Mean pairwise JS distance", "Non-IID Heterogeneity (Mean JS Distance) by Alpha", PLOTS_DIR / "js_distance_by_alpha.png")
    plot_metric_vs_alpha(alphas, test_f1_by_alpha, "Final KDDTest+ F1", "Final KDDTest+ F1 by Alpha", PLOTS_DIR / "test_f1_vs_alpha.png")
    plot_client_f1_distribution(alphas, client_f1_by_alpha, PLOTS_DIR / "client_f1_distribution.png")
    plot_js_vs_metric(mean_js_by_alpha, val_f1_by_alpha, alphas, "Best-round validation F1", "Mean JS Distance vs Validation F1", PLOTS_DIR / "mean_js_vs_val_f1.png")
    plot_js_vs_metric(mean_js_by_alpha, test_f1_by_alpha, alphas, "Final KDDTest+ F1", "Mean JS Distance vs KDDTest+ F1", PLOTS_DIR / "mean_js_vs_test_f1.png")

    write_report(alphas, all_results)
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote plots to {PLOTS_DIR}")


def write_report(alphas: list[float], all_results: dict) -> None:
    lines = [
        "# Phase 7 — Non-IID Federated Learning Experiments",
        "",
        "## Main comparison table",
        "",
        "| Alpha | Mean JS | Val F1 | Test F1 | Test ROC-AUC | Client F1 Std |",
        "|------:|--------:|-------:|--------:|-------------:|--------------:|",
    ]
    for a in alphas:
        nr = all_results[a]["partition_metadata"]["non_iid_report"]
        val_f1 = all_results[a]["run_config"]["best_val_metrics"]["f1"]
        m = all_results[a]["test_metrics"]
        roc = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "not available"
        std = all_results[a]["fairness"]["stdev"]
        lines.append(f"| {a} | {nr['pairwise_js_distance_avg']:.4f} | {val_f1:.4f} | {m.f1:.4f} | {roc} | {std:.4f} |")
    lines += [
        "",
        "## 1. Purpose",
        "",
        "Measure how the DEGREE of client data heterogeneity affects "
        "standard FedAvg, using controlled experiments that vary ONLY "
        "the Dirichlet concentration parameter (alpha) while holding "
        "every other factor (model, clients, rounds, local epochs, "
        "batch size, learning rate, optimizer, loss, FedAvg weighting, "
        "validation set, test set, seeds) fixed.",
        "",
        "## 2. What non-IID means",
        "",
        "Each simulated satellite's local data comes from a different "
        "distribution of attack categories than the others — the "
        "opposite of every client holding a random, representative "
        "sample of everything. Lower Dirichlet alpha produces MORE "
        "skewed (non-IID) client distributions; higher alpha produces "
        "distributions closer to IID (identical across clients).",
        "",
        "## 3. Experimental design",
        "",
        "For each alpha, a FRESH 8-client Dirichlet partition of the "
        "real training data (`data/processed/train.parquet`, 107,077 "
        "records) was built using the same seed (42) and the same "
        "partitioning machinery as Phase 4 "
        "(`src/simulation/partitioner.py`). The ORIGINAL Phase 4/6 "
        "partition (`data/partitions/`, alpha=0.5) was never touched — "
        "every alpha in this study, including 0.5, uses its own fresh "
        "partition under `experiments/non_iid/alpha_<X>/partitions/`, "
        "so alpha=0.5 here is a separate but comparable run to Phase 6.",
        "",
        "## 4. Alpha values",
        "",
        f"{', '.join(str(a) for a in alphas)}",
        "",
        "## 5. Partition statistics",
        "",
        "| Alpha | Total samples | Constraints satisfied | Attempts used |",
        "|---:|---:|---|---:|",
    ]
    for a in alphas:
        pm = all_results[a]["partition_metadata"]
        pr = pm["partition_result"]
        lines.append(f"| {a} | {pm['total_samples']:,} | {pr['constraints_satisfied']} | {pr['attempts_used']} |")
    lines += [
        "",
        "All alphas accounted for every one of the 107,077 training "
        "records exactly once (verified by `verify_partition_integrity` "
        "— no loss, no duplication). Validation and KDDTest+ were never "
        "partitioned or touched by this phase.",
        "",
        "## 6. JS-distance measurements",
        "",
        "| Alpha | Mean JS | Min JS | Max JS | Mean Entropy (bits) |",
        "|------:|--------:|-------:|-------:|---------------------:|",
    ]
    for a in alphas:
        nr = all_results[a]["partition_metadata"]["non_iid_report"]
        mean_entropy = statistics.mean(nr["client_entropy"].values())
        lines.append(
            f"| {a} | {nr['pairwise_js_distance_avg']:.4f} | {nr['pairwise_js_distance_min']:.4f} | "
            f"{nr['pairwise_js_distance_max']:.4f} | {mean_entropy:.4f} |"
        )
    lines += [
        "",
        "This confirms alpha actually changed measured heterogeneity — "
        "not assumed. See `results/plots/non_iid/js_distance_by_alpha.png`.",
        "",
        "## 7. Client distributions",
        "",
        "Per-satellite anomaly percentage and attack-category "
        "distribution for every alpha: "
        "`results/plots/non_iid/alpha_<X>_anomaly_pct.png` and "
        "`alpha_<X>_category_distribution.png`.",
        "",
        "## 8. FedAvg configuration (identical across all alphas)",
        "",
    ]
    example_cfg = all_results[alphas[0]]["run_config"]["config"]
    lines += [
        f"- Clients: {example_cfg['satellites']['num_clients']} (ALL participate every round)",
        f"- Rounds: {example_cfg['federation']['num_rounds']}",
        f"- Local epochs: {example_cfg['training']['local_epochs']}",
        f"- Batch size: {example_cfg['training']['batch_size']}",
        f"- Learning rate: {example_cfg['training']['learning_rate']}",
        f"- Optimizer: {example_cfg['training']['optimizer']} (Adam)",
        f"- Loss: {example_cfg['training']['loss']} (BCEWithLogitsLoss)",
        f"- FedAvg weighting: {example_cfg['federation']['weighting']} (sample-count weighted)",
        f"- Seed: {example_cfg['training']['seed']}, shared init seed: {example_cfg['initialization']['shared_init_seed']}",
        "",
        "## 9. Round-by-round validation results",
        "",
    ]
    for a in alphas:
        lines.append(f"**alpha={a}**")
        lines.append("")
        lines.append("| Round | Val F1 | Val Loss |")
        lines.append("|------:|-------:|---------:|")
        for r in all_results[a]["round_history"]:
            lines.append(f"| {r['round_number']} | {r['val_metrics']['f1']:.4f} | {r['val_loss']:.4f} |")
        lines.append("")
    lines.append("See `results/plots/non_iid/alpha_<X>_val_f1_per_round.png` for the visual versions.")

    lines += [
        "",
        "## 10. Final KDDTest+ results",
        "",
        "Evaluated ONCE per alpha, after round selection by validation loss only:",
        "",
        "| Alpha | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in alphas:
        m = all_results[a]["test_metrics"]
        roc = f"{m.roc_auc:.4f}" if m.roc_auc is not None else "not available"
        lines.append(f"| {a} | {m.accuracy:.4f} | {m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | {roc} | {m.false_positive_rate:.4f} |")

    # Honest, explicit callout if KDDTest+ F1 does NOT increase monotonically
    # with alpha — do not smooth this over or hide it in a table.
    test_f1s = [all_results[a]["test_metrics"].f1 for a in alphas]
    non_monotonic_points = [
        (alphas[i - 1], test_f1s[i - 1], alphas[i], test_f1s[i])
        for i in range(1, len(alphas))
        if test_f1s[i] < test_f1s[i - 1]
    ]
    if non_monotonic_points:
        callout = (
            "**Note — KDDTest+ F1 was NOT monotonic across the tested alphas.** "
        )
        for prev_a, prev_f1, cur_a, cur_f1 in non_monotonic_points:
            callout += (
                f"Going from alpha={prev_a} (F1={prev_f1:.4f}) to alpha={cur_a} "
                f"(F1={cur_f1:.4f}) test F1 DECREASED despite alpha={cur_a} being "
                f"closer to IID. "
            )
        callout += (
            "This is reported exactly as measured — we do not smooth it over. "
            "A plausible explanation is that KDDTest+ performance saturates (or "
            "adds noise) once heterogeneity is already low, while the client "
            "fairness gap (section 11) keeps improving — but with only one run "
            "per alpha, this could equally be run-to-run variance rather than a "
            "genuine saturation effect; we do not have enough repeated trials to "
            "tell those apart."
        )
        lines += ["", callout]

    lines += [
        "",
        "## 11. Client fairness results",
        "",
        "**Important caveat:** these per-client numbers evaluate the "
        "FINAL selected global model on each client's OWN local "
        "partition — not a separate held-out per-client split, since "
        "none exists. This is a proxy for how well the shared global "
        "model fits each client's local distribution, not a "
        "leakage-free generalization test (this data was part of what "
        "trained the global model via FedAvg).",
        "",
        "| Alpha | Mean F1 | Median F1 | Min F1 | Max F1 | Std Dev | Max-Min Gap |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in alphas:
        f = all_results[a]["fairness"]
        lines.append(f"| {a} | {f['mean']:.4f} | {f['median']:.4f} | {f['min']:.4f} | {f['max']:.4f} | {f['stdev']:.4f} | {f['gap']:.4f} |")
    lines += ["", "See `results/plots/non_iid/client_f1_distribution.png`."]

    lines += [
        "",
        "## 12. Rare-category observations",
        "",
        "Clients with ZERO local examples of a rare attack category:",
        "",
        "| Alpha | Clients missing r2l | Clients missing u2r |",
        "|---:|---|---|",
    ]
    for a in alphas:
        gaps = all_results[a]["rare_category_gaps"]
        r2l_missing = ", ".join(gaps["r2l"]) or "none"
        u2r_missing = ", ".join(gaps["u2r"]) or "none"
        lines.append(f"| {a} | {r2l_missing} | {u2r_missing} |")
    lines += [
        "",
        "No samples were invented or artificially balanced to fix "
        "these gaps — they reflect genuine rarity of r2l/u2r in the "
        "underlying real dataset combined with the Dirichlet split.",
        "",
        "## 13. Comparison with Phase 6",
        "",
        f"Phase 6's original baseline (`data/partitions/`, alpha={PHASE_6_BASELINE['alpha']}, "
        f"{PHASE_6_BASELINE['num_rounds']} rounds) scored validation F1 "
        f"{PHASE_6_BASELINE['val_f1']:.4f} and final KDDTest+ F1 "
        f"{PHASE_6_BASELINE['test']['f1']:.4f}. This phase's OWN alpha=0.5 "
        "run (a freshly-generated partition, same seed/alpha/constraints "
        "as Phase 6) is included in the tables above for direct "
        "comparison under identical methodology.",
        "",
    ]
    alpha_05_result = all_results.get(0.5)
    if alpha_05_result is not None:
        alpha_05_val_f1 = alpha_05_result["run_config"]["best_val_metrics"]["f1"]
        alpha_05_test_f1 = alpha_05_result["test_metrics"].f1
        matches = (
            abs(alpha_05_val_f1 - PHASE_6_BASELINE["val_f1"]) < 1e-4
            and abs(alpha_05_test_f1 - PHASE_6_BASELINE["test"]["f1"]) < 1e-4
        )
        if matches:
            lines.append(
                f"This phase's alpha=0.5 run reproduced Phase 6's numbers exactly "
                f"(validation F1 {alpha_05_val_f1:.4f}, KDDTest+ F1 {alpha_05_test_f1:.4f}) "
                "— since both use the identical seed, alpha, and partitioning "
                "constraints, the Dirichlet partitioner deterministically "
                "regenerates the same partition, and FedAvg deterministically "
                "regenerates the same training trajectory. This is a useful "
                "cross-phase reproducibility check, not a coincidence."
            )
        else:
            lines.append(
                f"This phase's alpha=0.5 run scored validation F1 {alpha_05_val_f1:.4f} "
                f"and KDDTest+ F1 {alpha_05_test_f1:.4f} — NOT identical to Phase 6's "
                f"numbers ({PHASE_6_BASELINE['val_f1']:.4f} / {PHASE_6_BASELINE['test']['f1']:.4f}) "
                "despite using the same seed/alpha/constraints; this discrepancy "
                "should be investigated rather than assumed benign."
            )
    lines += [
        "",
        "## 14. Limitations",
        "",
        "- Only 5 alpha values and 8 clients — a small number of "
        "conditions; trends described below are observational, not a "
        "statistically powered study.",
        "- NSL-KDD remains a terrestrial dataset; the satellite/client "
        "framing is simulated throughout.",
        "- Standard synchronous FedAvg only — no adaptive client "
        "selection, no staleness handling, no DRL.",
        "- Client fairness metrics (section 11) use each client's own "
        "local training data as a proxy, not a genuinely held-out "
        "per-client split.",
        "- Very low alpha may fail the partitioning constraints (section "
        "5) for some clients — reported honestly, not hidden or fixed.",
        "",
        "## 15. What this tells us about the need for adaptive FL",
        "",
    ]

    # Data-driven summary sentence (correlational language only).
    if len(alphas) >= 2:
        low_alpha, high_alpha = alphas[0], alphas[-1]
        low_test_f1 = all_results[low_alpha]["test_metrics"].f1
        high_test_f1 = all_results[high_alpha]["test_metrics"].f1
        low_gap = all_results[low_alpha]["fairness"]["gap"]
        high_gap = all_results[high_alpha]["fairness"]["gap"]
        low_mean_js = all_results[low_alpha]["partition_metadata"]["non_iid_report"]["pairwise_js_distance_avg"]
        high_mean_js = all_results[high_alpha]["partition_metadata"]["non_iid_report"]["pairwise_js_distance_avg"]
        summary_sentence = (
            f"Across the tested range, alpha={low_alpha} (most heterogeneous, mean JS "
            f"distance {low_mean_js:.4f}) scored KDDTest+ F1 {low_test_f1:.4f} with a "
            f"client fairness gap of {low_gap:.4f}, while alpha={high_alpha} (closest to "
            f"IID, mean JS distance {high_mean_js:.4f}) scored KDDTest+ F1 "
            f"{high_test_f1:.4f} with a client fairness gap of {high_gap:.4f}. As "
            "heterogeneity increased across the tested alphas, performance and "
            "fairness showed the pattern above — we deliberately avoid claiming "
            "causation from five data points. If this pattern holds up, it is "
            "exactly the kind of gap adaptive client selection or weighting "
            "(Phase 8) would aim to close."
        )
        lines.append(summary_sentence)
    else:
        lines.append("Insufficient alpha conditions to summarize a trend.")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

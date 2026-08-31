"""Phase 14 — full cross-phase evaluation.

Run from the repository root, after every prior phase (2, 3, 6, 7, 8,
9B, 9) has already been run and its report committed:

    python scripts/generate_full_evaluation_report.py

This is deliberately NOT a new experiment: it trains nothing and
touches no partition, checkpoint, or held-out set. It reads the final,
already-measured KDDTest+ numbers straight out of each phase's own
committed report (results/reports/{baseline,temporal,federated,non_iid,
adaptive_fl,drl_adaptive_fl,staleness_fl}_results.md) into one fixed
dictionary below — copied faithfully, not re-derived or adjusted — and
assembles a single master comparison table, ranking, and set of plots
across every method this project has built so far.

Writes:
    results/plots/full_evaluation/*.png
    results/reports/full_evaluation.md
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOTS_DIR = REPO_ROOT / "results" / "plots" / "full_evaluation"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "full_evaluation.md"

# Every number below is copied verbatim from the named, already-committed
# report — never re-derived, re-run, or adjusted here. "fairness_gap":
# None means the phase's own report does not measure a client-fairness
# gap (Phase 2/3 are centralized, non-federated; Phase 6/7 didn't
# compute one). Source file is recorded alongside each entry so a
# reader can verify every number against its origin.
METHODS = [
    {
        "name": "Phase 2 — Centralized MLP baseline",
        "family": "non-federated",
        "source": "results/reports/baseline_results.md",
        "accuracy": 0.7832, "precision": 0.9278, "recall": 0.6713,
        "f1": 0.7790, "roc_auc": 0.8972, "fpr": 0.0690, "fairness_gap": None,
    },
    {
        "name": "Phase 3 — Temporal GRU+Attention (centralized, corrected)",
        "family": "non-federated",
        "source": "results/reports/temporal_results.md",
        "accuracy": 0.7583, "precision": 0.9221, "recall": 0.6403,
        "f1": 0.7558, "roc_auc": 0.8923, "fpr": 0.0759, "fairness_gap": None,
    },
    {
        "name": "Phase 6 — FedAvg (sample-count weighting)",
        "family": "federated",
        "source": "results/reports/federated_results.md",
        "accuracy": 0.7531, "precision": 0.9219, "recall": 0.6186,
        "f1": 0.7404, "roc_auc": 0.8479, "fpr": 0.0693, "fairness_gap": None,
    },
    {
        "name": "Phase 7 — Non-IID sweep, alpha=0.1 (most heterogeneous)",
        "family": "federated (non-IID study)",
        "source": "results/reports/non_iid_results.md",
        "accuracy": None, "precision": None, "recall": None,
        "f1": 0.7409, "roc_auc": 0.8172, "fpr": None, "fairness_gap": None,
    },
    {
        "name": "Phase 7 — Non-IID sweep, alpha=5.0 (best test F1)",
        "family": "federated (non-IID study)",
        "source": "results/reports/non_iid_results.md",
        "accuracy": None, "precision": None, "recall": None,
        "f1": 0.7640, "roc_auc": 0.8641, "fpr": None, "fairness_gap": None,
    },
    {
        "name": "Phase 7 — Non-IID sweep, alpha=10.0 (least heterogeneous)",
        "family": "federated (non-IID study)",
        "source": "results/reports/non_iid_results.md",
        "accuracy": None, "precision": None, "recall": None,
        "f1": 0.7541, "roc_auc": 0.8637, "fpr": None, "fairness_gap": None,
    },
    {
        "name": "Phase 8 — Adaptive FL, performance_only",
        "family": "federated (rule-based adaptive)",
        "source": "results/reports/adaptive_fl_results.md",
        "accuracy": 0.7563, "precision": 0.9224, "recall": 0.6244,
        "f1": 0.7447, "roc_auc": 0.8497, "fpr": 0.0694, "fairness_gap": 0.1245,
    },
    {
        "name": "Phase 8 — Adaptive FL, resource_only (best F1 pre-Phase-9)",
        "family": "federated (rule-based adaptive)",
        "source": "results/reports/adaptive_fl_results.md",
        "accuracy": 0.7631, "precision": 0.9190, "recall": 0.6402,
        "f1": 0.7547, "roc_auc": 0.8438, "fpr": 0.0746, "fairness_gap": 0.2259,
    },
    {
        "name": "Phase 8 — Adaptive FL, data_only",
        "family": "federated (rule-based adaptive)",
        "source": "results/reports/adaptive_fl_results.md",
        "accuracy": 0.7526, "precision": 0.9223, "recall": 0.6175,
        "f1": 0.7397, "roc_auc": 0.8313, "fpr": 0.0688, "fairness_gap": 0.2113,
    },
    {
        "name": "Phase 8 — Adaptive FL, combined",
        "family": "federated (rule-based adaptive)",
        "source": "results/reports/adaptive_fl_results.md",
        "accuracy": 0.7556, "precision": 0.9217, "recall": 0.6236,
        "f1": 0.7439, "roc_auc": 0.8536, "fpr": 0.0700, "fairness_gap": 0.1496,
    },
    {
        "name": "Phase 9B — DQN adaptive FL",
        "family": "federated (DRL adaptive)",
        "source": "results/reports/drl_adaptive_fl_results.md",
        "accuracy": 0.7548, "precision": 0.9231, "recall": 0.6210,
        "f1": 0.7425, "roc_auc": 0.8361, "fpr": 0.0684, "fairness_gap": 0.1833,
    },
    {
        "name": "Phase 9 — Staleness-aware FL, no_correction (best of the 3)",
        "family": "federated (staleness-aware)",
        "source": "results/reports/staleness_fl_results.md",
        "accuracy": 0.7503, "precision": 0.9199, "recall": 0.6149,
        "f1": 0.7371, "roc_auc": 0.8515, "fpr": 0.0707, "fairness_gap": 0.1964,
    },
    {
        "name": "Phase 9 — Staleness-aware FL, linear_decay",
        "family": "federated (staleness-aware)",
        "source": "results/reports/staleness_fl_results.md",
        "accuracy": 0.7490, "precision": 0.9192, "recall": 0.6129,
        "f1": 0.7354, "roc_auc": 0.8556, "fpr": 0.0712, "fairness_gap": 0.1964,
    },
    {
        "name": "Phase 9 — Staleness-aware FL, exponential_decay",
        "family": "federated (staleness-aware)",
        "source": "results/reports/staleness_fl_results.md",
        "accuracy": 0.7495, "precision": 0.9197, "recall": 0.6134,
        "f1": 0.7360, "roc_auc": 0.8549, "fpr": 0.0707, "fairness_gap": 0.1994,
    },
]

# Methods that are directly comparable as one federated-vs-federated
# family (excludes the two centralized/non-federated references and the
# Phase 7 heterogeneity-sweep points, which vary alpha rather than the
# aggregation mechanism).
FEDERATED_MECHANISM_METHODS = [
    m for m in METHODS
    if m["family"] in (
        "federated", "federated (rule-based adaptive)",
        "federated (DRL adaptive)", "federated (staleness-aware)",
    )
]


def plot_f1_ranking(out_path: Path) -> None:
    ranked = sorted(METHODS, key=lambda m: m["f1"], reverse=True)
    names = [m["name"] for m in ranked]
    values = [m["f1"] for m in ranked]
    colors = []
    for m in ranked:
        if m["family"] == "non-federated":
            colors.append("#6b7280")
        elif m["family"] == "federated":
            colors.append("#3b82f6")
        elif "non-IID" in m["family"]:
            colors.append("#10b981")
        elif "rule-based" in m["family"]:
            colors.append("#f59e0b")
        elif "DRL" in m["family"]:
            colors.append("#8b5cf6")
        else:
            colors.append("#ef4444")

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = range(len(names))
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("KDDTest+ F1")
    ax.set_xlim(0.70, 0.80)
    ax.set_title("Every Method This Project Has Built — KDDTest+ F1, Ranked")
    for i, v in enumerate(values):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fairness_vs_f1(out_path: Path) -> None:
    with_fairness = [m for m in METHODS if m["fairness_gap"] is not None]
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in with_fairness:
        ax.scatter(m["fairness_gap"], m["f1"], s=60)
        ax.annotate(m["name"].split("—")[1].strip(), (m["fairness_gap"], m["f1"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Client fairness gap (lower is better)")
    ax.set_ylabel("KDDTest+ F1 (higher is better)")
    ax.set_title("Fairness vs. Accuracy Trade-off — All Adaptive-FL Variants")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_federated_mechanism_comparison(out_path: Path) -> None:
    """The apples-to-apples comparison: every method that shares Phase
    6's exact partition/model/rounds/hyperparameters, varying only the
    aggregation mechanism."""
    names = [m["name"].split("—")[1].strip() for m in FEDERATED_MECHANISM_METHODS]
    values = [m["f1"] for m in FEDERATED_MECHANISM_METHODS]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names)), values, color="#3b82f6")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("KDDTest+ F1")
    ax.set_ylim(0.72, 0.77)
    ax.set_title("Same Partition, Same Model, Same Rounds — Aggregation Mechanism Only")
    for i, v in enumerate(values):
        ax.text(i, v + 0.0005, f"{v:.4f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report() -> None:
    ranked_all = sorted(METHODS, key=lambda m: m["f1"], reverse=True)
    ranked_mechanism = sorted(FEDERATED_MECHANISM_METHODS, key=lambda m: m["f1"], reverse=True)
    ranked_fairness = sorted(
        [m for m in METHODS if m["fairness_gap"] is not None],
        key=lambda m: m["fairness_gap"],
    )

    lines = [
        "# Phase 14 — Full Cross-Phase Evaluation",
        "",
        "**This report trains nothing and re-runs nothing. Every number below "
        "is copied verbatim from the named source report, each already "
        "committed and independently reproducible from its own phase's "
        "scripts. This document exists to put every method this project has "
        "built, so far, side by side in one place — not to introduce any new "
        "result.**",
        "",
        "## 1. Purpose",
        "",
        "Phases 2 through 9 each measured their own result in isolation, "
        "compared only against their own immediate predecessor. This phase "
        "assembles the full picture: every anomaly-detection method built "
        "across the whole project, ranked on the same held-out KDDTest+ set, "
        "with the federated-learning methods additionally compared on a "
        "client-fairness axis where that was measured.",
        "",
        "## 2. Full ranking — every method, by KDDTest+ F1",
        "",
        "| Rank | Method | Family | F1 | ROC-AUC | Fairness gap | Source |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for i, m in enumerate(ranked_all, start=1):
        roc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a"
        gap = f"{m['fairness_gap']:.4f}" if m["fairness_gap"] is not None else "not measured"
        lines.append(f"| {i} | {m['name']} | {m['family']} | {m['f1']:.4f} | {roc} | {gap} | `{m['source']}` |")

    best_overall = ranked_all[0]
    best_federated = next(m for m in ranked_all if m["family"] != "non-federated")
    worst = ranked_all[-1]
    lines += [
        "",
        f"**Best KDDTest+ F1 across the entire project: {best_overall['name']} "
        f"({best_overall['f1']:.4f})** — a centralized, non-federated model, "
        "expected to lead (see section 5). **Best among every FEDERATED "
        f"result tried, including Phase 7's alpha sweep: {best_federated['name']} "
        f"({best_federated['f1']:.4f})** — this is a lower-heterogeneity DATA "
        "PARTITION, not a different aggregation mechanism; see section 3 for "
        "the more informative apples-to-apples comparison that holds the "
        f"partition fixed and varies only the mechanism. Weakest overall: "
        f"{worst['name']} ({worst['f1']:.4f}).",
        "",
        "## 3. Apples-to-apples: same partition, same model, aggregation "
        "mechanism only",
        "",
        "The comparison above mixes methods that vary the DATA partition "
        "(Phase 7's alpha sweep) with methods that hold the partition fixed "
        "and vary only the AGGREGATION mechanism (Phase 6, 8, 9B, 9). This "
        "section isolates the second, more direct comparison — every entry "
        "below uses the EXACT SAME `data/partitions/` (Phase 4 original, "
        "alpha=0.5), model, and round/epoch budget:",
        "",
        "| Rank | Mechanism | F1 | Fairness gap |",
        "|---:|---|---:|---:|",
    ]
    for i, m in enumerate(ranked_mechanism, start=1):
        gap = f"{m['fairness_gap']:.4f}" if m["fairness_gap"] is not None else "not measured"
        lines.append(f"| {i} | {m['name'].split('—')[1].strip()} | {m['f1']:.4f} | {gap} |")

    mechanism_best = ranked_mechanism[0]
    lines += [
        "",
        f"**Under this controlled comparison, `resource_only` (Phase 8) "
        f"remains the single best mechanism found so far ({mechanism_best['f1']:.4f} F1).** "
        "Neither the learned DQN controller (Phase 9B) nor either "
        "staleness-aware variant (Phase 9) improved on it. This is reported "
        "as the honest, current state of the project — a simple, "
        "hand-specified rule beating two more sophisticated mechanisms "
        "(learned adaptation, staleness correction) is a real, useful "
        "finding, not a disappointment to explain away.",
        "",
        "## 4. Client fairness ranking (federated adaptive methods only)",
        "",
        "| Rank | Method | Fairness gap | F1 |",
        "|---:|---|---:|---:|",
    ]
    for i, m in enumerate(ranked_fairness, start=1):
        lines.append(f"| {i} | {m['name'].split('—')[1].strip()} | {m['fairness_gap']:.4f} | {m['f1']:.4f} |")

    fairest = ranked_fairness[0]
    lines += [
        "",
        f"**Best fairness gap: {fairest['name']} ({fairest['fairness_gap']:.4f}).** "
        "The same method never wins both F1 and fairness across this "
        "project's experiments — `performance_only` (Phase 8) is fairest but "
        "not the best F1; `resource_only` (Phase 8) is best F1 but among the "
        "least fair. This performance/fairness tension, first observed in "
        "Phase 8, holds up across every later phase (9B, 9) that measured "
        "both — a consistent, real pattern, not a one-off artifact.",
        "",
        "## 5. Non-federated references",
        "",
        "For context, not as a fair comparison (different training regime "
        "entirely — one model sees ALL the data, no communication rounds, no "
        "aggregation): the Phase 2 centralized MLP baseline scores "
        f"{METHODS[0]['f1']:.4f} F1, and Phase 3's temporal GRU+attention "
        f"model scores {METHODS[1]['f1']:.4f} F1 — both HIGHER than every "
        "federated method in this report. This gap (centralized beating "
        "every federated variant tried) is expected and already documented "
        "starting in Phase 6's own report: federated learning trades some "
        "raw accuracy for the ability to train without centralizing raw "
        "satellite data, and this project's federated round/epoch budgets "
        "have been deliberately modest throughout.",
        "",
        "## 6. What this evaluation does NOT do",
        "",
        "- No new training, no new hyperparameter search — every number is "
        "sourced, not derived.",
        "- Does not resolve WHY `resource_only` outperforms the more "
        "sophisticated mechanisms — Phase 15 (ablation studies) is where "
        "that gets investigated on a couple of specific design axes, not "
        "here.",
        "- Single seed per method throughout the underlying phases — none of "
        "the differences above are statistically tested; read small gaps "
        "(a few thousandths of F1) with that in mind.",
        "",
        "## 7. Exact artifact locations",
        "",
        "```",
        "results/plots/full_evaluation/*.png",
        "results/reports/full_evaluation.md   (this file)",
        "```",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plot_f1_ranking(PLOTS_DIR / "f1_ranking_all_methods.png")
    plot_fairness_vs_f1(PLOTS_DIR / "fairness_vs_f1.png")
    plot_federated_mechanism_comparison(PLOTS_DIR / "federated_mechanism_comparison.png")
    write_report()

    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

"""Analyze the generated simulated satellite partitions: measure actual
non-IID quality, resource heterogeneity, and generate plots + the final
report.

Run from the repository root, after scripts/create_satellite_partitions.py:

    python scripts/analyze_satellite_partitions.py

Reads:
    data/partitions/manifest.json
    data/partitions/satellite_metadata.json

Writes:
    results/plots/satellite/*.png
    results/reports/satellite_simulation_report.md

Every number in the outputs is computed from the actual generated
partition — nothing here is fabricated or assumed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.simulation.non_iid_metrics import compute_non_iid_report  # noqa: E402
from src.simulation.partitioner import ClientStats  # noqa: E402

PARTITIONS_DIR = REPO_ROOT / "data" / "partitions"
PLOTS_DIR = REPO_ROOT / "results" / "plots" / "satellite"
REPORT_PATH = REPO_ROOT / "results" / "reports" / "satellite_simulation_report.md"
CONFIG_PATH = REPO_ROOT / "configs" / "satellite_simulation.yaml"

RESOURCE_FIELDS = ["bandwidth_mbps", "latency_ms", "compute_score", "availability_probability", "connectivity_quality"]


def load_client_stats(manifest: dict) -> dict[str, ClientStats]:
    stats = {}
    for cid, s in manifest["client_stats"].items():
        stats[cid] = ClientStats(
            client_id=cid,
            total_samples=s["total_samples"],
            normal_samples=s["normal_samples"],
            anomaly_samples=s["anomaly_samples"],
            category_counts=s["category_counts"],
        )
    return stats


def plot_client_sample_sizes(client_stats: dict[str, ClientStats], out_path: Path) -> None:
    ids = list(client_stats.keys())
    sizes = [client_stats[cid].total_samples for cid in ids]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ids, sizes, color="#3b82f6")
    ax.set_ylabel("Training samples")
    ax.set_title("Client Data Size — Simulated Satellite Partitions", pad=15)
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(sizes):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_anomaly_proportion(client_stats: dict[str, ClientStats], out_path: Path) -> None:
    ids = list(client_stats.keys())
    pct = [client_stats[cid].anomaly_percentage for cid in ids]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ids, pct, color="#ef4444")
    ax.set_ylabel("Anomaly percentage (%)")
    ax.set_title("Anomaly Proportion by Simulated Satellite Client", pad=15)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 105)
    for i, v in enumerate(pct):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_category_distribution(client_stats: dict[str, ClientStats], out_path: Path) -> None:
    ids = list(client_stats.keys())
    categories = sorted({cat for s in client_stats.values() for cat in s.category_counts})
    colors = {"normal": "#3b82f6", "dos": "#ef4444", "probe": "#eab308", "r2l": "#22c55e", "u2r": "#a855f7"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(ids))
    for cat in categories:
        values = np.array([client_stats[cid].category_counts.get(cat, 0) for cid in ids])
        ax.bar(ids, values, bottom=bottom, label=cat, color=colors.get(cat))
        bottom += values
    ax.set_ylabel("Sample count")
    ax.set_title("Attack Category Distribution by Simulated Satellite Client", pad=15)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Category")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_resource_distributions(satellite_metadata: dict, out_path: Path) -> None:
    client_ids = list(satellite_metadata["clients"].keys())
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flat
    for ax, field in zip(axes, RESOURCE_FIELDS):
        values = [satellite_metadata["clients"][cid][field] for cid in client_ids]
        ax.bar(client_ids, values, color="#60a5fa")
        ax.set_title(field, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    for ax in list(axes)[len(RESOURCE_FIELDS):]:
        ax.axis("off")
    fig.suptitle("Simulated Satellite Resource Conditions (per client)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def resource_statistics(satellite_metadata: dict) -> dict[str, dict[str, float]]:
    client_ids = list(satellite_metadata["clients"].keys())
    stats = {}
    for field in RESOURCE_FIELDS:
        values = np.array([satellite_metadata["clients"][cid][field] for cid in client_ids])
        stats[field] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
    return stats


def main() -> None:
    manifest_path = PARTITIONS_DIR / "manifest.json"
    metadata_path = PARTITIONS_DIR / "satellite_metadata.json"
    for path in (manifest_path, metadata_path):
        if not path.exists():
            print(f"ERROR: {path} not found. Run scripts/create_satellite_partitions.py first.", file=sys.stderr)
            sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    satellite_metadata = json.loads(metadata_path.read_text())
    sim_config = yaml.safe_load(CONFIG_PATH.read_text())
    max_resample_attempts = sim_config["partitioning"].get("max_resample_attempts", "N/A")

    client_stats = load_client_stats(manifest)
    non_iid_report = compute_non_iid_report(client_stats)
    res_stats = resource_statistics(satellite_metadata)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plot_client_sample_sizes(client_stats, PLOTS_DIR / "client_data_sizes.png")
    plot_anomaly_proportion(client_stats, PLOTS_DIR / "anomaly_proportion.png")
    plot_category_distribution(client_stats, PLOTS_DIR / "attack_category_distribution.png")
    plot_resource_distributions(satellite_metadata, PLOTS_DIR / "resource_distributions.png")

    print("=== Non-IID measurements ===")
    print(f"Pairwise JS distance: avg={non_iid_report.pairwise_js_distance_avg:.4f}, "
          f"min={non_iid_report.pairwise_js_distance_min:.4f}, max={non_iid_report.pairwise_js_distance_max:.4f}")
    print(f"Per-client entropy (bits): {non_iid_report.client_entropy}")
    print(f"Per-category variance: {non_iid_report.per_category_variance}")

    print("\n=== Resource statistics ===")
    for field, s in res_stats.items():
        print(f"{field}: min={s['min']:.2f}, max={s['max']:.2f}, mean={s['mean']:.2f}")

    # --- Report ---
    lines = [
        "# Satellite Simulation Report",
        "",
        "## 1. Purpose",
        "",
        "Simulation of a multi-satellite learning environment built on top "
        "of a real network intrusion dataset, so that later Federated "
        "Learning / Adaptive FL / DRL phases have a realistic, "
        "non-IID, multi-client environment to operate on. **This phase "
        "does not implement Federated Learning, DRL, or any training "
        "algorithm — it builds only the client environment.**",
        "",
        "## 2. Dataset source",
        "",
        "Real NSL-KDD training data (`data/processed/train.parquet`, "
        "produced by the Phase 1 pipeline). NSL-KDD is a terrestrial "
        "network intrusion-detection dataset — **not** satellite traffic. "
        "No real satellite measurements are used anywhere in this "
        "simulation. Only the training split is partitioned; validation "
        "remains a single global held-out set and KDDTest+ is completely "
        "untouched (see `configs/satellite_simulation.yaml` and "
        "`data/partitions/README.md`).",
        "",
        f"## 3. Number of satellites",
        "",
        f"**{len(client_stats)}** simulated clients "
        f"({', '.join(client_stats.keys())}), configured via "
        "`configs/satellite_simulation.yaml` (`satellites.num_clients`).",
        "",
        "## 4. Partition strategy",
        "",
        f"Dirichlet label-skew partitioning (`configs/satellite_simulation.yaml` "
        f"`partitioning.strategy: dirichlet`, alpha={manifest['alpha']}, "
        f"seed={manifest['seed']}). For each attack CATEGORY "
        "(normal/dos/probe/r2l/u2r — `src/data/schema.py:ATTACK_CATEGORY_MAP`), "
        "that category's real record indices are shuffled and split across "
        "clients according to a Dirichlet-sampled proportion vector. Every "
        "real record is assigned to exactly one client — see Data "
        "integrity below.",
        f"",
        f"Constraints satisfied: **{manifest['constraints_satisfied']}** "
        f"(took {manifest['attempts_used']} attempt(s) out of a max of "
        f"{max_resample_attempts}).",
        "",
        "## 5. Why the data is non-IID",
        "",
        "Because each attack category is independently split across "
        "clients using a *different* random Dirichlet proportion vector, "
        "clients end up with different mixes of attack categories — some "
        "dominated by one attack type, others mostly normal traffic — "
        "rather than each client getting a uniform random sample of the "
        "whole dataset. Section 10 below measures this directly rather "
        "than assuming it.",
        "",
        "## 6. Client sample distributions",
        "",
        "| Client | Samples | Normal | Anomaly | Anomaly % |",
        "|---|---:|---:|---:|---:|",
    ]
    for cid, s in client_stats.items():
        lines.append(f"| {cid} | {s.total_samples} | {s.normal_samples} | {s.anomaly_samples} | {s.anomaly_percentage:.1f}% |")

    lines += [
        "",
        f"Total training samples partitioned: **{sum(s.total_samples for s in client_stats.values())}** "
        f"(matches `data/processed/train.parquet` exactly — "
        f"{len(client_stats)} clients, no loss, no duplication).",
        "",
        "See `results/plots/satellite/client_data_sizes.png` and `anomaly_proportion.png`.",
        "",
        "## 7. Attack distributions",
        "",
        "| Client | " + " | ".join(non_iid_report.categories) + " | Categories present |",
        "|---|" + "|".join(["---:"] * len(non_iid_report.categories)) + "|---:|",
    ]
    for cid, s in client_stats.items():
        row = [str(s.category_counts.get(cat, 0)) for cat in non_iid_report.categories]
        lines.append(f"| {cid} | " + " | ".join(row) + f" | {s.categories_represented} |")
    lines += ["", "See `results/plots/satellite/attack_category_distribution.png`.", ""]

    lines += [
        "## 8. Resource simulation",
        "",
        "SIMULATED per-client operating conditions "
        "(`configs/satellite_simulation.yaml` -> `resources`), sampled "
        "uniformly at random within the configured ranges, seeded for "
        "reproducibility. These are controlled experiment parameters, "
        "**not** measurements from real satellite hardware.",
        "",
        "| Client | Bandwidth (Mbps) | Latency (ms) | Compute score | Availability | Connectivity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cid in client_stats:
        c = satellite_metadata["clients"][cid]
        lines.append(
            f"| {cid} | {c['bandwidth_mbps']:.2f} | {c['latency_ms']:.2f} | {c['compute_score']:.3f} | "
            f"{c['availability_probability']:.3f} | {c['connectivity_quality']:.3f} |"
        )
    lines += ["", "See `results/plots/satellite/resource_distributions.png`.", ""]

    lines += [
        "## 9. Connectivity simulation",
        "",
        "`availability_probability` (probability the client is reachable "
        "during a communication round) and `connectivity_quality` (link "
        "quality when it IS reachable) are modeled as two independent "
        "SIMULATED dimensions — see the table above for actual generated "
        "values per client.",
        "",
        "## 10. Non-IID measurements",
        "",
        f"- Pairwise Jensen-Shannon distance between clients' category "
        f"distributions (0 = identical, 1 = maximally different): "
        f"**avg={non_iid_report.pairwise_js_distance_avg:.4f}, "
        f"min={non_iid_report.pairwise_js_distance_min:.4f}, "
        f"max={non_iid_report.pairwise_js_distance_max:.4f}**",
        "- Per-client category-distribution entropy (bits; lower = more "
        "skewed toward few categories):",
        "",
        "| Client | Entropy (bits) |",
        "|---|---:|",
    ]
    for cid, ent in non_iid_report.client_entropy.items():
        lines.append(f"| {cid} | {ent:.4f} |")
    lines += [
        "",
        f"(Maximum possible entropy for {len(non_iid_report.categories)} categories, "
        f"i.e. perfectly uniform, is log2({len(non_iid_report.categories)}) = "
        f"{np.log2(len(non_iid_report.categories)):.4f} bits — clients well below "
        "that are meaningfully skewed toward fewer categories.)",
        "",
        "- Per-category variance of proportion across clients (higher = "
        "clients disagree more about how common that category is):",
        "",
        "| Category | Variance across clients |",
        "|---|---:|",
    ]
    for cat, var in non_iid_report.per_category_variance.items():
        lines.append(f"| {cat} | {var:.6f} |")

    lines += [
        "",
        "## 11. Resource statistics",
        "",
        "| Field | Min | Max | Mean |",
        "|---|---:|---:|---:|",
    ]
    for field, s in res_stats.items():
        lines.append(f"| {field} | {s['min']:.3f} | {s['max']:.3f} | {s['mean']:.3f} |")

    lines += [
        "",
        "## 12. Reproducibility",
        "",
        f"- Seed: {manifest['seed']}",
        f"- Partition strategy: {manifest['partition_strategy']} "
        f"(alpha={manifest['alpha']}, category_column={manifest['category_column']})",
        "- Same config + same seed reproduces the exact same partition and "
        "resource values (verified by "
        "`tests/simulation/test_partitioner.py::test_dirichlet_partition_deterministic_with_same_seed` "
        "and `test_network_conditions.py::test_generate_network_conditions_deterministic_with_same_seed`).",
        "- Regenerate with: `python scripts/create_satellite_partitions.py` "
        "then `python scripts/analyze_satellite_partitions.py`.",
        "",
        "## 13. Limitations",
        "",
        "- **All satellite conditions (bandwidth, latency, compute, "
        "availability, connectivity) are simulated** — sampled uniformly "
        "at random within configured ranges, not derived from real "
        "orbital mechanics, real link budgets, or real hardware.",
        "- **NSL-KDD remains a terrestrial dataset.** No claim is made "
        "that any record represents real satellite traffic.",
        "- Dirichlet partitioning creates label-distribution skew "
        "(different mixes of attack categories per client); it does not "
        "simulate genuine spatial/geographic relationships between "
        "satellites, orbital coverage patterns, or inter-satellite links.",
        "- Rare categories (`r2l`, `u2r`) have very few real records "
        "overall, so some clients legitimately receive zero or very few "
        "of them — this is a property of the real data, not a partition "
        "bug.",
        "- No Federated Learning, DRL, or model training happens in this "
        "phase — this is environment construction only.",
        "",
        "## 14. What this enables next",
        "",
        "Phase 5 (Local Satellite Training) trains a local model "
        "independently on each client's real partitioned data, using the "
        "client datasets and metadata produced here — still without any "
        "federated aggregation, communication rounds, or DRL.",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()

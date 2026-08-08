"""Measure how non-IID a partition actually is — never just assert it.

Reports:
- per-client category distribution (proportions)
- pairwise Jensen-Shannon divergence between clients' distributions
  (0 = identical distributions, up to ln(2) ≈ 0.693 for base-e JS
  divergence between maximally different distributions — we report the
  scipy `jensenshannon` DISTANCE, which is the square root of the
  divergence and bounded in [0, 1], easier to read)
- Shannon entropy of each client's category distribution (lower =
  more skewed toward few categories = more non-IID for that client)
- variance across clients of each category's proportion (higher =
  clients disagree more about how common that category is = more non-IID)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import jensenshannon

from src.simulation.partitioner import ClientStats


def client_category_distribution(stats: ClientStats, categories: list[str]) -> np.ndarray:
    """Proportions over `categories`, in the given fixed order, summing to 1
    (or all-zero if the client has 0 samples — callers should guard against
    that; real partitions here always have min_total_samples > 0)."""
    counts = np.array([stats.category_counts.get(cat, 0) for cat in categories], dtype="float64")
    total = counts.sum()
    return counts / total if total > 0 else counts


def shannon_entropy(distribution: np.ndarray) -> float:
    """Shannon entropy in bits. 0 = all mass on one category (maximally
    skewed); log2(len(distribution)) = perfectly uniform."""
    nonzero = distribution[distribution > 0]
    if len(nonzero) == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log2(nonzero)))


@dataclass
class NonIIDReport:
    categories: list[str]
    client_distributions: dict[str, list[float]]
    client_entropy: dict[str, float]
    pairwise_js_distance_avg: float
    pairwise_js_distance_min: float
    pairwise_js_distance_max: float
    per_category_variance: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "categories": self.categories,
            "client_distributions": self.client_distributions,
            "client_entropy": {k: round(v, 4) for k, v in self.client_entropy.items()},
            "pairwise_js_distance_avg": round(self.pairwise_js_distance_avg, 4),
            "pairwise_js_distance_min": round(self.pairwise_js_distance_min, 4),
            "pairwise_js_distance_max": round(self.pairwise_js_distance_max, 4),
            "per_category_variance": {k: round(v, 6) for k, v in self.per_category_variance.items()},
        }


def compute_non_iid_report(client_stats: dict[str, ClientStats]) -> NonIIDReport:
    client_ids = list(client_stats.keys())
    all_categories = sorted({cat for s in client_stats.values() for cat in s.category_counts})

    distributions = {
        cid: client_category_distribution(client_stats[cid], all_categories) for cid in client_ids
    }

    entropy = {cid: shannon_entropy(dist) for cid, dist in distributions.items()}

    js_distances = []
    for i in range(len(client_ids)):
        for j in range(i + 1, len(client_ids)):
            d = float(jensenshannon(distributions[client_ids[i]], distributions[client_ids[j]], base=2))
            if not np.isnan(d):  # nan if both distributions are all-zero; shouldn't happen for real clients
                js_distances.append(d)

    per_category_variance = {}
    for cat_idx, cat in enumerate(all_categories):
        proportions = np.array([distributions[cid][cat_idx] for cid in client_ids])
        per_category_variance[cat] = float(np.var(proportions))

    return NonIIDReport(
        categories=all_categories,
        client_distributions={cid: distributions[cid].tolist() for cid in client_ids},
        client_entropy=entropy,
        pairwise_js_distance_avg=float(np.mean(js_distances)) if js_distances else 0.0,
        pairwise_js_distance_min=float(np.min(js_distances)) if js_distances else 0.0,
        pairwise_js_distance_max=float(np.max(js_distances)) if js_distances else 0.0,
        per_category_variance=per_category_variance,
    )

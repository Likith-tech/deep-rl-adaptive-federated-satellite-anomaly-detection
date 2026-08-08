"""Dirichlet-based non-IID partitioning of the (real) training data
across simulated satellite clients.

Strategy (standard label-distribution-skew Dirichlet partitioning, as
used widely in the non-IID federated learning literature):

    for each attack CATEGORY present in the training data
        (normal / dos / probe / r2l / u2r — see src/data/schema.py):
            shuffle that category's row indices (seeded)
            draw a Dirichlet(alpha) proportion vector over the clients
            split the shuffled indices across clients by that proportion

Because every category's indices are partitioned (not resampled or
duplicated) across ALL clients, the union of all client index sets is
always exactly the full set of input indices, with no overlap — data
integrity (no loss, no duplication) holds BY CONSTRUCTION, not just by
a post-hoc check (the post-hoc check in tests/simulation exists anyway,
as a guard against regressions).

Lower `alpha` => more skewed (non-IID) client distributions. Higher
`alpha` => closer to IID. This only uses REAL, already-existing labels
(label_original / label_binary from Phase 1) — no labels or samples are
invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ClientStats:
    client_id: str
    total_samples: int
    normal_samples: int
    anomaly_samples: int
    category_counts: dict[str, int]

    @property
    def anomaly_percentage(self) -> float:
        return 100 * self.anomaly_samples / self.total_samples if self.total_samples else 0.0

    @property
    def categories_represented(self) -> int:
        return sum(1 for count in self.category_counts.values() if count > 0)

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "total_samples": self.total_samples,
            "normal_samples": self.normal_samples,
            "anomaly_samples": self.anomaly_samples,
            "anomaly_percentage": round(self.anomaly_percentage, 4),
            "category_counts": self.category_counts,
            "categories_represented": self.categories_represented,
        }


@dataclass
class PartitionResult:
    client_indices: dict[str, np.ndarray]
    category_column: str
    alpha: float
    seed: int
    attempts_used: int
    constraints_satisfied: bool
    constraint_violations: dict[str, list[str]] = field(default_factory=dict)

    def to_summary_dict(self) -> dict:
        return {
            "category_column": self.category_column,
            "alpha": self.alpha,
            "seed": self.seed,
            "attempts_used": self.attempts_used,
            "constraints_satisfied": self.constraints_satisfied,
            "constraint_violations": self.constraint_violations,
        }


def _allocate_counts(n: int, proportions: np.ndarray) -> list[int]:
    """Split `n` items across len(proportions) buckets according to
    `proportions` (which sum to ~1), using the largest-remainder method
    so the counts always sum EXACTLY to n."""
    raw = proportions * n
    counts = np.floor(raw).astype(int)
    remainder = n - counts.sum()
    if remainder > 0:
        fractional = raw - counts
        top_up = np.argsort(-fractional)[:remainder]
        counts[top_up] += 1
    return counts.tolist()


def compute_category_series(df: pd.DataFrame, category_column: str, category_map: dict[str, str] | None) -> pd.Series:
    series = df[category_column]
    return series.map(category_map) if category_map else series


def compute_client_stats(
    df: pd.DataFrame,
    client_indices: dict[str, np.ndarray],
    category_series: pd.Series,
    label_binary_column: str = "label_binary",
) -> dict[str, ClientStats]:
    """Real, measured per-client statistics — never fabricated."""
    stats: dict[str, ClientStats] = {}
    for client_id, indices in client_indices.items():
        subset_labels = df.loc[indices, label_binary_column]
        subset_categories = category_series.loc[indices]
        category_counts = {cat: int((subset_categories == cat).sum()) for cat in sorted(category_series.unique())}
        stats[client_id] = ClientStats(
            client_id=client_id,
            total_samples=len(indices),
            normal_samples=int((subset_labels == 0).sum()),
            anomaly_samples=int((subset_labels == 1).sum()),
            category_counts=category_counts,
        )
    return stats


def _check_constraints(
    stats: dict[str, ClientStats], constraints: dict
) -> dict[str, list[str]]:
    violations: dict[str, list[str]] = {}
    for client_id, s in stats.items():
        failed = []
        if s.total_samples < constraints.get("min_total_samples", 0):
            failed.append(f"total_samples {s.total_samples} < {constraints['min_total_samples']}")
        if s.normal_samples < constraints.get("min_normal_samples", 0):
            failed.append(f"normal_samples {s.normal_samples} < {constraints['min_normal_samples']}")
        if s.anomaly_samples < constraints.get("min_anomaly_samples", 0):
            failed.append(f"anomaly_samples {s.anomaly_samples} < {constraints['min_anomaly_samples']}")
        if s.categories_represented < constraints.get("min_categories_represented", 0):
            failed.append(
                f"categories_represented {s.categories_represented} < {constraints['min_categories_represented']}"
            )
        if failed:
            violations[client_id] = failed
    return violations


def dirichlet_partition(
    df: pd.DataFrame,
    client_ids: list[str],
    category_column: str,
    alpha: float,
    seed: int,
    category_map: dict[str, str] | None = None,
    label_binary_column: str = "label_binary",
    constraints: dict | None = None,
    max_attempts: int = 20,
) -> PartitionResult:
    """Partition `df`'s rows across `client_ids` via Dirichlet label-skew
    partitioning on `category_column` (optionally mapped through
    `category_map`, e.g. ATTACK_CATEGORY_MAP to group by attack
    category rather than fine-grained attack name).

    Retries (same seeded RNG, so still fully deterministic for a given
    `seed`) up to `max_attempts` times if `constraints` aren't met by
    every client, then returns the last attempt with
    constraints_satisfied=False and the specific violations recorded —
    never silently hides a constraint failure.
    """
    constraints = constraints or {}
    category_series = compute_category_series(df, category_column, category_map)
    categories = sorted(category_series.unique())
    category_index_groups = {cat: df.index[category_series == cat].to_numpy() for cat in categories}

    rng = np.random.default_rng(seed)
    num_clients = len(client_ids)

    last_client_indices: dict[str, np.ndarray] = {}
    last_violations: dict[str, list[str]] = {}

    for attempt in range(1, max_attempts + 1):
        client_indices: dict[str, list[int]] = {cid: [] for cid in client_ids}

        for cat, indices in category_index_groups.items():
            shuffled = rng.permutation(indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            counts = _allocate_counts(len(shuffled), proportions)
            start = 0
            for cid, count in zip(client_ids, counts):
                client_indices[cid].extend(shuffled[start : start + count].tolist())
                start += count

        client_indices_arr = {cid: np.array(idx, dtype="int64") for cid, idx in client_indices.items()}
        stats = compute_client_stats(df, client_indices_arr, category_series, label_binary_column)
        violations = _check_constraints(stats, constraints) if constraints else {}

        last_client_indices = client_indices_arr
        last_violations = violations

        if not violations:
            return PartitionResult(
                client_indices=client_indices_arr,
                category_column=category_column,
                alpha=alpha,
                seed=seed,
                attempts_used=attempt,
                constraints_satisfied=True,
                constraint_violations={},
            )

    return PartitionResult(
        client_indices=last_client_indices,
        category_column=category_column,
        alpha=alpha,
        seed=seed,
        attempts_used=max_attempts,
        constraints_satisfied=False,
        constraint_violations=last_violations,
    )


def verify_partition_integrity(df: pd.DataFrame, client_indices: dict[str, np.ndarray]) -> None:
    """Raise AssertionError if the partition lost, duplicated, or
    invented any sample. Used both in tests and as a runtime guard when
    generating the real partition."""
    all_indices = np.concatenate(list(client_indices.values())) if client_indices else np.array([], dtype="int64")
    assert len(all_indices) == len(set(all_indices.tolist())), "Partition assigned a sample to more than one client"
    assert set(all_indices.tolist()) == set(df.index.tolist()), (
        "Partition does not exactly cover the source dataset (samples lost or invented)"
    )

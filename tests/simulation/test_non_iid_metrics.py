from __future__ import annotations

import numpy as np
import pytest

from src.simulation.non_iid_metrics import (
    client_category_distribution,
    compute_non_iid_report,
    shannon_entropy,
)
from src.simulation.partitioner import ClientStats


def test_shannon_entropy_uniform_distribution_is_maximal():
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    assert shannon_entropy(uniform) == pytest.approx(2.0)  # log2(4) = 2


def test_shannon_entropy_single_category_is_zero():
    skewed = np.array([1.0, 0.0, 0.0, 0.0])
    assert shannon_entropy(skewed) == 0.0


def test_shannon_entropy_empty_distribution_is_zero():
    assert shannon_entropy(np.array([0.0, 0.0])) == 0.0


def test_client_category_distribution_sums_to_one():
    stats = ClientStats(
        client_id="SAT-01", total_samples=100, normal_samples=60, anomaly_samples=40,
        category_counts={"normal": 60, "dos": 30, "probe": 10},
    )
    dist = client_category_distribution(stats, ["normal", "dos", "probe"])
    assert np.isclose(dist.sum(), 1.0)
    assert np.allclose(dist, [0.6, 0.3, 0.1])


def test_compute_non_iid_report_identical_clients_have_zero_js_distance():
    identical_counts = {"normal": 50, "dos": 50}
    client_stats = {
        "SAT-01": ClientStats("SAT-01", 100, 50, 50, dict(identical_counts)),
        "SAT-02": ClientStats("SAT-02", 100, 50, 50, dict(identical_counts)),
    }
    report = compute_non_iid_report(client_stats)
    assert report.pairwise_js_distance_avg == pytest.approx(0.0, abs=1e-9)
    assert report.per_category_variance["normal"] == pytest.approx(0.0, abs=1e-9)


def test_compute_non_iid_report_disjoint_clients_have_high_js_distance():
    client_stats = {
        "SAT-01": ClientStats("SAT-01", 100, 100, 0, {"normal": 100, "dos": 0}),
        "SAT-02": ClientStats("SAT-02", 100, 0, 100, {"normal": 0, "dos": 100}),
    }
    report = compute_non_iid_report(client_stats)
    # Maximally different (disjoint) distributions -> JS distance close to 1
    assert report.pairwise_js_distance_avg > 0.9
    assert report.per_category_variance["normal"] > 0.0


def test_compute_non_iid_report_includes_all_categories_and_clients():
    client_stats = {
        "SAT-01": ClientStats("SAT-01", 100, 60, 40, {"normal": 60, "dos": 40}),
        "SAT-02": ClientStats("SAT-02", 100, 20, 80, {"normal": 20, "dos": 50, "probe": 30}),
    }
    report = compute_non_iid_report(client_stats)
    assert set(report.categories) == {"normal", "dos", "probe"}
    assert set(report.client_distributions.keys()) == {"SAT-01", "SAT-02"}
    assert set(report.client_entropy.keys()) == {"SAT-01", "SAT-02"}

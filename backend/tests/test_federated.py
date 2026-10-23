"""Tests for the federated overview/rounds/adaptive-control API."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_overview_identifies_resource_only_as_best_mechanism():
    """Ground truth from Phase 14's full evaluation: resource_only
    (F1 0.7547) is the best aggregation mechanism found so far —
    higher than the DQN and every staleness-aware variant."""
    response = client.get("/federated/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["best_mechanism"]["id"] == "phase8_resource_only"
    assert body["best_mechanism"]["test_f1"] == 0.7547


def test_overview_identifies_performance_only_as_fairest_mechanism():
    response = client.get("/federated/overview")
    body = response.json()
    assert body["fairest_mechanism"]["id"] == "phase8_performance_only"
    assert body["fairest_mechanism"]["fairness_gap"] == 0.1245


def test_overview_mechanisms_excludes_non_federated_and_non_iid_sweep():
    response = client.get("/federated/overview")
    families = {m["family"] for m in response.json()["mechanisms"]}
    assert "non-federated" not in families
    assert "federated (non-IID study)" not in families


def test_rounds_endpoint_returns_real_round_history():
    response = client.get("/federated/rounds/phase6_fedavg")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 10
    assert body[-1]["round_number"] == 10


def test_rounds_endpoint_404_for_unknown_experiment():
    response = client.get("/federated/rounds/does_not_exist")
    assert response.status_code == 404


def test_adaptive_control_returns_all_four_phase8_rules():
    response = client.get("/federated/adaptive-control")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 4
    resource_only = next(r for r in body if r["experiment_id"] == "phase8_resource_only")
    assert resource_only["weights"]["resource"] == 1.0

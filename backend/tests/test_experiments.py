"""Tests for the experiments/models registry API — asserts against the
real, already-committed data these endpoints read (not synthetic
fixtures), same as the reference numbers in
scripts/generate_full_evaluation_report.py."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_list_experiments_returns_all_fourteen_methods():
    response = client.get("/experiments")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 14
    ids = {e["id"] for e in body}
    assert "phase6_fedavg" in ids
    assert "phase9b_dqn" in ids


def test_experiment_detail_includes_real_round_history():
    response = client.get("/experiments/phase8_resource_only")
    assert response.status_code == 200
    body = response.json()
    assert body["test_f1"] == 0.7547
    assert body["has_round_history"] is True
    assert len(body["rounds"]) == 10
    assert body["rounds"][0]["round_number"] == 1


def test_experiment_without_round_history_returns_empty_rounds():
    response = client.get("/experiments/phase2_centralized_baseline")
    assert response.status_code == 200
    body = response.json()
    assert body["has_round_history"] is False
    assert body["rounds"] == []


def test_unknown_experiment_id_returns_404():
    response = client.get("/experiments/does_not_exist")
    assert response.status_code == 404


def test_models_endpoint_mirrors_experiments():
    experiments = client.get("/experiments").json()
    models = client.get("/models").json()
    assert {e["id"] for e in experiments} == {m["id"] for m in models}


def test_model_detail_matches_experiment_detail():
    response = client.get("/models/phase9b_dqn")
    assert response.status_code == 200
    assert response.json()["test_f1"] == 0.7425


def test_unknown_model_id_returns_404():
    response = client.get("/models/does_not_exist")
    assert response.status_code == 404

"""Tests for the Phase 9B DQN controller API."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_controller_summary_reports_real_pretraining_scale():
    response = client.get("/drl/controller")
    assert response.status_code == 200
    body = response.json()
    assert body["episodes_trained"] == 50
    assert body["rounds_per_episode"] == 15
    assert body["total_env_steps"] == 750
    assert body["state_dimensions"] == 44
    assert len(body["action_catalogue"]) == 9


def test_controller_summary_honestly_reports_dqn_did_not_beat_best_rule():
    response = client.get("/drl/controller")
    body = response.json()
    assert body["test_f1"] == 0.7425
    assert body["beat_best_rule_based_baseline"] is False


def test_decisions_returns_real_final_run_rounds():
    response = client.get("/drl/decisions")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 10
    assert body[0]["round_number"] == 1
    assert body[0]["action_name"]
    assert abs(sum(body[0]["weights"].values()) - 1.0) < 1e-4

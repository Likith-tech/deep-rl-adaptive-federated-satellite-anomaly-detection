"""Tests for the health check endpoint."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "orbitshield-backend"


def test_root_endpoint_returns_running_status():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

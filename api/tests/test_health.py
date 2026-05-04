"""Smoke test for the health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import API_VERSION, app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == API_VERSION
    assert "timestamp" in body


def test_openapi_docs_available():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "EdgeAcademy API"

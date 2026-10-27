"""API tests with FastAPI's TestClient. The agent is never reached."""

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(monkeypatch):
    """Client whose agent blows up, so any accidental LLM call fails the test."""

    def fail(*args, **kwargs):
        raise AssertionError("the agent must not be called in these tests")

    monkeypatch.setattr(main, "run_triage", fail)
    # Built without the context manager on purpose: that skips the lifespan,
    # so no Qdrant ingestion happens while testing.
    return TestClient(main.app)


def test_blank_description_and_no_image_is_rejected(client):
    response = client.post("/api/v1/claims/triage", data={"description": "   "})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_input"


def test_unknown_claim_report_is_404(client):
    assert client.get("/api/v1/claims/does-not-exist").status_code == 404


def test_unknown_claim_trace_is_404(client):
    assert client.get("/api/v1/agent/trace/does-not-exist").status_code == 404

"""API integration tests: health, ingest, retrieve (with inline extraction)."""

import pytest
from fastapi.testclient import TestClient

from mnemo.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ingest_returns_200(client: TestClient):
    r = client.post(
        "/memory/ingest",
        json={
            "user_id": "test_user",
            "messages": [
                {"role": "user", "content": "I use Python and FastAPI for backend."},
                {"role": "assistant", "content": "Noted."},
            ],
            "session_id": "s1",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ingested"
    assert data["extraction"] == "queued"
    assert "episode_id" in data and data["episode_id"]


def test_retrieve_empty(client: TestClient):
    r = client.get("/memory/retrieve", params={"user_id": "nonexistent", "query": "python"})
    assert r.status_code == 200
    data = r.json()
    assert "memories" in data
    assert "profile" in data
    assert data["memories"] == []
    assert data["profile"] == {}


def test_profile_empty(client: TestClient):
    r = client.get("/memory/profile", params={"user_id": "nobody"})
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == "nobody"
    assert data["facts"] == {}

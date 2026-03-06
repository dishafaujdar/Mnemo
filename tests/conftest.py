"""Pytest fixtures: app, client, db session, in-memory SQLite."""

import asyncio
import os
import tempfile

import pytest

# Use session-scoped temp paths so app and fixture share the same DB and Qdrant storage
_session_tmp = tempfile.mkdtemp(prefix="mnemo_test_")
_session_db = os.path.join(_session_tmp, "test.db")
_session_qdrant = os.path.join(_session_tmp, "qdrant")
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_session_db}"
os.environ["QDRANT_PATH"] = _session_qdrant

from mnemo.app.main import app  # creates engine with above env
from mnemo.app.db.qdrant import ensure_collection, get_qdrant_client
from mnemo.app.db.sqlite import init_db


def _setup_db_and_qdrant():
    async def _():
        await init_db()
        q = get_qdrant_client()
        await ensure_collection(q)
    asyncio.run(_())


# Create tables and Qdrant collection at conftest load so app and tests share them
_setup_db_and_qdrant()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def anyio_backend():
    return "asyncio"

"""Bi-temporal close semantics tests."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from mnemo.app.db.models import SemanticEdge
from mnemo.app.services.conflict.resolver import close_edges
from mnemo.app.services.conflict.temporal import retracted_at, utc_now


def _edge(valid_at: datetime) -> SemanticEdge:
    return SemanticEdge(
        id="edge-1",
        user_id="u_test",
        subject="user",
        relation="IS",
        object="backend engineer",
        fact_string="user is a backend engineer",
        confidence=0.9,
        valid_at=valid_at,
        invalid_at=None,
        episode_id="ep1",
        created_at=valid_at,
    )


@pytest.mark.asyncio
async def test_close_edges_sets_valid_until_and_retracted_at():
    old_valid_at = utc_now() - timedelta(hours=1)
    new_valid_at = utc_now()
    retracted = utc_now()
    edge = _edge(old_valid_at)
    db = AsyncMock()

    await close_edges(db, [(edge, new_valid_at)], retracted, qdrant_client=None)

    assert edge.invalid_at == new_valid_at
    assert retracted_at(edge) == retracted

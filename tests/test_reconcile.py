"""Post-ingest reconciliation tests."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from mnemo.app.db.models import SemanticEdge
from mnemo.app.services.conflict.resolver import reconcile_active_edges


def _edge(relation: str, obj: str, *, edge_id: str, minutes: int = 0) -> SemanticEdge:
    now = datetime.utcnow() + timedelta(minutes=minutes)
    return SemanticEdge(
        id=edge_id,
        user_id="u_test",
        subject="user",
        relation=relation,
        object=obj,
        fact_string=f"user {relation.lower()} {obj}",
        confidence=0.9,
        valid_at=now,
        invalid_at=None,
        episode_id="ep1",
        created_at=now,
    )


@pytest.mark.asyncio
async def test_reconcile_retires_conflicting_roles_and_employment():
    active = [
        _edge("IS", "backend engineer", edge_id="role-old", minutes=0),
        _edge("HAS_ROLE", "ai engineer", edge_id="role-new", minutes=5),
        _edge("WORKED_AT", "Slice", edge_id="past-job", minutes=6),
        _edge("WORKS_AT", "Deepmind", edge_id="current-job", minutes=7),
        _edge("SWITCHED_FROM", "Slice", edge_id="switch", minutes=8),
        _edge("BORN_IN", "Delhi", edge_id="born", minutes=1),
        _edge("LIVES_IN", "Delhi", edge_id="live", minutes=2),
    ]
    db = AsyncMock()

    with patch(
        "mnemo.app.services.conflict.resolver.get_all_active_edges",
        new=AsyncMock(return_value=active),
    ), patch(
        "mnemo.app.services.conflict.resolver.close_edges",
        new=AsyncMock(),
    ) as mock_close, patch(
        "mnemo.app.services.conflict.resolver.get_qdrant_client",
        return_value=None,
    ):
        count = await reconcile_active_edges(db, "u_test")

    assert count == 4
    retired_ids = {edge.id for edge, _valid_until in mock_close.call_args[0][1]}
    assert retired_ids == {"role-old", "past-job", "switch", "born"}
    valid_until_by_id = {
        edge.id: valid_until for edge, valid_until in mock_close.call_args[0][1]
    }
    assert valid_until_by_id["role-old"] == active[1].valid_at
    assert valid_until_by_id["past-job"] == active[3].valid_at

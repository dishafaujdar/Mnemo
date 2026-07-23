"""Tests for retrieve filtering and profile company sync."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemo.app.db.models import SemanticEdge
from mnemo.app.services.conflict.temporal import active_edge_sql_conditions, is_active_for_agent


def test_is_active_for_agent():
    active = SemanticEdge(
        id="1",
        user_id="u",
        subject="user",
        relation="WORKS_AT",
        object="Deepmind",
        fact_string="x",
        valid_at=datetime.utcnow(),
        invalid_at=None,
        episode_id="e",
        created_at=datetime.utcnow(),
    )
    assert is_active_for_agent(active) is True

    retracted = SemanticEdge(
        id="2",
        user_id="u",
        subject="user",
        relation="WORKS_AT",
        object="Slice",
        fact_string="x",
        valid_at=datetime.utcnow(),
        invalid_at=datetime.utcnow(),
        episode_id="e",
        created_at=datetime.utcnow(),
        metadata_={"retracted_at": datetime.utcnow().isoformat()},
    )
    assert is_active_for_agent(retracted) is False


@pytest.mark.asyncio
async def test_sync_profile_sets_company_from_works_at():
    from mnemo.app.services.conflict.resolver import sync_profile_from_active_edges

    edge = SemanticEdge(
        id="w1",
        user_id="u_test",
        subject="user",
        relation="WORKS_AT",
        object="Deepmind",
        fact_string="user works at Deepmind",
        confidence=0.9,
        valid_at=datetime.utcnow(),
        invalid_at=None,
        episode_id="ep1",
        created_at=datetime.utcnow(),
    )
    db = AsyncMock()

    async def _active(db, user_id, subject, relation):
        if relation == "WORKS_AT":
            return [edge]
        return []

    with patch(
        "mnemo.app.services.conflict.resolver.get_active_edges",
        side_effect=_active,
    ), patch(
        "mnemo.app.services.conflict.resolver.set_fact",
        new=AsyncMock(),
    ) as mock_set, patch(
        "mnemo.app.services.conflict.resolver.delete_fact",
        new=AsyncMock(),
    ):
        await sync_profile_from_active_edges(db, "u_test")

    company_calls = [c for c in mock_set.call_args_list if c[0][2] == "company"]
    assert company_calls
    assert company_calls[-1][0][3] == "Deepmind"

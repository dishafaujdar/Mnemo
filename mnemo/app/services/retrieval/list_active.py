"""List active semantic edges when no search query is provided."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.db.models import SemanticEdge
from mnemo.app.models.extraction import REVIEW_CONFIRMED
from mnemo.app.services.conflict.groups import TRANSITION_RELATIONS
from mnemo.app.services.conflict.temporal import retracted_at
from mnemo.app.services.retrieval.bm25_search import BM25Result

# Same tuple shape as BM25Result
ListResult = BM25Result


async def list_active_memories(
    db: AsyncSession,
    user_id: str,
    valid_only: bool = True,
    confirmed_only: bool = True,
    limit: int = 50,
) -> list[ListResult]:
    """Return semantic edges for a user, newest first (used when retrieve query is empty)."""
    conditions = [SemanticEdge.user_id == user_id]
    if valid_only:
        conditions.append(SemanticEdge.invalid_at.is_(None))
    if confirmed_only:
        conditions.append(SemanticEdge.review_status == REVIEW_CONFIRMED)
    # Hide transition/event facts (legacy rows or extraction slip-through).
    conditions.append(SemanticEdge.relation.notin_(list(TRANSITION_RELATIONS)))

    q = (
        select(SemanticEdge)
        .where(and_(*conditions))
        .order_by(SemanticEdge.valid_at.desc())
        .limit(limit)
    )
    result = await db.execute(q)
    edges = list(result.scalars().all())
    out: list[ListResult] = []
    for e in edges:
        out.append(
            (
                e.id,
                e.fact_string,
                e.confidence,
                e.valid_at,
                e.invalid_at,
                retracted_at(e),
                e.episode_id,
                1.0,
            )
        )
    return out

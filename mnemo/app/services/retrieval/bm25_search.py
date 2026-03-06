"""BM25 full-text search over fact_string for a user's active edges."""

from __future__ import annotations

from datetime import datetime

from rank_bm25 import BM25Okapi
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.db.models import SemanticEdge

# (edge_id, fact_string, confidence, valid_at, invalid_at, episode_id, score)
BM25Result = tuple[str, str, float, datetime, datetime | None, str, float]


async def bm25_search(
    db: AsyncSession,
    query: str,
    user_id: str,
    valid_only: bool = True,
    limit: int = 20,
) -> list[BM25Result]:
    """Run BM25 over fact_string for user's edges; return list of (id, fact, confidence, valid_at, invalid_at, episode_id, score)."""
    conditions = [SemanticEdge.user_id == user_id]
    if valid_only:
        conditions.append(SemanticEdge.invalid_at.is_(None))
    q = select(SemanticEdge).where(and_(*conditions))
    result = await db.execute(q)
    edges = list(result.scalars().all())
    if not edges:
        return []
    corpus = [e.fact_string for e in edges]
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)
    query_tokens = query.lower().split()
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    # Top-k by score
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    out: list[BM25Result] = []
    for idx, score in indexed_scores[:limit]:
        if score <= 0:
            continue
        e = edges[idx]
        out.append(
            (
                e.id,
                e.fact_string,
                e.confidence,
                e.valid_at,
                e.invalid_at,
                e.episode_id,
                float(score),
            )
        )
    return out

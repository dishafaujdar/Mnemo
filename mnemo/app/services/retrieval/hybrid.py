"""RRF fusion of BM25 + vector search and token budget."""

from __future__ import annotations

import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.core.config import settings
from mnemo.app.db.qdrant import get_qdrant_client
from mnemo.app.services.retrieval.bm25_search import bm25_search
from mnemo.app.services.retrieval.budget import RetrievalItem, fit
from mnemo.app.services.retrieval.vector_search import vector_search

K_RRF = 60


def reciprocal_rank_fusion(
    bm25_items: list[RetrievalItem],
    vector_items: list[RetrievalItem],
    k: int = K_RRF,
) -> list[RetrievalItem]:
    """Fuse two ranked lists by RRF; dedupe by edge_id, sort by combined score."""
    scores: dict[str, float] = {}
    id_to_item: dict[str, RetrievalItem] = {}
    for rank, item in enumerate(bm25_items):
        eid = item[0]
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
        id_to_item[eid] = item
    for rank, item in enumerate(vector_items):
        eid = item[0]
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
        id_to_item[eid] = item
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [id_to_item[eid] for eid in sorted_ids]


async def retrieve(
    db: AsyncSession,
    query: str,
    user_id: str,
    token_budget: int | None = None,
    valid_only: bool = True,
    limit: int = 50,
) -> list[RetrievalItem]:
    """
    Run BM25 and vector search in parallel, fuse with RRF, keep only valid facts, apply token budget.
    Returns list of (edge_id, fact_string, confidence, valid_at, invalid_at, episode_id, score).
    """
    qdrant = get_qdrant_client()
    budget = token_budget or settings.default_token_budget
    bm25_task = bm25_search(db, query, user_id, valid_only=valid_only, limit=limit)
    vector_task = vector_search(query, user_id, valid_only=valid_only, limit=limit, qdrant_client=qdrant)
    bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)
    fused = reciprocal_rank_fusion(bm25_results, vector_results, k=K_RRF)
    return fit(fused, budget=budget)

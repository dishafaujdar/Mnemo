"""RRF fusion of BM25 + vector search and token budget."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.core.config import settings
from mnemo.app.db.qdrant import get_qdrant_client
from mnemo.app.services.retrieval.bm25_search import bm25_search
from mnemo.app.services.retrieval.budget import RetrievalItem, fit
from mnemo.app.services.retrieval.vector_search import vector_search

K_RRF = 60
logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    bm25_items: list[RetrievalItem],
    vector_items: list[RetrievalItem],
    k: int = K_RRF,
) -> list[RetrievalItem]:
    """Fuse two ranked lists by RRF; dedupe by edge_id, sort by fused rank."""
    scores: dict[str, float] = {}
    id_to_item: dict[str, RetrievalItem] = {}
    bm25_ranks: dict[str, int] = {}
    vector_ranks: dict[str, int] = {}
    for rank, item in enumerate(bm25_items):
        eid = item[0]
        bm25_ranks[eid] = rank
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
        id_to_item[eid] = item
    for rank, item in enumerate(vector_items):
        eid = item[0]
        vector_ranks[eid] = rank
        scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
        id_to_item[eid] = item
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    for eid in sorted_ids:
        bm25_rank = bm25_ranks.get(eid)
        vector_rank = vector_ranks.get(eid)
        bm25_contrib = 0.0 if bm25_rank is None else 1.0 / (k + bm25_rank + 1)
        vector_contrib = 0.0 if vector_rank is None else 1.0 / (k + vector_rank + 1)
        logger.debug(
            "RRF score edge_id=%s bm25_rank=%s bm25_contrib=%.6f vector_rank=%s vector_contrib=%.6f fused_score=%.6f",
            eid,
            bm25_rank,
            bm25_contrib,
            vector_rank,
            vector_contrib,
            scores[eid],
        )
    # Preserve the original retrieval score in the tuple; use RRF only for ranking.
    return [item for eid in sorted_ids if (item := id_to_item.get(eid)) is not None]


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

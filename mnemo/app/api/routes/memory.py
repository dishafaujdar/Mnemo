"""Memory routes: ingest, retrieve, delete."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.api.dependencies import get_session, require_api_key
from mnemo.app.core.config import settings
from mnemo.app.models.memory import (
    DeleteMemoryResponse,
    IngestRequest,
    IngestResponse,
    RetrieveResponse,
    RetrievedMemory,
)
from mnemo.app.services.memory.episodic import store_turn
from mnemo.app.services.memory.profile import get_profile
from mnemo.app.db.qdrant import get_qdrant_client
from mnemo.app.services.conflict.resolver import invalidate_memory_by_id, rebuild_missing_qdrant_points
from mnemo.app.services.retrieval.budget import count_tokens
from mnemo.app.services.retrieval.hybrid import retrieve as hybrid_retrieve
from mnemo.app.workers.queue import enqueue_extraction

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    body: IngestRequest,
    _api_key: str = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
):
    """Store turn(s), queue async extraction."""
    episode_id = await store_turn(
        session,
        user_id=body.user_id,
        messages=body.messages,
        session_id=body.session_id,
        metadata=body.metadata,
    )
    await session.commit() # adding to db
    await enqueue_extraction(episode_id, body.user_id) # calling spacy or llm api (low confi.) for extraction
    return IngestResponse(episode_id=episode_id, status="ingested", extraction="queued") # ingestig again to db


@router.get("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    user_id: str,
    query: str = "",
    token_budget: int | None = None,
    _api_key: str = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
):
    """Hybrid search → ranked context; include profile."""
    budget = token_budget or settings.default_token_budget
    memories = await hybrid_retrieve(session, query, user_id, token_budget=budget)
    print(f"[DEBUG] retrieve relevance_scores={[m[6] for m in memories]}")
    profile = await get_profile(session, user_id)
    token_count = sum(count_tokens(m[1]) for m in memories)
    return RetrieveResponse(
        memories=[
            RetrievedMemory(
                fact=m[1],
                confidence=m[2],
                valid_at=m[3],
                invalid_at=m[4],
                source_episode_id=m[5],
                relevance_score=m[6],
            )
            for m in memories
        ],
        profile=profile,
        token_count=token_count,
    )


@router.delete("/{memory_id}", response_model=DeleteMemoryResponse)
async def delete_memory(
    memory_id: str,
    _api_key: str = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
):
    """Soft-delete: set invalid_at = now on the semantic edge."""
    from datetime import datetime
    qdrant = get_qdrant_client()
    ok = await invalidate_memory_by_id(session, memory_id, qdrant)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return DeleteMemoryResponse(
        id=memory_id,
        status="invalidated",
        invalid_at=datetime.utcnow(),
    )


@router.post("/admin/repair-qdrant")
async def repair_qdrant(
    user_id: str | None = None,
    _api_key: str = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
):
    """Repair missing active Qdrant points from SQLite."""
    qdrant = get_qdrant_client()
    rebuilt = await rebuild_missing_qdrant_points(session, qdrant, user_id=user_id)
    return {
        "status": "completed",
        "user_id": user_id,
        "rebuilt_points": rebuilt,
    }

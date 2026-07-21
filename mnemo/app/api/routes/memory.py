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
from mnemo.app.services.extraction.service import process_episode_extraction
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
    """Store turn(s), extract facts (sync by default), store confirmed facts."""
    episode_id = await store_turn(
        session,
        user_id=body.user_id,
        messages=body.messages,
        session_id=body.session_id,
        metadata=body.metadata,
    )

    facts_stored = 0
    extraction_status = "queued"

    if settings.extraction_mode == "sync":
        facts_stored = await process_episode_extraction(session, episode_id, body.user_id)
        extraction_status = "completed"
        await session.commit()
    else:
        await session.commit()
        enqueued = await enqueue_extraction(episode_id, body.user_id)
        if not enqueued:
            extraction_status = "enqueue_failed"

    return IngestResponse(
        episode_id=episode_id,
        status="ingested",
        extraction=extraction_status,
        facts_stored=facts_stored,
    )


@router.get("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    user_id: str,
    query: str = "",
    token_budget: int | None = None,
    valid_only: bool = False,
    _api_key: str = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
):
    """Hybrid search → ranked context; include profile.

    By default returns all facts (active and retracted). Retracted facts include
    ``valid_until`` and ``retracted_at``. Pass ``valid_only=true`` for current facts only.
    """
    budget = token_budget or settings.default_token_budget
    memories = await hybrid_retrieve(
        session, query, user_id, token_budget=budget, valid_only=valid_only
    )
    print(f"[DEBUG] retrieve relevance_scores={[m[7] for m in memories]}")
    profile = await get_profile(session, user_id)
    token_count = sum(count_tokens(m[1]) for m in memories)
    return RetrieveResponse(
        memories=[
            RetrievedMemory(
                fact=m[1],
                confidence=m[2],
                valid_from=m[3],
                valid_until=m[4],
                source_episode_id=m[6],
                relevance_score=m[7],
                recorded_at=m[3],
                retracted_at=m[5],
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
    """Soft-retract: close the fact's validity window (never delete the row)."""
    from datetime import datetime
    qdrant = get_qdrant_client()
    ok = await invalidate_memory_by_id(session, memory_id, qdrant)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    now = datetime.utcnow()
    return DeleteMemoryResponse(
        id=memory_id,
        status="invalidated",
        retracted_at=now,
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

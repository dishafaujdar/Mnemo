"""Semantic and hybrid search."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.api.dependencies import get_session, require_api_key
from mnemo.app.models.memory import SearchRequest, SearchResponse, SearchResultItem
from mnemo.app.services.retrieval.bm25_search import bm25_search
from mnemo.app.services.retrieval.vector_search import vector_search


router = APIRouter(prefix="/memory", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    _api_key: str = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
):
    """Semantic search over facts.

    valid_only=true (default): active facts only.
    valid_only=false: all facts including retracted (SQLite audit view).
    """
    if body.valid_only:
        items = await vector_search(
            body.query,
            body.user_id,
            valid_only=True,
            limit=body.limit,
        )
    else:
        items = await bm25_search(
            session,
            body.query,
            body.user_id,
            valid_only=False,
            limit=body.limit,
        )
    return SearchResponse(
        results=[
            SearchResultItem(
                fact=item[1],
                confidence=item[2],
                valid_from=item[3],
                valid_until=item[4],
                source_episode_id=item[6],
                relevance_score=item[7],
                recorded_at=item[3],
                retracted_at=item[5],
            )
            for item in items
        ]
    )

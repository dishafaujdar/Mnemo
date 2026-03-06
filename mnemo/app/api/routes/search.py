"""Semantic and hybrid search."""

from fastapi import APIRouter, Depends

from mnemo.app.api.dependencies import require_api_key
from mnemo.app.models.memory import SearchRequest, SearchResponse, SearchResultItem
from mnemo.app.services.retrieval.vector_search import vector_search


router = APIRouter(prefix="/memory", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    _api_key: str = Depends(require_api_key),
):
    """Semantic search over facts; optional valid_only and include_history."""
    items = await vector_search(
        body.query,
        body.user_id,
        valid_only=body.valid_only,
        limit=body.limit,
    )
    return SearchResponse(
        results=[
            SearchResultItem(
                fact=item[1],
                confidence=item[2],
                valid_at=item[3],
                invalid_at=item[4],
                source_episode_id=item[5],
                relevance_score=item[6],
            )
            for item in items
        ]
    )

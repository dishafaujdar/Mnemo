"""Qdrant client wrapper: local embedded storage, collection init, upsert, search."""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    IsNullCondition,
    MatchValue,
    PayloadField,
    PointStruct,
    VectorParams,
)

from mnemo.app.core.config import settings

COLLECTION_NAME = "mnemo_semantic"
# VECTOR_SIZE = 1536  # text-embedding-3-small
VECTOR_SIZE = 768


def get_qdrant_client() -> AsyncQdrantClient:
    """Create async Qdrant client with local path (embedded mode)."""
    return AsyncQdrantClient(url=settings.qdrant_url)


async def ensure_collection(client: AsyncQdrantClient) -> None:
    """Create collection if it does not exist."""
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if COLLECTION_NAME not in names:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


async def upsert_points(
    client: AsyncQdrantClient,
    points: list[PointStruct],
) -> None:
    """Upsert points into the semantic collection."""
    if not points:
        return
    await client.upsert(collection_name=COLLECTION_NAME, points=points)


async def set_point_payload(
    client: AsyncQdrantClient,
    point_id: str,
    payload: dict[str, Any],
) -> None:
    """Overwrite payload for a point (e.g. set invalid_at when invalidating)."""
    await client.set_payload(
        collection_name=COLLECTION_NAME,
        payload=payload,
        points=[point_id],
    )


async def point_exists(
    client: AsyncQdrantClient,
    point_id: str,
) -> bool:
    """Return True if a point exists in the semantic collection."""
    points = await client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=False,
        with_vectors=False,
    )
    return bool(points)


async def search_semantic(
    client: AsyncQdrantClient,
    query_vector: list[float],
    user_id: str,
    valid_only: bool = True,
    limit: int = 20,
) -> list[tuple[str, float, dict[str, Any]]]:
    """
    Search by vector; filter by user_id and optionally only valid (invalid_at is null) edges.
    Returns list of (point_id, score, payload).
    """
    must: list[Any] = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if valid_only:
        must.append(IsNullCondition(is_null=PayloadField(key="invalid_at")))
    query_filter = Filter(must=must)
    results = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    out: list[tuple[str, float, dict[str, Any]]] = []
    for p in results.points:
        print(f"[DEBUG] p='{p}'")
        pid = str(p.id) if p.id is not None else ""
        score = float(p.score) if p.score is not None else 0.0
        payload = dict(p.payload or {})
        out.append((pid, score, payload))
    return out

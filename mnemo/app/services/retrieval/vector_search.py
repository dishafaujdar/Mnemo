"""Vector (semantic) search via Qdrant for a user's edges."""

from __future__ import annotations

from datetime import datetime

from mnemo.app.db.qdrant import get_qdrant_client, search_semantic
from mnemo.app.services.embeddings import get_embedding

# Same shape as BM25Result: (edge_id, fact_string, confidence, valid_at, invalid_at, episode_id, score)
VectorResult = tuple[str, str, float, datetime, datetime | None, str, float]


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception as e:
        return None


async def vector_search(
    query: str,
    user_id: str,
    valid_only: bool = True,
    limit: int = 20,
    qdrant_client=None,
) -> list[VectorResult]:
    """Semantic search; returns list of (id, fact_string, confidence, valid_at, invalid_at, episode_id, score)."""
    if not query.strip():
        return []
    if qdrant_client is None:
        qdrant_client = get_qdrant_client()
    vector = await get_embedding(query)
    print(f"[DEBUG] query='{query}' vector_norm={sum(x**2 for x in vector)**0.5:.4f}")
    hits = await search_semantic(qdrant_client, vector, user_id, valid_only=valid_only, limit=limit)
    print(f"[DEBUG] hits='{hits}'")
    out: list[VectorResult] = []
    for point_id, score, payload in hits:
        fact_string = (payload.get("fact_string") or "").strip()
        if not fact_string:
            continue
        confidence = float(payload.get("confidence", 1.0))
        valid_at = _parse_dt(payload.get("valid_at")) or datetime.utcnow()
        invalid_at = _parse_dt(payload.get("invalid_at"))
        episode_id = str(payload.get("episode_id", ""))
        out.append((point_id, fact_string, confidence, valid_at, invalid_at, episode_id, score))
    return out

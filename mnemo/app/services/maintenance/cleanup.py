"""Hard-delete expired retracted facts (retention policy)."""

from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.core.config import settings
from mnemo.app.db.models import SemanticEdge
from mnemo.app.db.qdrant import get_qdrant_client
from mnemo.app.services.conflict.temporal import parse_timestamp, utc_now

logger = logging.getLogger(__name__)


async def cleanup_retracted_facts(db: AsyncSession) -> int:
    """Delete facts with retracted_at older than the retention window.

    Only rows with metadata.retracted_at set are eligible. Facts with
    invalid_at but no retracted_at are never deleted.
    """
    cutoff = utc_now() - timedelta(days=settings.retracted_fact_retention_days)
    q = select(SemanticEdge).where(
        func.json_extract(SemanticEdge.metadata_, "$.retracted_at").is_not(None)
    )
    result = await db.execute(q)
    candidates = list(result.scalars().all())

    to_delete: list[SemanticEdge] = []
    for edge in candidates:
        retracted = parse_timestamp((edge.metadata_ or {}).get("retracted_at"))
        if retracted is not None and retracted < cutoff:
            to_delete.append(edge)

    if not to_delete:
        return 0

    qdrant = get_qdrant_client()
    point_ids = [edge.qdrant_id or edge.id for edge in to_delete if edge.qdrant_id or edge.id]
    if point_ids:
        try:
            from qdrant_client.http.models import PointIdsList

            from mnemo.app.db.qdrant import COLLECTION_NAME

            await qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=point_ids),
            )
        except Exception:
            logger.exception("Qdrant delete failed during retracted fact cleanup")

    edge_ids = [edge.id for edge in to_delete]
    await db.execute(delete(SemanticEdge).where(SemanticEdge.id.in_(edge_ids)))
    await db.flush()
    logger.info("cleanup_retracted_facts deleted=%d cutoff=%s", len(edge_ids), cutoff.isoformat())
    return len(edge_ids)

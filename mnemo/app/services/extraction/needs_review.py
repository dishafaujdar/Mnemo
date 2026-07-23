"""Step 4: log rejected/low-confidence facts for manual review."""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.db.models import NeedsReviewFact
from mnemo.app.models.extraction import TripletFact

logger = logging.getLogger(__name__)


async def log_to_needs_review(
    db: AsyncSession,
    *,
    user_id: str,
    episode_id: str,
    fact: TripletFact,
    rejection_reason: str,
) -> None:
    """Persist a fact that was not stored automatically."""
    now = datetime.utcnow()
    row = NeedsReviewFact(
        id=str(uuid4()),
        user_id=user_id,
        episode_id=episode_id,
        subject=fact.subject.lower(),
        relation=fact.relation,
        object=fact.object,
        fact_string=fact.fact_string,
        source_span=fact.source_span or None,
        temporal_status=fact.temporal_status or None,
        confidence=fact.confidence,
        rejection_reason=rejection_reason,
        relation_raw=fact.relation_raw or None,
        created_at=now,
    )
    db.add(row)
    await db.flush()
    logger.info(
        "needs_review user=%s episode=%s reason=%s fact=%s",
        user_id,
        episode_id,
        rejection_reason,
        fact.fact_string,
    )

"""Pre-insert duplicate detection via embedding similarity."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.core.config import settings
from mnemo.app.db.models import SemanticEdge
from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.conflict.semantic import cosine_similarity
from mnemo.app.services.embeddings import get_embedding

logger = logging.getLogger(__name__)


class DuplicateVerdict(BaseModel):
    is_duplicate: bool = Field(description="True if the two facts express the same information")


async def _llm_confirm_duplicate(fact_a: str, fact_b: str) -> bool:
    from mnemo.app.core.config import settings as cfg

    api_key = cfg.groq_api_key or cfg.openai_api_key
    if not api_key:
        return False
    try:
        import instructor
        from openai import AsyncOpenAI

        client = instructor.from_openai(
            AsyncOpenAI(api_key=api_key, base_url=cfg.groq_base_url),
            mode=instructor.Mode.JSON,
        )
        result: DuplicateVerdict = await client.chat.completions.create(
            model=cfg.extraction_model,
            response_model=DuplicateVerdict,
            max_retries=1,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Decide if two memory facts express the same information "
                        "(duplicate) or meaningfully different facts."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Fact A: {fact_a}\nFact B: {fact_b}",
                },
            ],
        )
        return result.is_duplicate
    except Exception as exc:
        logger.warning("LLM duplicate check failed: %s", exc)
        return False


async def should_skip_as_duplicate(
    db: AsyncSession,
    user_id: str,
    fact: TripletFact,
) -> bool:
    """Return True if an active duplicate already exists (check before INSERT)."""
    q = select(SemanticEdge).where(
        and_(
            SemanticEdge.user_id == user_id,
            SemanticEdge.relation == fact.relation.upper(),
            SemanticEdge.invalid_at.is_(None),
        )
    )
    result = await db.execute(q)
    existing = list(result.scalars().all())
    if not existing:
        return False

    obj_lower = fact.object.strip().lower()
    for edge in existing:
        if edge.object.strip().lower() == obj_lower:
            return True

    new_vec = await get_embedding(fact.fact_string)
    skip_threshold = settings.dedup_skip_threshold
    llm_threshold = settings.dedup_llm_threshold

    for edge in existing:
        if edge.relation.upper() != fact.relation.upper():
            continue
        existing_vec = await get_embedding(edge.fact_string)
        sim = cosine_similarity(new_vec, existing_vec)
        if sim > skip_threshold:
            return True
        if sim >= llm_threshold:
            if await _llm_confirm_duplicate(fact.fact_string, edge.fact_string):
                return True
    return False

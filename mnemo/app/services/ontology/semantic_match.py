"""Semantic relation matching via embeddings + optional LLM fallback."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from mnemo.app.core.config import settings
from mnemo.app.services.embeddings import get_embedding
from mnemo.app.services.ontology.canonical import CANONICAL_RELATIONS
from mnemo.app.services.ontology.manager import MatchResult, TIER_CONFIRMED, TIER_REJECT, TIER_UNKNOWN
from mnemo.app.services.ontology.seed import DEFAULT_BEHAVIOR, SEED_BEHAVIOR

logger = logging.getLogger(__name__)

_description_vectors: dict[str, list[float]] | None = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


async def _ensure_description_vectors() -> dict[str, list[float]]:
    global _description_vectors
    if _description_vectors is not None:
        return _description_vectors
    vectors: dict[str, list[float]] = {}
    for relation, description in CANONICAL_RELATIONS.items():
        text = f"{relation.replace('_', ' ').lower()}: {description}"
        vectors[relation] = await get_embedding(text)
    _description_vectors = vectors
    return vectors


class RelationPick(BaseModel):
    relation: str | None = Field(
        default=None,
        description="Best matching canonical relation token, or null if none fit",
    )


async def _llm_pick_relation(relation_raw: str) -> str | None:
    api_key = settings.groq_api_key or settings.openai_api_key
    if not api_key:
        return None
    try:
        import instructor
        from openai import AsyncOpenAI

        client = instructor.from_openai(
            AsyncOpenAI(api_key=api_key, base_url=settings.groq_base_url),
            mode=instructor.Mode.JSON,
        )
        options = "\n".join(
            f"- {rel}: {desc}" for rel, desc in CANONICAL_RELATIONS.items()
        )
        result: RelationPick = await client.chat.completions.create(
            model=settings.extraction_model,
            response_model=RelationPick,
            max_retries=1,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Pick the single best canonical relation for the raw relation phrase, "
                        "or return null if none apply.\n"
                        f"Options:\n{options}"
                    ),
                },
                {"role": "user", "content": f"Raw relation: {relation_raw!r}"},
            ],
        )
        if result.relation and result.relation.upper() in CANONICAL_RELATIONS:
            return result.relation.upper()
    except Exception as exc:
        logger.warning("LLM relation pick failed for %r: %s", relation_raw, exc)
    return None


async def semantic_normalize(relation_raw: str) -> MatchResult:
    """Map a raw relation to a canonical relation using embeddings (+ LLM fallback)."""
    raw = relation_raw.strip()
    if not raw:
        return MatchResult("", relation_raw, 0.0, TIER_REJECT, DEFAULT_BEHAVIOR)

    probe = await get_embedding(raw.replace("_", " "))
    vectors = await _ensure_description_vectors()

    best_rel = ""
    best_score = 0.0
    for relation, vector in vectors.items():
        score = _cosine_similarity(probe, vector)
        if score > best_score:
            best_rel, best_score = relation, score

    threshold = settings.ontology_semantic_threshold
    if best_score >= threshold and best_rel:
        return MatchResult(
            best_rel,
            raw,
            round(best_score, 3),
            TIER_CONFIRMED,
            SEED_BEHAVIOR.get(best_rel, DEFAULT_BEHAVIOR),
        )

    llm_rel = await _llm_pick_relation(raw)
    if llm_rel:
        return MatchResult(
            llm_rel,
            raw,
            round(best_score, 3),
            TIER_CONFIRMED,
            SEED_BEHAVIOR.get(llm_rel, DEFAULT_BEHAVIOR),
        )

    return MatchResult(
        raw.upper().replace(" ", "_"),
        raw,
        round(best_score, 3),
        TIER_UNKNOWN,
        DEFAULT_BEHAVIOR,
    )

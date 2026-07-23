"""OpenAI-compatible embeddings for fact_string (Ollama local or hosted API)."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from mnemo.app.core.config import settings
from mnemo.app.db.qdrant import VECTOR_SIZE

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.embedding_api_key or "ollama",
            base_url=settings.embedding_base_url,
        )
    return _client


async def get_embedding(text: str) -> list[float]:
    """Return embedding vector for text. Falls back to zero vector on failure."""
    if not text.strip():
        return [0.0] * VECTOR_SIZE
    client = _get_client()
    try:
        r = await client.embeddings.create(
            model=settings.embedding_model,
            input=text.strip()[:8192],
        )
        if r.data and len(r.data) > 0:
            return list(r.data[0].embedding)
    except Exception as exc:
        logger.warning(
            "embedding failed (model=%s base_url=%s): %s",
            settings.embedding_model,
            settings.embedding_base_url,
            exc,
        )
    return [0.0] * VECTOR_SIZE

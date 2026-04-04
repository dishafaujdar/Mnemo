"""OpenAI embeddings for fact_string (used by semantic store and retrieval)."""

from __future__ import annotations

from openai import AsyncOpenAI

from mnemo.app.core.config import settings
from mnemo.app.db.qdrant import VECTOR_SIZE

_client: AsyncOpenAI | None = None


# def _get_client() -> AsyncOpenAI:
#     global _client
#     if _client is None:
#         _client = AsyncOpenAI(api_key=settings.openai_api_key)
#     return _client

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key="ollama",  # arbitrary, Ollama ignores it
            base_url="http://host.docker.internal:11434/v1",  # reach host from Docker
        )
    return _client


async def get_embedding(text: str) -> list[float]:
    """Return embedding vector for text (length VECTOR_SIZE). Returns zero vector if no API key."""
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
    except Exception as e:
        print(f"[DEBUG] embedding error: {e}")
    return [0.0] * VECTOR_SIZE

"""Token budget manager: pack ranked memories into a token limit."""

from __future__ import annotations

import tiktoken

from mnemo.app.core.config import settings


def get_encoding():
    """Encoding for token counting (cl100k_base for OpenAI models)."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Approximate token count for a string."""
    enc = get_encoding()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


# One memory item: (edge_id, fact_string, confidence, valid_at, invalid_at, episode_id, score)
RetrievalItem = tuple[str, str, float, object, object, str, float]


def fit(
    memories: list[RetrievalItem],
    budget: int | None = None,
) -> list[RetrievalItem]:
    """
    Greedily pack memories into token budget. Assumes already ranked (e.g. by RRF).
    Each item: (edge_id, fact_string, confidence, valid_at, invalid_at, episode_id, score).
    """
    budget = budget or settings.default_token_budget
    result: list[RetrievalItem] = []
    used = 0
    for item in memories:
        _id, fact, conf, valid_at, invalid_at, episode_id, score = item
        tokens = count_tokens(fact)
        if used + tokens <= budget:
            result.append(item)
            used += tokens
        else:
            break
    return result

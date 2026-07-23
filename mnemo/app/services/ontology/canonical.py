"""Canonical relations for semantic matching."""

from __future__ import annotations

CANONICAL_RELATIONS: dict[str, str] = {
    "WORKS_AT": "current employer or organization the user works for",
    "WORKED_AT": "past employer the user no longer works for",
    "HAS_ROLE": "current job title or professional role",
    "LIVES_IN": "current location or city of residence",
    "USES": "tool, programming language, or technology currently used",
    "PREFERS": "preference or favorite among options",
}

# Never stored (BORN_IN additionally gated by explicit source text elsewhere).
BLACKLISTED_RELATIONS: frozenset[str] = frozenset(
    {
        "BORN_IN",
        "SWITCHED_FROM",
        "SWITCHED_TO",
    }
)

MAX_OBJECT_WORDS_STORE = 5
MAX_OBJECT_WORDS_EXTRACT = 3


def normalize_object(value: str, *, max_words: int = MAX_OBJECT_WORDS_EXTRACT) -> str:
    """Trim object to a single entity (max N words)."""
    words = value.strip().strip(".,;:!?").split()
    if not words:
        return ""
    return " ".join(words[:max_words])


def object_word_count(value: str) -> int:
    return len(value.strip().split())


def is_blacklisted_relation(relation: str) -> bool:
    return relation.upper() in BLACKLISTED_RELATIONS

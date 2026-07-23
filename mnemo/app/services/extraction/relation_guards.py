"""Step 5: relation-specific validation guards."""

from __future__ import annotations

import re

from mnemo.app.models.extraction import TripletFact

# Objects that must never be LIVES_IN targets.
_NON_LOCATION_OBJECTS = frozenset(
    {
        "cat",
        "cats",
        "dog",
        "dogs",
        "pet",
        "pets",
        "guitar",
        "piano",
        "car",
        "bike",
        "luna",
        "milo",
    }
)

_LOCATION_LIKE = re.compile(
    r"\b(?:city|town|state|country|street|avenue|delhi|london|paris|"
    r"san francisco|new york|bangalore|mumbai|berlin|tokyo)\b",
    re.IGNORECASE,
)


def gliner_entity_type(object_text: str) -> str | None:
    """Lightweight entity-type hint for an object string."""
    lower = object_text.strip().lower()
    if lower in _NON_LOCATION_OBJECTS or lower.split()[0] in _NON_LOCATION_OBJECTS:
        return "animal" if lower in {"cat", "cats", "dog", "dogs", "pet", "pets", "luna", "milo"} else "object"
    if _LOCATION_LIKE.search(object_text):
        return "location"
    # Capitalized multi-word or single token place names
    words = object_text.split()
    if len(words) == 1 and object_text[0].isupper():
        return "location"
    return None


def validate_lives_in(fact: TripletFact) -> bool:
    """Reject LIVES_IN when the object is not location-like."""
    if fact.relation.upper() != "LIVES_IN":
        return True
    obj = fact.object.strip()
    etype = gliner_entity_type(obj)
    if etype is not None and etype != "location":
        return False
    lower = obj.lower()
    if lower in _NON_LOCATION_OBJECTS or lower.split()[0] in _NON_LOCATION_OBJECTS:
        return False
    # "live with cats" pattern in source span
    span = fact.source_span.lower()
    if "live with" in span or "lives with" in span:
        return False
    return True


def should_use_has_pet(source_span: str) -> bool:
    """Detect pet cohabitation phrasing."""
    lower = source_span.lower()
    return bool(re.search(r"\b(?:live|lives|living)\s+with\b", lower)) and bool(
        re.search(r"\b(?:cat|cats|dog|dogs|pet|pets|luna|milo)\b", lower)
    )

"""Step 3: grounding validation — source_span must exist in original text."""

from __future__ import annotations

import logging
import re

from mnemo.app.models.extraction import TripletFact

logger = logging.getLogger(__name__)


def validate_grounding(fact: TripletFact, original_text: str) -> bool:
    """Return True when source_span is a verbatim substring of the input."""
    span = fact.source_span.strip()
    if not span:
        return False
    return span.lower() in original_text.strip().lower()


def infer_source_span(original_text: str, fact: TripletFact) -> str:
    """Best-effort source span when the extractor omitted it."""
    if fact.source_span.strip():
        return fact.source_span.strip()
    lower = original_text.lower()
    obj = fact.object.strip()
    if not obj:
        return ""
    obj_lower = obj.lower()
    idx = lower.find(obj_lower)
    if idx >= 0:
        # Expand to clause boundaries lightly
        start = max(0, lower.rfind(".", 0, idx) + 1, lower.rfind(",", 0, idx) + 1)
        end = lower.find(".", idx)
        if end == -1:
            end = len(original_text)
        return original_text[start:end].strip()
    # token fallback
    token = obj_lower.split()[0]
    if token and token in lower:
        pos = lower.find(token)
        window = original_text[max(0, pos - 40) : pos + len(token) + 40]
        return window.strip()
    return ""


def ensure_source_spans(facts: list[TripletFact], original_text: str) -> list[TripletFact]:
    """Fill missing source_span fields before grounding validation."""
    updated: list[TripletFact] = []
    for fact in facts:
        span = infer_source_span(original_text, fact)
        updated.append(fact.model_copy(update={"source_span": span}))
    return updated

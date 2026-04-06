"""Hybrid extraction: spaCy first, LLM fallback when needed; merge and dedupe."""

from __future__ import annotations

import re

from mnemo.app.core.config import settings
from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.extraction.llm_extractor import extract as llm_extract
from mnemo.app.services.extraction.spacy_extractor import extract as spacy_extract
from mnemo.app.services.extraction.spacy_extractor import get_nlp

LOW_CONFIDENCE_THRESHOLD = 0.78
COMPLEX_LANGUAGE_PATTERN = re.compile(
    r"\b(?:prefer|love|hate|always|never|no longer|used to|switched to|switched from|goal is|goal are|but now)\b",
    re.IGNORECASE,
)


def _average_confidence(facts: list[TripletFact]) -> float:
    if not facts:
        return 0.0
    return sum(f.confidence for f in facts) / len(facts)


def _needs_llm_pass(facts: list[TripletFact], content: str) -> bool:
    """Trigger LLM when spaCy coverage or confidence is weak, or the language is update-heavy."""
    if not facts:
        return True
    if _average_confidence(facts) < LOW_CONFIDENCE_THRESHOLD:
        return True
    if len(facts) < 2:
        return True
    if COMPLEX_LANGUAGE_PATTERN.search(content) and len(facts) < 3:
        return True
    return False


def _merge_facts(spacy_facts: list[TripletFact], llm_facts: list[TripletFact]) -> list[TripletFact]:
    """Merge and dedupe by (subject, relation, object); keep higher confidence."""
    by_key: dict[tuple[str, str, str], TripletFact] = {}
    for f in spacy_facts:
        key = (f.subject.lower(), f.relation.upper(), f.object.lower())
        by_key[key] = f
    for f in llm_facts:
        key = (f.subject.lower(), f.relation.upper(), f.object.lower())
        if key not in by_key or f.confidence > by_key[key].confidence:
            by_key[key] = f
    return list(by_key.values())


async def extract_facts(content: str, spacy_model_name: str | None = None) -> list[TripletFact]:
    """
    Hybrid extraction: run spaCy first; if confidence/coverage is low or
    preference language detected, run LLM extraction; return merged, deduplicated triplets.
    """
    if not content or not content.strip():
        return []
    model_name = spacy_model_name or settings.spacy_model
    nlp = get_nlp(model_name) if model_name else None
    spacy_facts = spacy_extract(content, spacy_nlp=nlp)
    if not _needs_llm_pass(spacy_facts, content):
        return spacy_facts
    llm_facts = await llm_extract(content)
    return _merge_facts(spacy_facts, llm_facts)

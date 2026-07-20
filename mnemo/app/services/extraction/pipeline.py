"""Two-stage extraction orchestration.

Flow (see module ``gliner_extractor``, ``structured_extractor``, ``judge``):

    raw text
      -> STEP 1: GLiNER2 (fast, local) with confidence routing
           - confidence >= gliner_high_confidence  -> accept directly
           - confidence <  gliner_high_confidence   -> send to Step 2
      -> STEP 2: Instructor + Groq structured re-extraction (only if needed)
           - judge score gates store / review / discard
      -> merge + dedupe (LLM wins ties: it validated against the raw text)

Step 3 (bi-temporal resolution) happens in ``conflict.resolver``.
"""

from __future__ import annotations

import logging

from mnemo.app.core.config import settings
from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.extraction import gliner_extractor, structured_extractor
from mnemo.app.services.extraction.judge import apply_judge

logger = logging.getLogger(__name__)


def _dedupe_key(fact: TripletFact) -> tuple[str, str, str]:
    return (fact.subject.lower(), fact.relation.upper(), fact.object.lower())


def _merge_facts(
    gliner_facts: list[TripletFact],
    llm_facts: list[TripletFact],
) -> list[TripletFact]:
    """Merge two fact lists, deduped by (subject, relation, object).

    LLM facts take precedence on collision because they were validated against
    the raw user text; otherwise keep the higher-confidence fact.
    """
    by_key: dict[tuple[str, str, str], TripletFact] = {}
    for fact in gliner_facts:
        by_key[_dedupe_key(fact)] = fact
    for fact in llm_facts:
        key = _dedupe_key(fact)
        existing = by_key.get(key)
        if existing is None or fact.source == "llm" or fact.confidence > existing.confidence:
            by_key[key] = fact
    return list(by_key.values())


def _route_gliner(
    facts: list[TripletFact],
) -> tuple[list[TripletFact], list[TripletFact]]:
    """Split GLiNER facts into (high-confidence accept, needs-LLM)."""
    high: list[TripletFact] = []
    low: list[TripletFact] = []
    for fact in facts:
        if fact.confidence >= settings.gliner_high_confidence:
            high.append(fact)
        else:
            low.append(fact)
    return high, low


async def extract_facts(content: str, user_id: str | None = None) -> list[TripletFact]:
    """Run the two-stage extraction pipeline and return validated triplets."""
    if not content or not content.strip():
        return []

    # STEP 1: fast local extraction.
    gliner_facts = gliner_extractor.extract(content)
    high_conf, low_conf = _route_gliner(gliner_facts)

    # STEP 2: LLM re-extraction when GLiNER is unsure or found nothing.
    needs_llm = bool(low_conf) or not gliner_facts
    llm_facts: list[TripletFact] = []
    if needs_llm:
        raw_llm = await structured_extractor.extract(content, gliner_facts)
        llm_facts = apply_judge(raw_llm)

    # When the LLM validated facts, it supersedes the low-confidence GLiNER
    # candidates. If the LLM stage produced nothing (unavailable/failed), keep
    # the low-confidence GLiNER facts so we never silently drop information.
    base = high_conf if (needs_llm and llm_facts) else high_conf + low_conf
    merged = _merge_facts(base, llm_facts)
    logger.debug(
        "extraction: gliner=%d (high=%d low=%d) llm=%d merged=%d",
        len(gliner_facts),
        len(high_conf),
        len(low_conf),
        len(llm_facts),
        len(merged),
    )
    return merged

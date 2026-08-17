"""Multi-step extraction orchestration.

Step 1 — Temporal classification per clause (``temporal_classifier``)
Step 2 — Fact extraction (GLiNER2 + focused LLM)
Step 3 — Grounding validation (``grounding``) — in ``service.gate_facts``
Step 4 — Confidence gate + needs_review — in ``service.gate_facts``

Bi-temporal resolution happens in ``conflict.resolver`` after gating.
"""

from __future__ import annotations

import logging

from mnemo.app.core.config import settings
from mnemo.app.models.extraction import ExtractionResult, TripletFact
from mnemo.app.services.extraction import gliner_extractor, structured_extractor
from mnemo.app.services.extraction.grounding import ensure_source_spans
from mnemo.app.services.extraction.judge import apply_judge, review_status_for_tier
from mnemo.app.services.extraction.temporal_apply import apply_temporal_metadata
from mnemo.app.services.extraction.temporal_classifier import classify_clauses
from mnemo.app.services.extraction.temporal_signals import apply_temporal_signals
from mnemo.app.services.ontology.canonical import normalize_object
from mnemo.app.services.ontology.manager import TIER_UNKNOWN, get_ontology

logger = logging.getLogger(__name__)


def _dedupe_key(fact: TripletFact) -> tuple[str, str, str]:
    return (fact.subject.lower(), fact.relation.upper(), fact.object.lower())


def _merge_facts(
    gliner_facts: list[TripletFact],
    llm_facts: list[TripletFact],
) -> list[TripletFact]:
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
    high: list[TripletFact] = []
    low: list[TripletFact] = []
    for fact in facts:
        if fact.confidence >= settings.gliner_high_confidence:
            high.append(fact)
        else:
            low.append(fact)
    return high, low


async def _normalize_merged_facts(facts: list[TripletFact]) -> list[TripletFact]:
    ontology = get_ontology()
    normalized: list[TripletFact] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        raw = fact.relation_raw or fact.relation
        match = await ontology.normalize_async(raw)
        if match.is_rejected or match.tier == TIER_UNKNOWN:
            continue
        obj = normalize_object(fact.object)
        if not obj:
            continue
        key = (fact.subject.lower(), match.relation, obj.lower())
        if key in seen:
            continue
        seen.add(key)
        rel_phrase = match.relation.lower().replace("_", " ")
        normalized.append(
            fact.model_copy(
                update={
                    "relation": match.relation,
                    "object": obj,
                    "fact_string": f"{fact.subject} {rel_phrase} {obj}",
                    "relation_raw": raw,
                    "relation_match_score": match.match_score,
                    "review_status": review_status_for_tier(match.tier),
                }
            )
        )
    return normalized


async def extract_facts(content: str, user_id: str | None = None) -> ExtractionResult:
    """Run the multi-step extraction pipeline."""
    if not content or not content.strip():
        return ExtractionResult()

    # STEP 1 — temporal classification per clause
    classified = await classify_clauses(content)

    # STEP 2 — fact extraction (GLiNER + focused LLM)
    gliner_facts = gliner_extractor.extract(content)
    high_conf, low_conf = _route_gliner(gliner_facts)

    needs_llm = bool(low_conf) or not gliner_facts
    llm_facts: list[TripletFact] = []
    retract_others: str | None = None
    if needs_llm:
        logger.info(
            "Pipeline invoking LLM (gliner_high=%d gliner_low=%d)",
            len(high_conf),
            len(low_conf),
        )
        llm_result = await structured_extractor.extract(content, gliner_facts, classified)
        llm_facts = apply_judge(llm_result.facts)
        retract_others = llm_result.retract_others_in_category
    else:
        logger.info(
            "Pipeline skipping LLM — all %d GLiNER facts above confidence threshold",
            len(high_conf),
        )

    base = high_conf if (needs_llm and llm_facts) else high_conf + low_conf
    merged = _merge_facts(base, llm_facts)
    merged = await _normalize_merged_facts(merged)
    merged = ensure_source_spans(merged, content)
    merged = apply_temporal_metadata(merged, classified, content)
    merged, retract_others = apply_temporal_signals(
        content, merged, retract_others_in_category=retract_others
    )

    logger.debug(
        "extraction: clauses=%d gliner=%d llm=%d merged=%d retract_others=%s",
        len(classified),
        len(gliner_facts),
        len(llm_facts),
        len(merged),
        retract_others,
    )
    return ExtractionResult(facts=merged, retract_others_in_category=retract_others)

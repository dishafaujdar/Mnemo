"""Structured LLM fact extraction (Step 2 only — no temporal reasoning).

Temporal status is classified separately in ``temporal_classifier`` and applied
after extraction. This module focuses solely on extracting grounded triplets
with a mandatory ``source_span``.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from mnemo.app.core.config import settings
from mnemo.app.models.extraction import ExtractionResult, TripletFact
from mnemo.app.services.extraction.judge import review_status_for_tier
from mnemo.app.services.extraction.relation_guards import should_use_has_pet
from mnemo.app.services.extraction.temporal_classifier import ClassifiedClause
from mnemo.app.services.ontology.canonical import normalize_object
from mnemo.app.services.ontology.manager import TIER_UNKNOWN, get_ontology
from mnemo.app.services.ontology.seed import SEED_RELATIONS

from mnemo.app.services.extraction.llm_client import (
    get_instructor_client,
    llm_api_key,
    llm_key_format_warning,
    log_llm_failure,
    log_llm_skip,
)

logger = logging.getLogger(__name__)

_FIRST_PERSON = {"i", "i'm", "im", "me", "my", "myself", "we", "us", "our", ""}


class LLMFact(BaseModel):
    subject: str = Field(min_length=1, description="Fact subject; use 'user' for the speaker")
    relation_raw: str = Field(min_length=1, description="Relation as phrased in text")
    object: str = Field(min_length=1, description="Single entity, max 3 words")
    source_span: str = Field(
        min_length=1,
        description="Verbatim substring from the raw user text supporting this fact",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="0.90-1.00 directly stated")


class LLMFactList(BaseModel):
    facts: list[LLMFact] = Field(default_factory=list, max_length=20)
    unknown_relations: list[str] = Field(default_factory=list)
    retract_others_in_category: str | None = None


def _system_prompt() -> str:
    relations = ", ".join(sorted(SEED_RELATIONS))
    return (
        "You extract atomic facts from user text for a memory system.\n"
        "The RAW USER TEXT is the only ground truth.\n\n"
        "For EVERY fact you MUST provide source_span — an exact verbatim copy of a "
        "substring from the raw text that supports the fact. If you cannot copy such "
        "a substring, do not emit the fact.\n\n"
        "Object rules:\n"
        "- Single entity only, maximum 3 words.\n"
        "- Never put full sentences in object.\n\n"
        "Relation rules:\n"
        "- LIVES_IN is ONLY for cities/places of residence (e.g. 'live in Delhi').\n"
        "- 'live with two cats' / pets → HAS_PET (object = pet name), NEVER LIVES_IN.\n"
        "- Past activities → WORKED_AT / PLAYED / appropriate past relation.\n"
        "- Current activities → present-tense relations (USES, LEARNING, PLAYS, etc.).\n"
        "- Learning guitar → LEARNING relation, one fact only.\n\n"
        f"Known relations: {relations}.\n"
        "Do NOT classify temporal status — extraction only."
    )


def _user_prompt(
    content: str,
    gliner_facts: list[TripletFact],
    classified: list[ClassifiedClause],
) -> str:
    clause_lines = "\n".join(
        f"- [{c.temporal_status.upper()}] {c.text}" for c in classified
    )
    if gliner_facts:
        found = "\n".join(
            f"- {f.subject} [{f.relation_raw or f.relation}] {f.object} "
            f"(gliner_confidence={f.confidence:.2f})"
            for f in gliner_facts
        )
    else:
        found = "(none)"
    return (
        f"RAW USER TEXT:\n\"\"\"\n{content.strip()}\n\"\"\"\n\n"
        f"Clause temporal labels (metadata only — do not re-classify):\n{clause_lines}\n\n"
        f"GLiNER hints:\n{found}\n\n"
        "Extract facts with source_span copied verbatim from the raw text."
    )


def _normalize_subject(subject: str) -> str:
    value = subject.strip()
    return "user" if value.lower() in _FIRST_PERSON else value


def _maybe_rewrite_pet_relation(relation: str, obj: str, source_span: str) -> str:
    if relation.upper() == "LIVES_IN" and should_use_has_pet(source_span):
        return "HAS_PET"
    return relation


async def extract(
    content: str,
    gliner_facts: list[TripletFact] | None = None,
    classified: list[ClassifiedClause] | None = None,
) -> ExtractionResult:
    if not content.strip():
        return ExtractionResult()

    if not llm_api_key():
        log_llm_skip("no API key — structured extraction skipped")
        return ExtractionResult()

    warn = llm_key_format_warning()
    if warn:
        logger.warning(warn)

    gliner_facts = gliner_facts or []
    classified = classified or []
    logger.info(
        "LLM extraction starting (model=%s, clauses=%d, gliner_hints=%d)",
        settings.extraction_model,
        len(classified),
        len(gliner_facts),
    )
    try:
        client = get_instructor_client()
        result: LLMFactList = await client.chat.completions.create(
            model=settings.extraction_model,
            response_model=LLMFactList,
            max_retries=2,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(content, gliner_facts, classified)},
            ],
        )
    except Exception as exc:
        log_llm_failure("structured_extraction", exc)
        return ExtractionResult()

    logger.info(
        "LLM extraction response: raw_facts=%d retract_others=%s",
        len(result.facts),
        result.retract_others_in_category,
    )
    for i, item in enumerate(result.facts):
        logger.info(
            "LLM fact[%d]: %s [%s] %s conf=%.2f span=%r",
            i,
            item.subject,
            item.relation_raw,
            item.object,
            item.confidence,
            item.source_span[:80],
        )

    ontology = get_ontology()
    facts: list[TripletFact] = []
    seen: set[tuple[str, str, str]] = set()
    for item in result.facts:
        span = item.source_span.strip()
        if span.lower() not in content.lower():
            continue
        match = await ontology.normalize_async(item.relation_raw)
        if match.is_rejected or match.tier == TIER_UNKNOWN:
            continue
        relation = _maybe_rewrite_pet_relation(match.relation, item.object, span)
        subject = _normalize_subject(item.subject)
        obj = normalize_object(item.object)
        if not obj:
            continue
        key = (subject.lower(), relation, obj.lower())
        if key in seen:
            continue
        seen.add(key)
        rel_phrase = relation.lower().replace("_", " ")
        facts.append(
            TripletFact(
                subject=subject,
                relation=relation,
                object=obj,
                fact_string=f"{subject} {rel_phrase} {obj}",
                source_span=span,
                confidence=round(float(item.confidence), 3),
                relation_raw=item.relation_raw,
                relation_match_score=match.match_score,
                review_status=review_status_for_tier(match.tier),
                source="llm",
            )
        )
    logger.info("LLM extraction accepted %d/%d facts after grounding", len(facts), len(result.facts))
    return ExtractionResult(
        facts=facts,
        retract_others_in_category=result.retract_others_in_category,
    )

"""LLM-based triplet extraction with OpenAI structured output."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from mnemo.app.core.config import settings
from mnemo.app.models.extraction import TripletFact

logger = logging.getLogger(__name__)
_client: AsyncOpenAI | None = None

ALLOWED_RELATIONS = {
    "PREFERS",
    "DISLIKES",
    "WORKS_ON",
    "USES",
    "KNOWS",
    "IS",
    "HAS",
    "SWITCHED_FROM",
    "SWITCHED_TO",
    "BUILDING",
    "STRUGGLES_WITH",
    "LEARNED",
    "WORKS_AT",
    "GOAL_IS",
}

EXTRACTION_SYSTEM_PROMPT = """You are a fact extractor for a memory system used by a coding assistant.

Extract factual triplets from the conversation turn.
Focus on:
- User preferences (languages, frameworks, tools, editors)
- User's current projects and goals
- Technical context (stack, architecture decisions)
- Personal facts (name, role, experience level)
- Explicit user-stated facts from the current conversation turn

Return ONLY facts explicitly stated or strongly implied.
Do NOT invent or assume facts.
Never emit generic knowledge facts from named entities alone.
Use subject "user" unless the conversation explicitly states a different subject for a memory-worthy fact.
Prefer omitting a fact over returning a weak or ambiguous one.

Output as a JSON object with a top-level "facts" array.
Each fact must have: subject, relation, object, fact_string, confidence (0-1).
Relation types to use: PREFERS, DISLIKES, WORKS_ON, USES, KNOWS, IS, HAS, SWITCHED_FROM, SWITCHED_TO,
BUILDING, STRUGGLES_WITH, LEARNED, WORKS_AT, GOAL_IS."""


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    return _client


def _normalize_subject(subject: str) -> str:
    normalized = subject.strip().lower()
    if normalized in {"", "i", "me", "myself", "we", "us", "my", "our"}:
        return "user"
    return subject.strip() or "user"


def _normalize_relation(relation: str) -> str:
    return relation.strip().upper().replace(" ", "_")


def _normalize_object(value: str) -> str:
    return " ".join(value.strip().strip(".,;:!?").split())[:512]


def _normalize_fact_string(subject: str, relation: str, obj: str, provided: str) -> str:
    fact_string = " ".join(provided.strip().split())
    if fact_string:
        return fact_string[:512]
    return f"{subject} {relation.lower().replace('_', ' ')} {obj}"[:512]


def _dedupe_facts(facts: list[TripletFact]) -> list[TripletFact]:
    by_key: dict[tuple[str, str, str], TripletFact] = {}
    for fact in facts:
        key = (fact.subject.lower(), fact.relation.upper(), fact.object.lower())
        if key not in by_key or fact.confidence > by_key[key].confidence:
            by_key[key] = fact
    return list(by_key.values())


def _skeptical_confidence(raw_confidence: float, relation: str, obj: str) -> float:
    confidence = max(0.0, min(1.0, raw_confidence))
    confidence -= 0.08
    if relation == "IS" and len(obj.split()) > 4:
        confidence -= 0.05
    if len(obj) > 64:
        confidence -= 0.04
    return max(0.0, min(0.95, confidence))


async def extract(content: str) -> list[TripletFact]:
    """Call OpenAI to extract triplets; return parsed list. Returns [] if no API key or on error."""
    if not content.strip() or not settings.openai_api_key:
        logger.error("OPENAI_API_KEY not set — LLM extraction skipped")
        return []
    client = _get_client()
    try:
        resp = await client.chat.completions.create(
            model=settings.extraction_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.exception(f"OpenAI call failed: {e}")
        return []
    choice = resp.choices[0] if resp.choices else None
    if not choice or not choice.message or not choice.message.content:
        logger.warning("Empty response from OpenAI")
        return []
    raw = choice.message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON response: {raw}")
        return []
    # Accept either {"facts": [...]} or direct array
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "facts" in data:
        items = data["facts"]
    else:
        items = data.get("items", data.get("results", []))
    if not isinstance(items, list):
        logger.error(f"Invalid items type: {type(items)}")
        return []
    facts: list[TripletFact] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            subj = _normalize_subject(str(item.get("subject", "user")))
            rel = _normalize_relation(str(item.get("relation", "")))
            obj = _normalize_object(str(item.get("object", "")))
            fact_str = _normalize_fact_string(subj, rel, obj, str(item.get("fact_string", "")))
            conf = _skeptical_confidence(float(item.get("confidence", 0.9)), rel, obj)
            if not rel or rel not in ALLOWED_RELATIONS or not obj:
                continue
            facts.append(
                TripletFact(
                    subject=subj,
                    relation=rel,
                    object=obj,
                    fact_string=fact_str,
                    confidence=conf,
                )
            )
        except (TypeError, ValueError):
            logger.exception(f"Error processing item: {item}")
            continue
    return _dedupe_facts(facts)

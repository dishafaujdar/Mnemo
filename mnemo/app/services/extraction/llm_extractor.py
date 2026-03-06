"""LLM-based triplet extraction with OpenAI structured output."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from mnemo.app.core.config import settings
from mnemo.app.models.extraction import TripletFact

EXTRACTION_SYSTEM_PROMPT = """You are a fact extractor for a memory system used by a coding assistant.

Extract factual triplets from the conversation turn.
Focus on:
- User preferences (languages, frameworks, tools, editors)
- User's current projects and goals
- Technical context (stack, architecture decisions)
- Personal facts (name, role, experience level)
- Contradictions or updates to previously stated facts

Return ONLY facts explicitly stated or strongly implied.
Do NOT invent or assume facts.

Output as a JSON array. Each element must have: subject, relation, object, fact_string, confidence (0-1).
Relation types to use: PREFERS, DISLIKES, WORKS_ON, USES, KNOWS, IS, HAS, SWITCHED_FROM, SWITCHED_TO,
BUILDING, STRUGGLES_WITH, LEARNED, WORKS_AT, GOAL_IS."""


async def extract(content: str) -> list[TripletFact]:
    """Call OpenAI to extract triplets; return parsed list. Returns [] if no API key or on error."""
    if not content.strip() or not settings.openai_api_key:
        return []
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        resp = await client.chat.completions.create(
            model=settings.extraction_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
    except Exception:
        return []
    choice = resp.choices[0] if resp.choices else None
    if not choice or not choice.message or not choice.message.content:
        return []
    raw = choice.message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    # Accept either {"facts": [...]} or direct array
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "facts" in data:
        items = data["facts"]
    else:
        items = data.get("items", data.get("results", []))
    if not isinstance(items, list):
        return []
    facts: list[TripletFact] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            subj = str(item.get("subject", "user")).strip() or "user"
            rel = str(item.get("relation", "")).strip().upper().replace(" ", "_")
            obj = str(item.get("object", "")).strip()
            fact_str = str(item.get("fact_string", f"{subj} {rel} {obj}")).strip()
            conf = float(item.get("confidence", 0.9))
            conf = max(0.0, min(1.0, conf))
            if not rel or not obj:
                continue
            facts.append(
                TripletFact(
                    subject=subj,
                    relation=rel,
                    object=obj[:512],
                    fact_string=fact_str or f"User {rel.lower()} {obj}",
                    confidence=conf,
                )
            )
        except (TypeError, ValueError):
            continue
    return facts

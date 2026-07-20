"""Structured LLM re-extraction with Instructor + Groq (Step 2).

Used when GLiNER2 is unsure (low/medium confidence) or unavailable. The LLM
re-reads the *raw user text* (the ground truth), inspects what GLiNER found, and
returns validated triplets with a self-assessed judge score and reasoning. Output
is schema-enforced by Instructor (Pydantic), then relations are normalized
through the soft ontology.

Degrades gracefully: returns ``[]`` when no API key is configured or on error.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from mnemo.app.core.config import settings
from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.extraction.judge import review_status_for_tier
from mnemo.app.services.ontology.manager import get_ontology
from mnemo.app.services.ontology.seed import SEED_RELATIONS

logger = logging.getLogger(__name__)

_FIRST_PERSON = {"i", "i'm", "im", "me", "my", "myself", "we", "us", "our", ""}

_client = None  # cached instructor-patched async client


class LLMFact(BaseModel):
    """One extracted fact (Instructor-enforced schema)."""

    subject: str = Field(min_length=1, description="Fact subject; use 'user' for the speaker")
    relation_raw: str = Field(min_length=1, description="Relation as phrased in text")
    object: str = Field(min_length=1, description="Fact object / value")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Judge score: 0.90-1.00 directly stated, 0.70-0.89 strongly implied, "
            "0.40-0.69 weakly implied, <0.40 unsupported"
        ),
    )
    reasoning: str = Field(default="", description="Why the raw text supports this fact")
    temporal_hint: str | None = Field(
        default=None, description="Temporal cue e.g. 'last month', 'since 2020'"
    )


class LLMFactList(BaseModel):
    """Container the LLM must return."""

    facts: list[LLMFact] = Field(default_factory=list, max_length=20)
    unknown_relations: list[str] = Field(default_factory=list)


def _system_prompt() -> str:
    relations = ", ".join(sorted(SEED_RELATIONS))
    return (
        "You are a fact extractor for a memory system used by a coding assistant.\n"
        "The RAW USER TEXT is the only ground truth. Never trust extractor output "
        "blindly and never invent facts.\n\n"
        "For each fact:\n"
        "- Use subject 'user' for the speaker (I/me/my/we), unless another subject "
        "is explicitly named.\n"
        "- Set relation_raw to the relation as implied by the text.\n"
        "- Provide reasoning grounded in the raw text.\n"
        "- Assign confidence as a JUDGE SCORE:\n"
        "    0.90-1.00 = directly stated\n"
        "    0.70-0.89 = strongly implied, unambiguous\n"
        "    0.40-0.69 = weakly implied or ambiguous\n"
        "    0.00-0.39 = not supported or contradicted\n"
        "- Add facts GLiNER missed; correct or drop facts GLiNER got wrong.\n"
        "- Prefer omitting a fact over emitting a weak/ambiguous one.\n"
        f"Prefer these known relations when they fit: {relations}.\n"
        "List any relation not in that set under unknown_relations."
    )


def _user_prompt(content: str, gliner_facts: list[TripletFact]) -> str:
    if gliner_facts:
        found = "\n".join(
            f"- {f.subject} [{f.relation_raw or f.relation}] {f.object} "
            f"(gliner_confidence={f.confidence:.2f})"
            for f in gliner_facts
        )
    else:
        found = "(none)"
    return (
        f"RAW USER TEXT (ground truth):\n\"\"\"\n{content.strip()}\n\"\"\"\n\n"
        f"GLiNER pre-extraction (verify against the text):\n{found}\n\n"
        "Return the validated, corrected, and completed fact list."
    )


def _get_client():
    global _client
    if _client is None:
        import instructor
        from openai import AsyncOpenAI

        api_key = settings.groq_api_key or settings.openai_api_key
        base = AsyncOpenAI(api_key=api_key, base_url=settings.groq_base_url)
        _client = instructor.from_openai(base, mode=instructor.Mode.JSON)
    return _client


def _normalize_subject(subject: str) -> str:
    value = subject.strip()
    return "user" if value.lower() in _FIRST_PERSON else value


async def extract(content: str, gliner_facts: list[TripletFact] | None = None) -> list[TripletFact]:
    """Re-extract facts from raw text via Groq; normalize relations. [] on error."""
    api_key = settings.groq_api_key or settings.openai_api_key
    if not content.strip() or not api_key:
        logger.warning("Groq API key not set — structured extraction skipped")
        return []

    gliner_facts = gliner_facts or []
    try:
        client = _get_client()
        result: LLMFactList = await client.chat.completions.create(
            model=settings.extraction_model,
            response_model=LLMFactList,
            max_retries=2,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": _user_prompt(content, gliner_facts)},
            ],
        )
    except Exception as exc:
        logger.exception("Groq structured extraction failed: %s", exc)
        return []

    ontology = get_ontology()
    facts: list[TripletFact] = []
    seen: set[tuple[str, str, str]] = set()
    for item in result.facts:
        match = ontology.normalize(item.relation_raw)
        if match.is_rejected:
            continue
        subject = _normalize_subject(item.subject)
        obj = " ".join(item.object.strip().strip(".,;:!?").split())[:512]
        if not obj:
            continue
        key = (subject.lower(), match.relation, obj.lower())
        if key in seen:
            continue
        seen.add(key)
        rel_phrase = match.relation.lower().replace("_", " ")
        facts.append(
            TripletFact(
                subject=subject,
                relation=match.relation,
                object=obj,
                fact_string=f"{subject} {rel_phrase} {obj}",
                confidence=round(float(item.confidence), 3),
                relation_raw=item.relation_raw,
                relation_match_score=match.match_score,
                review_status=review_status_for_tier(match.tier),
                reasoning=item.reasoning.strip(),
                temporal_hint=item.temporal_hint,
                source="llm",
            )
        )
    return facts

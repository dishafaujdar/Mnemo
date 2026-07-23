"""Step 1: temporal classification per clause (focused LLM call)."""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from mnemo.app.core.config import settings
from mnemo.app.services.extraction.llm_client import (
    get_instructor_client,
    llm_api_key,
    log_llm_failure,
)

logger = logging.getLogger(__name__)

TemporalLabel = Literal["current", "past", "future", "unspecified"]

_PAST_RE = re.compile(
    r"\b(?:used to|previously|formerly|was|were|had been|worked at|played)\b",
    re.IGNORECASE,
)
_FUTURE_RE = re.compile(
    r"\b(?:will|going to|gonna|next month|next year|soon|plan to|planning to)\b",
    re.IGNORECASE,
)
_PRESENT_RE = re.compile(
    r"\b(?:am|is|are|work at|works at|live with|learning|learn|play|use)\b",
    re.IGNORECASE,
)


class TemporalClassification(BaseModel):
    label: TemporalLabel = Field(description="CURRENT|PAST|FUTURE|UNSPECIFIED")


class ClassifiedClause(BaseModel):
    text: str
    temporal_status: TemporalLabel


def split_clauses(text: str) -> list[str]:
    """Split input into sentence/clause segments."""
    parts = re.split(
        r"(?<=[.!?;])\s+|\s+;\s+|\s*,\s*but\s+|\s+but\s+",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return [p.strip() for p in parts if p.strip()]


def _heuristic_label(sentence: str) -> TemporalLabel:
    lower = sentence.lower()
    if _FUTURE_RE.search(lower):
        return "future"
    if re.search(r"\bswitched to\b", lower):
        return "current"
    if re.search(r"\bused to\b", lower):
        return "past"
    if _PAST_RE.search(lower) and not _PRESENT_RE.search(lower):
        return "past"
    if re.search(r"\bsince\b", lower) and re.search(r"\b(?:learning|learn)\b", lower):
        return "current"
    return "unspecified"


async def classify_temporal_status(sentence: str) -> TemporalLabel:
    """Classify a single sentence/clause as current, past, future, or unspecified."""
    sentence = sentence.strip()
    if not sentence:
        return "unspecified"

    if not llm_api_key():
        return _heuristic_label(sentence)

    try:
        client = get_instructor_client()
        result: TemporalClassification = await client.chat.completions.create(
            model=settings.extraction_model,
            response_model=TemporalClassification,
            max_retries=1,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the temporal status of a statement as one of: "
                        "current, past, future, unspecified. "
                        "Return only the label in the structured field."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Classify the temporal status of this statement as one of:\n"
                        "CURRENT, PAST, FUTURE, UNSPECIFIED\n"
                        "Only return the label, nothing else.\n"
                        f"Statement: {sentence}"
                    ),
                },
            ],
        )
        label = result.label.lower()
        if label in {"current", "past", "future", "unspecified"}:
            logger.debug("Temporal LLM label for %r -> %s", sentence[:60], label)
            return label  # type: ignore[return-value]
    except Exception as exc:
        log_llm_failure("temporal_classification", exc)

    return _heuristic_label(sentence)


async def classify_clauses(text: str) -> list[ClassifiedClause]:
    """Classify every clause in the input text."""
    clauses = split_clauses(text)
    if not clauses:
        return [ClassifiedClause(text=text.strip(), temporal_status="unspecified")]
    out: list[ClassifiedClause] = []
    for clause in clauses:
        status = await classify_temporal_status(clause)
        out.append(ClassifiedClause(text=clause, temporal_status=status))
    logger.info(
        "Temporal classification: %d clauses -> %s",
        len(out),
        [(c.text[:40], c.temporal_status) for c in out],
    )
    return out


def temporal_status_for_span(source_span: str, classified: list[ClassifiedClause]) -> TemporalLabel:
    """Pick the temporal label of the clause that best contains the source span."""
    span = source_span.strip().lower()
    if not span:
        return "unspecified"
    best: TemporalLabel = "unspecified"
    best_len = 0
    for clause in classified:
        lower = clause.text.lower()
        if span in lower and len(clause.text) > best_len:
            best = clause.temporal_status
            best_len = len(clause.text)
    if best == "unspecified":
        for clause in classified:
            head = span.split()[0]
            if head and head in clause.text.lower():
                return clause.temporal_status
    return best

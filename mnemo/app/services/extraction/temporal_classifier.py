"""Step 1: temporal classification via a dedicated, deterministic LLM call.

Clause splitting and tense assignment are both delegated to the LLM. This module
does not extract facts and does not interpret meaning — it only classifies time.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from pydantic import BaseModel

from mnemo.app.services.extraction.llm_client import (
    get_raw_client,
    llm_api_key,
    log_llm_failure,
)

logger = logging.getLogger(__name__)

TemporalLabel = Literal["current", "past", "future", "unspecified"]

VALID_STATUSES = ("PAST", "CURRENT", "FUTURE", "UNSPECIFIED")

SYSTEM_PROMPT = (
    "You are a temporal classifier. Your only job is to identify clauses in a "
    "sentence and assign each a temporal status. You do not extract facts. "
    "You do not interpret meaning. You only classify time."
)

USER_PROMPT = """Analyze this sentence. Split it into clauses and classify each as:
- CURRENT: happening now, ongoing, present state
- PAST: used to happen, no longer true, historical
- FUTURE: planned, will happen, not yet true
- UNSPECIFIED: no temporal signal, assume current

Rules:
- If the sentence contains a transition ('used to X but now Y', \
'switched from X to Y', 'was X now Y') you MUST return exactly \
two clauses — one PAST, one CURRENT
- Never merge a PAST and CURRENT state into one clause
- Return ONLY valid JSON, no explanation, no markdown

Return format:
[
  {{"clause": "exact substring from input", "status": "PAST|CURRENT|FUTURE|UNSPECIFIED"}},
  ...
]

Sentence: {sentence}"""

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class ClassifiedClause(BaseModel):
    text: str
    temporal_status: TemporalLabel
    start: int = -1
    end: int = -1


def _fallback(sentence: str) -> list[dict]:
    return [{"clause": sentence, "status": "UNSPECIFIED"}]


def _parse_clauses(raw: str, sentence: str) -> list[dict]:
    """Parse and validate the model's JSON response; raise on anything malformed."""
    payload = _JSON_FENCE_RE.sub("", raw.strip())
    clauses = json.loads(payload)

    if not isinstance(clauses, list) or not clauses:
        raise ValueError("expected a non-empty JSON list")
    for clause in clauses:
        if not isinstance(clause, dict):
            raise ValueError("clause entry is not an object")
        if "clause" not in clause or "status" not in clause:
            raise ValueError("clause entry missing required fields")
        if clause["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid status: {clause['status']!r}")
    return clauses


async def classify_temporal_status(sentence: str) -> list[dict]:
    """Split a sentence into clauses and label each PAST/CURRENT/FUTURE/UNSPECIFIED."""
    sentence = sentence.strip()
    if not sentence:
        return []

    if not llm_api_key():
        logger.warning("Temporal classifier has no LLM key; defaulting to UNSPECIFIED")
        return _fallback(sentence)

    try:
        client = get_raw_client()
        response = await client.chat.completions.create(
            model=_model(),
            temperature=0,  # deterministic — no variation
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(sentence=sentence)},
            ],
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        log_llm_failure("temporal_classification", exc)
        return _fallback(sentence)

    try:
        return _parse_clauses(raw, sentence)
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Temporal classifier failed to parse: %s", sentence)
        return _fallback(sentence)


def _model() -> str:
    from mnemo.app.core.config import settings

    return settings.extraction_model


def _locate(text: str, clause: str, cursor: int) -> tuple[int, int]:
    """Find a clause's character offsets, searching forward from ``cursor``."""
    idx = text.find(clause, cursor)
    if idx < 0:
        idx = text.lower().find(clause.lower(), cursor)
    if idx < 0:
        idx = text.lower().find(clause.lower())
    if idx < 0:
        return -1, -1
    return idx, idx + len(clause)


async def classify_clauses(text: str) -> list[ClassifiedClause]:
    """Classify every clause across every sentence in the input text."""
    stripped = text.strip()
    if not stripped:
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(stripped) if s.strip()]
    out: list[ClassifiedClause] = []
    cursor = 0

    for sentence in sentences:
        for item in await classify_temporal_status(sentence):
            clause_text = str(item["clause"]).strip()
            if not clause_text:
                continue
            start, end = _locate(stripped, clause_text, cursor)
            if start >= 0:
                cursor = start
            out.append(
                ClassifiedClause(
                    text=clause_text,
                    temporal_status=item["status"].lower(),
                    start=start,
                    end=end,
                )
            )

    if not out:
        return [ClassifiedClause(text=stripped, temporal_status="unspecified")]

    logger.info(
        "Temporal classification: %d clauses -> %s",
        len(out),
        [(c.text[:40], c.temporal_status) for c in out],
    )
    return out


def temporal_status_for_span(
    source_span: str,
    classified: list[ClassifiedClause],
    original_text: str = "",
) -> TemporalLabel:
    """Pick the temporal label of the clause the source span overlaps most.

    Matching is positional when offsets are available: a span is attributed to the
    clause it physically overlaps, never to a clause that merely shares a word.
    """
    span = source_span.strip()
    if not span:
        return "unspecified"

    if original_text:
        start = original_text.lower().find(span.lower())
        if start >= 0:
            end = start + len(span)
            best: TemporalLabel | None = None
            best_overlap = 0
            for clause in classified:
                if clause.start < 0 or clause.end < 0:
                    continue
                overlap = min(end, clause.end) - max(start, clause.start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = clause.temporal_status
            if best is not None:
                return best

    # Offsets unavailable: fall back to containment only (shortest containing clause).
    lower_span = span.lower()
    best_label: TemporalLabel = "unspecified"
    best_len: int | None = None
    for clause in classified:
        lower = clause.text.lower()
        if lower_span in lower and (best_len is None or len(lower) < best_len):
            best_label = clause.temporal_status
            best_len = len(lower)
    return best_label

"""GLiNER2 fast local extraction (Step 1).

Runs a local GLiNER2 model (or the hosted API) to pull relation triplets with
confidence scores. Relations are normalized through the soft ontology. This
module degrades gracefully: if ``gliner2`` (or its ``[local]`` extra) isn't
installed or the model can't load, ``extract`` returns ``[]`` and the pipeline
falls back to the LLM stage.
"""

from __future__ import annotations

import logging

from mnemo.app.core.config import settings
from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.extraction.judge import review_status_for_tier
from mnemo.app.services.ontology.canonical import normalize_object
from mnemo.app.services.ontology.manager import get_ontology
from mnemo.app.services.ontology.seed import GLINER_RELATION_LABELS

logger = logging.getLogger(__name__)

_FIRST_PERSON = {"i", "i'm", "im", "me", "my", "myself", "we", "us", "our", "user"}

_model = None  # cached GLiNER2 instance
_load_failed = False  # remember failure so we don't retry every call


def _get_model():
    """Lazily load the GLiNER2 model; cache the result (or the failure)."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    try:
        from gliner2 import GLiNER2

        if settings.gliner_api_enabled:
            _model = GLiNER2.from_api()
        else:
            _model = GLiNER2.from_pretrained(settings.gliner_model)
    except Exception as exc:  # ImportError, model download/load errors, etc.
        logger.warning("GLiNER2 unavailable, skipping local extraction: %s", exc)
        _load_failed = True
        _model = None
    return _model


def is_available() -> bool:
    return _get_model() is not None


def _normalize_subject(head: str) -> str:
    value = head.strip()
    if value.lower() in _FIRST_PERSON:
        return "user"
    return value or "user"


def _pair_confidence(item: dict) -> float:
    """Confidence for a relation pair = min(head, tail) when available."""
    scores = [
        part.get("confidence")
        for part in (item.get("head", {}), item.get("tail", {}))
        if isinstance(part, dict) and part.get("confidence") is not None
    ]
    return float(min(scores)) if scores else 0.75


def _iter_pairs(rel_map: dict) -> list[tuple[str, str, str, float]]:
    """Yield (relation_label, head, tail, confidence) from GLiNER output.

    Handles both the plain tuple form ``('head', 'tail')`` and the enriched
    dict form ``{'head': {'text', 'confidence'}, 'tail': {...}}``.
    """
    pairs: list[tuple[str, str, str, float]] = []
    for label, items in (rel_map or {}).items():
        for item in items or []:
            if isinstance(item, dict):
                head = str(item.get("head", {}).get("text", "")).strip()
                tail = str(item.get("tail", {}).get("text", "")).strip()
                conf = _pair_confidence(item)
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                head, tail, conf = str(item[0]).strip(), str(item[1]).strip(), 0.75
            else:
                continue
            if head and tail:
                pairs.append((label, head, tail, conf))
    return pairs


def extract(content: str) -> list[TripletFact]:
    """Extract relation triplets locally with GLiNER2. Returns [] if unavailable."""
    if not settings.gliner_enabled or not content or not content.strip():
        return []
    model = _get_model()
    if model is None:
        return []

    try:
        result = model.extract_relations(
            content,
            GLINER_RELATION_LABELS,
            include_confidence=True,
        )
    except Exception as exc:
        logger.warning("GLiNER2 relation extraction failed: %s", exc)
        return []

    rel_map = result.get("relation_extraction", result) if isinstance(result, dict) else {}
    ontology = get_ontology()
    facts: list[TripletFact] = []
    seen: set[tuple[str, str, str]] = set()

    for label, head, tail, conf in _iter_pairs(rel_map):
        match = ontology.normalize(label)
        if match.is_rejected:
            continue
        subject = _normalize_subject(head)
        obj = normalize_object(tail)
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
                confidence=round(conf, 3),
                relation_raw=label,
                relation_match_score=match.match_score,
                review_status=review_status_for_tier(match.tier),
                source="gliner",
            )
        )
    return facts

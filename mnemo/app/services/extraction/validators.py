"""Ground extracted facts against the source episode text."""

from __future__ import annotations

import re

from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.conflict.groups import (
    EMPLOYMENT_OBJECT_RELATIONS,
    EMPLOYMENT_PAST_RELATIONS,
    TRANSITION_RELATIONS,
    slot_for_relation,
)

_BORN_PATTERN = re.compile(r"\b(?:born|birth)\b", re.IGNORECASE)
_RESIDENCE_PATTERN = re.compile(
    r"\b(?:living|live|lives|lived|reside|resides|residing|based)\s+in\b",
    re.IGNORECASE,
)

# High-risk relations require explicit lexical evidence in the source text.
HIGH_RISK_RELATIONS: frozenset[str] = frozenset(
    {
        "BORN_IN",
        "LIVED_IN",
        "WORKED_AT",
        "VISITED",
        "MET",
    }
)

TECHNOLOGY_OBJECTS: frozenset[str] = frozenset(
    {
        "python",
        "go",
        "golang",
        "rust",
        "typescript",
        "javascript",
        "java",
        "c++",
        "ruby",
        "fastapi",
        "django",
        "react",
        "vue",
        "docker",
        "kubernetes",
        "cursor",
        "vscode",
        "vim",
        "sqlite",
        "postgres",
        "redis",
        "qdrant",
        "langchain",
    }
)

_COMPANY_LIKE = re.compile(
    r"\b(?:inc|llc|labs|corp|corporation|company|google|deepmind|stripe|slice|microsoft|amazon|meta|apple)\b",
    re.IGNORECASE,
)


def is_technology_object(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TECHNOLOGY_OBJECTS:
        return True
    if _COMPANY_LIKE.search(value):
        return False
    # Short tokens that look like languages/tools (not multi-word companies).
    tokens = normalized.split()
    return len(tokens) == 1 and len(normalized) <= 20 and normalized.isascii()


def _text_supports_relation(text: str, relation: str, obj: str) -> bool:
    lower = text.lower()
    obj_lower = obj.lower()

    if relation == "BORN_IN":
        return bool(_BORN_PATTERN.search(text)) and obj_lower.split()[0] in lower

    if relation == "LIVED_IN":
        return bool(re.search(rf"\b(?:lived|living)\s+in\b.{0,40}\b{re.escape(obj_lower.split()[0])}\b", lower))

    if relation == "WORKED_AT":
        # Past employment must use explicit past tense near the company name.
        return bool(
            re.search(
                rf"\b(?:worked|used to work)\s+(?:at|for)\b.{0,40}\b{re.escape(obj_lower.split()[0])}\b",
                lower,
            )
        )

    return True


def _object_position(text: str, obj: str) -> int:
    """Last occurrence of the object's head token; later in text = more current."""
    lower = text.lower()
    head = obj.strip().lower().split()[0] if obj.strip() else ""
    if not head:
        return -1
    return lower.rfind(head)


def validate_fact_against_source(content: str, fact: TripletFact) -> bool:
    """Reject facts not supported by the raw user text."""
    text = content.strip()
    if not text:
        return False
    lower = text.lower()
    obj = fact.object.strip()
    obj_lower = obj.lower()
    if not obj_lower:
        return False

    relation = fact.relation.upper()

    # BORN_IN is never inferred — require explicit birth language in the source.
    if relation == "BORN_IN" and not _BORN_PATTERN.search(text):
        return False

    # Never store transition/event relations.
    if relation in TRANSITION_RELATIONS:
        return False

    # SWITCHED_TO is only valid for technology stack, never companies (handled if it slips through).
    if relation == "SWITCHED_TO" and not is_technology_object(obj):
        return False

    # Object tokens must appear in source.
    obj_tokens = [t for t in re.findall(r"[a-z0-9]+", obj_lower) if len(t) > 2]
    if obj_tokens and not any(token in lower for token in obj_tokens):
        return False

    # High-risk predicates need explicit supporting language.
    if relation in HIGH_RISK_RELATIONS and not _text_supports_relation(text, relation, obj):
        return False

    if relation == "BORN_IN":
        if not _BORN_PATTERN.search(text):
            return False
        if _RESIDENCE_PATTERN.search(text) and not _BORN_PATTERN.search(text):
            return False

    return True


def _pick_slot_winner(candidates: list[TripletFact], source_text: str) -> TripletFact:
    """Keep the most current fact for a singular slot (later in source text wins)."""
    if len(candidates) == 1:
        return candidates[0]
    if source_text.strip():
        return max(candidates, key=lambda f: _object_position(source_text, f.object))
    # Prefer HAS_ROLE over IS; otherwise keep the last candidate in extraction order.
    role_rank = {"HAS_ROLE": 2, "IS": 1}

    def _rank(fact: TripletFact) -> tuple[int, float, int]:
        return (
            role_rank.get(fact.relation.upper(), 0),
            fact.confidence,
            candidates.index(fact),
        )

    return max(candidates, key=_rank)


def dedupe_batch_facts(
    facts: list[TripletFact],
    source_text: str = "",
) -> list[TripletFact]:
    """Within one episode: one winner per singular slot; collapse employment object dupes."""
    has_current_employer = any(f.relation.upper() == "WORKS_AT" for f in facts)
    if has_current_employer:
        # End-state only: drop past/transition employment from the same utterance.
        facts = [
            f
            for f in facts
            if f.relation.upper() not in EMPLOYMENT_PAST_RELATIONS
            and f.relation.upper() not in TRANSITION_RELATIONS
        ]

    by_slot: dict[str, list[TripletFact]] = {}
    employment_by_object: dict[str, TripletFact] = {}
    rest: list[TripletFact] = []

    for fact in facts:
        rel = fact.relation.upper()
        if rel in TRANSITION_RELATIONS:
            continue

        slot = slot_for_relation(fact.relation)
        if slot:
            by_slot.setdefault(slot, []).append(fact)
            continue

        if rel in EMPLOYMENT_OBJECT_RELATIONS or rel in EMPLOYMENT_PAST_RELATIONS:
            key = fact.object.strip().lower()
            # Prefer current employer over past tense for the same company.
            existing = employment_by_object.get(key)
            if existing is None or rel == "WORKS_AT":
                employment_by_object[key] = fact
            continue

        rest.append(fact)

    slot_facts = [_pick_slot_winner(cands, source_text) for cands in by_slot.values()]
    out = slot_facts + list(employment_by_object.values()) + rest
    seen: set[tuple[str, str, str]] = set()
    unique: list[TripletFact] = []
    for fact in out:
        key = (fact.subject.lower(), fact.relation.upper(), fact.object.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(fact)
    return unique

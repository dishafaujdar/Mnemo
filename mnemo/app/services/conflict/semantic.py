"""Semantic conflict detection: slots, object overlap, embedding similarity."""

from __future__ import annotations

import re

from mnemo.app.core.config import settings
from mnemo.app.db.models import SemanticEdge
from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.conflict.groups import (
    EMPLOYMENT_CURRENT_RELATIONS,
    EMPLOYMENT_OBJECT_RELATIONS,
    EMPLOYMENT_PAST_RELATIONS,
    LOCATION_OBJECT_RELATIONS,
    TRANSITION_RELATIONS,
    slot_for_relation,
    supersedes,
)
from mnemo.app.services.embeddings import get_embedding

_NON_WORD = re.compile(r"[^a-z0-9]+")


def _norm_object(value: str) -> str:
    return _NON_WORD.sub(" ", value.strip().lower()).strip()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def same_object(fact: TripletFact, edge: SemanticEdge) -> bool:
    return _norm_object(fact.object) == _norm_object(edge.object)


def is_exact_duplicate(fact: TripletFact, edge: SemanticEdge) -> bool:
    return (
        fact.relation.upper() == edge.relation.upper()
        and _norm_object(fact.object) == _norm_object(edge.object)
    )


def is_cross_relation_duplicate(fact: TripletFact, edge: SemanticEdge) -> bool:
    """Same singular slot + same object (IS vs HAS_ROLE 'backend engineer')."""
    fact_slot = slot_for_relation(fact.relation)
    edge_slot = slot_for_relation(edge.relation)
    if not fact_slot or fact_slot != edge_slot:
        return False
    return _norm_object(fact.object) == _norm_object(edge.object)


def is_object_redundant(fact: TripletFact, edge: SemanticEdge) -> bool:
    """Same entity, different surface relation (e.g. WORKED_AT vs SWITCHED_FROM Slice)."""
    if not same_object(fact, edge):
        return False
    fr = fact.relation.upper()
    er = edge.relation.upper()
    if fr in EMPLOYMENT_OBJECT_RELATIONS and er in EMPLOYMENT_OBJECT_RELATIONS:
        return fr != er
    if fr in LOCATION_OBJECT_RELATIONS and er in LOCATION_OBJECT_RELATIONS:
        return fr != er
    return False


def should_invalidate(
    fact: TripletFact,
    edge: SemanticEdge,
    *,
    similarity: float = 0.0,
) -> bool:
    """Return True if storing ``fact`` should retract ``edge``."""
    if is_exact_duplicate(fact, edge):
        return False

    fr = fact.relation.upper()
    er = edge.relation.upper()

    # Cross-relation singular slot (IS vs HAS_ROLE, etc.).
    fact_slot = slot_for_relation(fr)
    edge_slot = slot_for_relation(er)
    if fact_slot and fact_slot == edge_slot and _norm_object(fact.object) != _norm_object(edge.object):
        return True

    # New current employer invalidates past employment + any transition records.
    if fr in EMPLOYMENT_CURRENT_RELATIONS:
        if er in EMPLOYMENT_PAST_RELATIONS or er in TRANSITION_RELATIONS:
            return True
        if er == "WORKS_AT" and _norm_object(fact.object) != _norm_object(edge.object):
            return True

    # Explicit supersession (LIVES_IN over BORN_IN at same place).
    if supersedes(fr, er) and same_object(fact, edge):
        return True

    # Employment / location object dedup across relation phrasing.
    if is_object_redundant(fact, edge):
        return True

    # High semantic similarity on fact strings (paraphrase dedup).
    threshold = settings.semantic_conflict_threshold
    if similarity >= threshold and fr == er:
        return _norm_object(fact.object) != _norm_object(edge.object)

    return False


async def embedding_similarity(fact: TripletFact, edge: SemanticEdge) -> float:
    vec_a = await get_embedding(fact.fact_string)
    vec_b = await get_embedding(edge.fact_string)
    return cosine_similarity(vec_a, vec_b)


async def find_conflicting_edges(
    fact: TripletFact,
    active_edges: list[SemanticEdge],
) -> list[SemanticEdge]:
    """Return active edges that should be invalidated before storing ``fact``."""
    conflicts: list[SemanticEdge] = []
    seen_ids: set[str] = set()
    for edge in active_edges:
        if should_invalidate(fact, edge, similarity=0.0):
            if edge.id not in seen_ids:
                seen_ids.add(edge.id)
                conflicts.append(edge)
            continue
        sim = await embedding_similarity(fact, edge)
        if should_invalidate(fact, edge, similarity=sim):
            if edge.id not in seen_ids:
                seen_ids.add(edge.id)
                conflicts.append(edge)
    return conflicts

"""Bi-temporal conflict resolution: detect duplicates/contradictions, invalidate, insert."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client.http.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from mnemo.app.db.models import SemanticEdge, UnknownRelation
from mnemo.app.db.qdrant import ensure_collection, get_qdrant_client, point_exists, set_point_payload, upsert_points
from mnemo.app.models.extraction import REVIEW_PENDING, TripletFact
from mnemo.app.services.conflict.temporal import stamp_retracted_at, utc_now
from mnemo.app.services.embeddings import get_embedding
from mnemo.app.services.memory.profile import set_fact
from mnemo.app.services.conflict.semantic import (
    find_conflicting_edges,
    is_cross_relation_duplicate,
    is_exact_duplicate,
)
from mnemo.app.services.conflict.groups import (
    EMPLOYMENT_PAST_RELATIONS,
    TRANSITION_RELATIONS,
    slot_for_relation,
)
from mnemo.app.services.ontology.manager import get_ontology
from mnemo.app.services.extraction.validators import is_technology_object

# Kept for backwards compatibility; canonical behavior now lives in the ontology.
SINGULAR_RELATIONS = {"IS", "WORKS_AT", "SWITCHED_TO", "GOAL_IS"}


PROFILE_RELATIONS = {
    "IS": "role",
    "HAS_ROLE": "role",
    "WORKS_AT": "company",
    "WORKS_ON": "current_project",
    "GOAL_IS": "goal",
    "LIVES_IN": "location",
}


def _gen_id() -> str:
    return str(uuid4())


def _profile_value_for_fact(fact: TripletFact) -> str | None:
    return _profile_value_for_object(fact.relation, fact.object)


def _profile_value_for_object(relation: str, obj: str) -> str | None:
    value = obj.strip()
    if not value:
        return None
    if relation in {"IS", "HAS_ROLE"}:
        lowered = value.lower()
        if lowered.startswith(("a ", "an ", "the ")):
            value = value.split(" ", 1)[1].strip()
        if len(value.split()) > 6:
            return None
    if relation == "WORKS_AT" and len(value.split()) > 4:
        # Reject noisy extractions like full sentence as company name.
        return None
    if relation == "WORKS_ON" and len(value.split()) > 12:
        return None
    if relation == "GOAL_IS" and len(value) > 200:
        return None
    return value


async def get_all_active_edges(
    db: AsyncSession,
    user_id: str,
) -> list[SemanticEdge]:
    """All active semantic edges for a user."""
    q = select(SemanticEdge).where(
        and_(
            SemanticEdge.user_id == user_id,
            SemanticEdge.invalid_at.is_(None),
        )
    )
    result = await db.execute(q)
    return list(result.scalars().all())


async def get_active_edges(
    db: AsyncSession,
    user_id: str,
    subject: str,
    relation: str,
) -> list[SemanticEdge]:
    """Return semantic edges with same subject+relation and invalid_at IS NULL."""
    q = select(SemanticEdge).where(
        and_(
            SemanticEdge.user_id == user_id,
            SemanticEdge.subject == subject,
            SemanticEdge.relation == relation,
            SemanticEdge.invalid_at.is_(None),
        )
    )
    result = await db.execute(q)
    return list(result.scalars().all())


def is_duplicate(fact: TripletFact, existing: list[SemanticEdge]) -> bool:
    """True if an active edge already has the same object."""
    obj = fact.object.strip().lower()
    for e in existing:
        if getattr(e, "object", "").strip().lower() == obj:
            return True
    return False


def is_contradiction(fact: TripletFact, existing: list[SemanticEdge]) -> bool:
    """True for singular-value relations when existing has a different object.

    Behavior is driven by the soft ontology: only SINGULAR relations can
    contradict; MULTI and TEMPORAL relations always coexist.
    """
    if not get_ontology().is_singular(fact.relation):
        return False
    obj = fact.object.strip().lower()
    for e in existing:
        if getattr(e, "object", "").strip().lower() != obj:
            return True
    return False


async def close_edges(
    db: AsyncSession,
    closures: list[tuple[SemanticEdge, datetime]],
    retracted_at: datetime,
    qdrant_client=None,
) -> None:
    """Close old facts by setting valid_until and retracted_at; rows are never deleted."""
    for edge, valid_until in closures:
        edge.invalid_at = valid_until
        stamp_retracted_at(edge, retracted_at)
    await db.flush()
    if qdrant_client is not None:
        for edge, valid_until in closures:
            point_id = edge.qdrant_id or edge.id
            if not point_id:
                continue
            try:
                await set_point_payload(
                    qdrant_client,
                    point_id,
                    {
                        "invalid_at": valid_until.isoformat(),
                        "retracted_at": retracted_at.isoformat(),
                    },
                )
            except UnexpectedResponse as exc:
                if exc.status_code == 404:
                    print(f"[WARN] missing qdrant point during close edge_id={edge.id} point_id={point_id}")
                    continue
                raise


async def invalidate_edges(
    db: AsyncSession,
    edges: list[SemanticEdge],
    invalidated_at: datetime,
    qdrant_client=None,
) -> None:
    """Backward-compatible wrapper: valid_until and retracted_at both set to ``invalidated_at``."""
    closures = [(edge, invalidated_at) for edge in edges]
    await close_edges(db, closures, invalidated_at, qdrant_client)


async def insert_edge(
    db: AsyncSession,
    fact: TripletFact,
    episode_id: str,
    user_id: str,
    qdrant_client,
    valid_at: datetime | None = None,
) -> str:
    """Insert new semantic edge and upsert vector; return edge id."""
    now = valid_at or utc_now()
    edge_id = _gen_id()
    vector = await get_embedding(fact.fact_string)
    payload = {
        "user_id": user_id,
        "edge_id": edge_id,
        "episode_id": episode_id,
        "invalid_at": None,
        "relation": fact.relation,
        "valid_at": now.isoformat(),
        "fact_string": fact.fact_string,
        "confidence": fact.confidence,
        "retracted_at": None,
    }
    point = PointStruct(id=edge_id, vector=vector, payload=payload)
    await upsert_points(qdrant_client, [point])
    row = SemanticEdge(
        id=edge_id,
        user_id=user_id,
        subject=fact.subject.lower(),
        relation=fact.relation,
        object=fact.object,
        fact_string=fact.fact_string,
        relation_raw=fact.relation_raw or None,
        relation_match_score=fact.relation_match_score,
        review_status=fact.review_status,
        qdrant_id=edge_id,
        episode_id=episode_id,
        confidence=fact.confidence,
        valid_at=now,
        invalid_at=None,
        created_at=now,
    )
    db.add(row)
    await db.flush()
    if fact.review_status == REVIEW_PENDING:
        await _record_unknown_relation(db, fact)
    await _maybe_update_profile(db, user_id, fact)
    return edge_id


async def _record_unknown_relation(db: AsyncSession, fact: TripletFact) -> None:
    """Upsert an unknown relation into the audit queue (count + running avg)."""
    result = await db.execute(
        select(UnknownRelation).where(UnknownRelation.relation == fact.relation)
    )
    row = result.scalars().first()
    now = datetime.utcnow()
    if row is None:
        db.add(
            UnknownRelation(
                relation=fact.relation,
                relation_raw=fact.relation_raw or None,
                count=1,
                avg_confidence=fact.confidence,
                status="pending",
                created_at=now,
                updated_at=now,
            )
        )
    else:
        total = row.avg_confidence * row.count + fact.confidence
        row.count += 1
        row.avg_confidence = total / row.count
        row.updated_at = now
    await db.flush()


async def rebuild_missing_qdrant_points(
    db: AsyncSession,
    qdrant_client=None,
    user_id: str | None = None,
) -> int:
    """
    Rebuild active semantic edges that exist in SQLite but are missing in Qdrant.
    SQLite remains the source of truth; only active edges with enough data are rehydrated.
    """
    if qdrant_client is None:
        qdrant_client = get_qdrant_client()
    await ensure_collection(qdrant_client)

    q = select(SemanticEdge).where(SemanticEdge.invalid_at.is_(None))
    if user_id is not None:
        q = q.where(SemanticEdge.user_id == user_id)
    result = await db.execute(q)
    edges = list(result.scalars().all())

    rebuilt = 0
    for edge in edges:
        point_id = edge.qdrant_id or edge.id
        if not point_id or not edge.fact_string.strip():
            continue
        try:
            if await point_exists(qdrant_client, point_id):
                continue
        except UnexpectedResponse:
            continue

        vector = await get_embedding(edge.fact_string)
        payload = {
            "user_id": edge.user_id,
            "edge_id": edge.id,
            "episode_id": edge.episode_id,
            "invalid_at": None,
            "relation": edge.relation,
            "valid_at": edge.valid_at.isoformat(),
            "fact_string": edge.fact_string,
            "confidence": edge.confidence,
        }
        point = PointStruct(id=point_id, vector=vector, payload=payload)
        await upsert_points(qdrant_client, [point])
        if edge.qdrant_id != point_id:
            edge.qdrant_id = point_id
        rebuilt += 1

    if rebuilt:
        await db.flush()
    return rebuilt


async def invalidate_memory_by_id(
    db: AsyncSession,
    edge_id: str,
    qdrant_client=None,
) -> bool:
    """Manually retract a semantic edge by id. Returns True if found and closed."""
    from sqlalchemy import select
    result = await db.execute(select(SemanticEdge).where(SemanticEdge.id == edge_id))
    edge = result.scalars().first()
    if edge is None:
        return False
    now = utc_now()
    await close_edges(db, [(edge, now)], now, qdrant_client)
    return True


def _norm_object(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()


async def reconcile_active_edges(
    db: AsyncSession,
    user_id: str,
    qdrant_client=None,
) -> int:
    """Invalidate contradictory active edges; keep the newest winner per slot.

    Runs after every ingest so duplicate-skipped re-ingests still repair legacy state.
    """
    if qdrant_client is None:
        qdrant_client = get_qdrant_client()
    retracted_at = utc_now()
    active = await get_all_active_edges(db, user_id)
    if not active:
        return 0

    retire_ids: set[str] = set()
    closures: list[tuple[SemanticEdge, datetime]] = []

    def _retire(edge: SemanticEdge, valid_until: datetime) -> None:
        if edge.id not in retire_ids:
            retire_ids.add(edge.id)
            closures.append((edge, valid_until))

    durable = [
        e
        for e in active
        if e.relation.upper() not in TRANSITION_RELATIONS
    ]
    fallback_valid_until = (
        max(durable, key=lambda e: e.valid_at).valid_at if durable else retracted_at
    )

    # Transition relations are never durable state.
    for edge in active:
        if edge.relation.upper() in TRANSITION_RELATIONS:
            _retire(edge, fallback_valid_until)

    living = [
        e
        for e in active
        if e.relation.upper() == "LIVES_IN" and e.id not in retire_ids
    ]
    for edge in active:
        if edge.id in retire_ids:
            continue
        if edge.relation.upper() == "BORN_IN":
            for live in living:
                if _norm_object(live.object) == _norm_object(edge.object):
                    _retire(edge, live.valid_at)
                    break

    # One winner per singular slot (role, employer, residence, origin, goal).
    by_slot: dict[str, list[SemanticEdge]] = {}
    for edge in active:
        if edge.id in retire_ids:
            continue
        slot = slot_for_relation(edge.relation)
        if slot:
            by_slot.setdefault(slot, []).append(edge)

    for edges in by_slot.values():
        if len(edges) <= 1:
            continue
        winner = max(edges, key=lambda e: e.valid_at)
        for edge in edges:
            if edge.id != winner.id:
                _retire(edge, winner.valid_at)

    # Current employer is end-state: retract past employment and extra WORKS_AT rows.
    works_at = [
        e
        for e in active
        if e.relation.upper() == "WORKS_AT" and e.id not in retire_ids
    ]
    if works_at:
        current = max(works_at, key=lambda e: e.valid_at)
        for edge in works_at:
            if edge.id != current.id:
                _retire(edge, current.valid_at)
        for edge in active:
            if edge.id in retire_ids:
                continue
            if edge.relation.upper() in EMPLOYMENT_PAST_RELATIONS:
                _retire(edge, current.valid_at)

    if closures:
        await close_edges(db, closures, retracted_at, qdrant_client)
    return len(closures)


async def resolve_and_store(
    new_facts: list[TripletFact],
    user_id: str,
    episode_id: str,
    db: AsyncSession,
    qdrant_client=None,
) -> None:
    """
    Bi-temporal resolution with semantic conflict detection:
    - exact / cross-relation duplicate → skip
    - slot conflicts, object-redundant phrasing, supersession, embedding similarity → close
    - then insert new fact
    """
    if qdrant_client is None:
        qdrant_client = get_qdrant_client()
    active_edges = await get_all_active_edges(db, user_id)

    # Store durable state facts before lower-priority relations so invalidation runs first.
    priority = {"WORKS_AT": 0, "IS": 0, "HAS_ROLE": 0, "LIVES_IN": 0, "USES": 5, "PREFERS": 5}
    ordered_facts = sorted(new_facts, key=lambda f: priority.get(f.relation.upper(), 3))

    for fact in ordered_facts:
        rel = fact.relation.upper()
        if rel in EMPLOYMENT_PAST_RELATIONS and any(
            e.relation.upper() == "WORKS_AT" for e in active_edges
        ):
            continue

        if any(is_exact_duplicate(fact, edge) or is_cross_relation_duplicate(fact, edge) for edge in active_edges):
            continue

        fact_valid_at = utc_now()
        retracted_at = utc_now()
        conflicts = await find_conflicting_edges(fact, active_edges)
        if conflicts:
            closures = [(edge, fact_valid_at) for edge in conflicts]
            await close_edges(db, closures, retracted_at, qdrant_client)
            conflict_ids = {e.id for e in conflicts}
            active_edges = [e for e in active_edges if e.id not in conflict_ids]

        edge_id = await insert_edge(
            db, fact, episode_id, user_id, qdrant_client, valid_at=fact_valid_at
        )
        # Track newly inserted edge so later facts in the batch see it.
        active_edges.append(
            SemanticEdge(
                id=edge_id,
                user_id=user_id,
                subject=fact.subject.lower(),
                relation=fact.relation,
                object=fact.object,
                fact_string=fact.fact_string,
                confidence=fact.confidence,
                valid_at=fact_valid_at,
                invalid_at=None,
                episode_id=episode_id,
                created_at=fact_valid_at,
            )
        )

    await reconcile_active_edges(db, user_id, qdrant_client)


async def sync_profile_from_active_edges(db: AsyncSession, user_id: str) -> None:
    """Rebuild profile keys from current active semantic edges (source of truth)."""
    ontology = get_ontology()
    for relation, key in PROFILE_RELATIONS.items():
        edges = await get_active_edges(db, user_id, "user", relation)
        if not edges:
            continue
        if ontology.is_singular(relation):
            edge = max(edges, key=lambda e: e.valid_at)
            value = _profile_value_for_object(edge.relation, edge.object)
            if value:
                await set_fact(db, user_id, key, value)
        else:
            values: list[str] = []
            seen: set[str] = set()
            for edge in sorted(edges, key=lambda e: e.valid_at, reverse=True):
                value = _profile_value_for_object(edge.relation, edge.object)
                if value and value.lower() not in seen:
                    seen.add(value.lower())
                    values.append(value)
            if values:
                await set_fact(db, user_id, key, values, value_type="list")

    # current_stack = languages/tools from USES (never companies).
    uses_edges = await get_active_edges(db, user_id, "user", "USES")
    stack: list[str] = []
    seen_stack: set[str] = set()
    for edge in sorted(uses_edges, key=lambda e: e.valid_at, reverse=True):
        if not is_technology_object(edge.object):
            continue
        token = edge.object.strip()
        if token.lower() not in seen_stack:
            seen_stack.add(token.lower())
            stack.append(token)
    if stack:
        await set_fact(db, user_id, "current_stack", stack, value_type="list")


async def _maybe_update_profile(db, user_id: str, fact: TripletFact) -> None:
    """Per-edge profile hint; ``sync_profile_from_active_edges`` is authoritative."""
    key = PROFILE_RELATIONS.get(fact.relation)
    if not key:
        return
    value = _profile_value_for_fact(fact)
    if value is None:
        return
    await set_fact(db, user_id, key, value)
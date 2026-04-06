"""Bi-temporal conflict resolution: detect duplicates/contradictions, invalidate, insert."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client.http.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from mnemo.app.db.models import SemanticEdge
from mnemo.app.db.qdrant import ensure_collection, get_qdrant_client, point_exists, set_point_payload, upsert_points
from mnemo.app.models.extraction import TripletFact
from mnemo.app.services.embeddings import get_embedding
from mnemo.app.services.memory.profile import set_fact

SINGULAR_RELATIONS = {"IS", "WORKS_AT", "SWITCHED_TO", "GOAL_IS"}


PROFILE_RELATIONS = {
    "IS": "role",
    "WORKS_AT": "company",
    "WORKS_ON": "current_project",
    "GOAL_IS": "goal",
    "SWITCHED_TO": "current_stack",
}


def _gen_id() -> str:
    return str(uuid4())


def _profile_value_for_fact(fact: TripletFact) -> str | None:
    value = fact.object.strip()
    if not value:
        return None
    if fact.relation == "IS":
        lowered = value.lower()
        if lowered.startswith(("a ", "an ", "the ")):
            value = value.split(" ", 1)[1].strip()
        if len(value.split()) > 6:
            return None
    if fact.relation == "WORKS_ON" and len(value.split()) > 12:
        return None
    if fact.relation == "GOAL_IS" and len(value) > 200:
        return None
    return value


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
    """True for singular-value relations when existing has different object."""
    if fact.relation not in SINGULAR_RELATIONS:
        return False
    obj = fact.object.strip().lower()
    for e in existing:
        if getattr(e, "object", "").strip().lower() != obj:
            return True
    return False


async def invalidate_edges(
    db: AsyncSession,
    edges: list[SemanticEdge],
    invalidated_at: datetime,
    qdrant_client=None,
) -> None:
    """Set invalid_at on given edges (soft invalidation); update Qdrant payload so valid_only filter excludes them."""
    for e in edges:
        e.invalid_at = invalidated_at
    await db.flush()
    if qdrant_client is not None:
        at_str = invalidated_at.isoformat()
        for e in edges:
            point_id = e.qdrant_id or e.id
            if not point_id:
                continue
            try:
                await set_point_payload(qdrant_client, point_id, {"invalid_at": at_str})
            except UnexpectedResponse as exc:
                if exc.status_code == 404:
                    print(f"[WARN] missing qdrant point during invalidation edge_id={e.id} point_id={point_id}")
                    continue
                raise


async def insert_edge(
    db: AsyncSession,
    fact: TripletFact,
    episode_id: str,
    user_id: str,
    qdrant_client,
) -> str:
    """Insert new semantic edge and upsert vector; return edge id."""
    now = datetime.utcnow()
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
        qdrant_id=edge_id,
        episode_id=episode_id,
        confidence=fact.confidence,
        valid_at=now,
        invalid_at=None,
        created_at=now,
    )
    db.add(row)
    await db.flush()
    await _maybe_update_profile(db, user_id, fact)
    return edge_id


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
    """Invalidate a single semantic edge by id. Returns True if found and invalidated."""
    from sqlalchemy import select
    result = await db.execute(select(SemanticEdge).where(SemanticEdge.id == edge_id))
    edge = result.scalars().first()
    if edge is None:
        return False
    now = datetime.utcnow()
    await invalidate_edges(db, [edge], now, qdrant_client)
    return True


async def resolve_and_store(
    new_facts: list[TripletFact],
    user_id: str,
    episode_id: str,
    db: AsyncSession,
    qdrant_client=None,
) -> None:
    """
    For each fact: if no existing active edge -> insert.
    If duplicate -> skip. If contradiction -> invalidate existing then insert.
    Otherwise -> insert (coexisting).
    """
    if qdrant_client is None:
        qdrant_client = get_qdrant_client()
    now = datetime.utcnow()
    for fact in new_facts:
        existing = await get_active_edges(db, user_id, fact.subject.lower(), fact.relation)
        print(f"[DEBUG] existing='{existing}'")
        if not existing:
            await insert_edge(db, fact, episode_id, user_id, qdrant_client)
            continue
        if is_duplicate(fact, existing):
            continue
        if is_contradiction(fact, existing):
            await invalidate_edges(db, existing, now, qdrant_client)
            await insert_edge(db, fact, episode_id, user_id, qdrant_client)
        else:
            await insert_edge(db, fact, episode_id, user_id, qdrant_client)


async def _maybe_update_profile(db, user_id: str, fact: TripletFact) -> None:
    key = PROFILE_RELATIONS.get(fact.relation)
    print(f"[DEBUG] profile key={key} relation={fact.relation} object={fact.object}")
    if not key:
        return
    value = _profile_value_for_fact(fact)
    print(f"[DEBUG] profile value={value}")
    if value is None:
        return
    await set_fact(db, user_id, key, value)
    print(f"[DEBUG] profile set {key}={value} for {user_id}")
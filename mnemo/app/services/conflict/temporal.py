"""Bi-temporal helpers for semantic edges (no schema changes)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mnemo.app.db.models import SemanticEdge


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def safe_invalid_at(existing_valid_at: datetime, new_fact_valid_at: datetime) -> datetime:
    """Ensure invalid_at is strictly after valid_at."""
    floor = existing_valid_at + timedelta(milliseconds=1)
    return max(new_fact_valid_at, floor)


def parse_timestamp(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def retracted_at(edge: SemanticEdge) -> datetime | None:
    """Processing time when the fact was retracted (stored in metadata)."""
    return parse_timestamp((edge.metadata_ or {}).get("retracted_at"))


def stamp_retracted_at(edge: SemanticEdge, retracted_at: datetime) -> None:
    meta = dict(edge.metadata_ or {})
    meta["retracted_at"] = retracted_at.isoformat()
    edge.metadata_ = meta


def retracted_at_from_payload(payload: dict) -> datetime | None:
    return parse_timestamp(payload.get("retracted_at"))


def is_active_for_agent(edge: SemanticEdge) -> bool:
    """True when a fact is current and not retracted (safe for agent retrieve)."""
    return edge.invalid_at is None and retracted_at(edge) is None


def active_edge_sql_conditions():
    """SQLAlchemy conditions: invalid_at IS NULL and metadata.retracted_at absent."""
    from sqlalchemy import func, or_

    from mnemo.app.db.models import SemanticEdge

    return (
        SemanticEdge.invalid_at.is_(None),
        or_(
            SemanticEdge.metadata_.is_(None),
            func.json_extract(SemanticEdge.metadata_, "$.retracted_at").is_(None),
        ),
    )

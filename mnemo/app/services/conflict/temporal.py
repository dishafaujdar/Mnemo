"""Bi-temporal helpers for semantic edges (no schema changes)."""

from __future__ import annotations

from datetime import datetime, timezone

from mnemo.app.db.models import SemanticEdge


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


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

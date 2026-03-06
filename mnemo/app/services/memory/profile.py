"""User profile: key-value facts with Redis cache."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.db.models import ProfileFact
from mnemo.app.db.redis_client import get_cached_profile, invalidate_profile_cache, set_cached_profile


async def get_profile(db: AsyncSession, user_id: str, use_cache: bool = True) -> dict[str, str | list | float]:
    """Return user profile as dict key -> value (strings, list, or number). Check Redis first."""
    if use_cache:
        cached = await get_cached_profile(user_id)
        if cached is not None:
            return cached
    q = select(ProfileFact).where(ProfileFact.user_id == user_id)
    result = await db.execute(q)
    rows = result.scalars().all()
    profile: dict[str, str | list | float] = {}
    for row in rows:
        if row.value_type == "list":
            try:
                import json
                profile[row.key] = json.loads(row.value)
            except Exception:
                profile[row.key] = row.value
        elif row.value_type == "number":
            try:
                profile[row.key] = float(row.value)
            except ValueError:
                profile[row.key] = row.value
        else:
            profile[row.key] = row.value
    if use_cache and profile is not None:
        await set_cached_profile(user_id, profile)
    return profile


async def set_fact(
    db: AsyncSession,
    user_id: str,
    key: str,
    value: str | list | float,
    value_type: str | None = None,
) -> None:
    """Upsert a single profile fact; invalidate cache."""
    now = datetime.utcnow()
    if value_type is None:
        value_type = "string" if isinstance(value, str) else "list" if isinstance(value, list) else "number"
    if isinstance(value, list):
        import json
        value_str = json.dumps(value)
    else:
        value_str = str(value)
    stmt = insert(ProfileFact).values(
        id=str(uuid4()),
        user_id=user_id,
        key=key,
        value=value_str,
        value_type=value_type,
        updated_at=now,
    ).on_conflict_do_update(
        index_elements=["user_id", "key"],
        set_={"value": value_str, "value_type": value_type, "updated_at": now},
    )
    await db.execute(stmt)
    await db.flush()
    await invalidate_profile_cache(user_id)

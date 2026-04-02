"""Redis client for profile cache."""

from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from mnemo.app.core.config import settings

_redis: Redis | None = None


def get_redis() -> Redis:
    """Get or create async Redis connection."""
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def get_cached_profile(user_id: str) -> dict[str, Any] | None:
    """Return cached profile dict or None if miss."""
    try:
        r = get_redis()
        raw = await r.get(f"mnemo:profile:{user_id}")
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        return None


async def set_cached_profile(user_id: str, profile: dict[str, Any], ttl_seconds: int = 3600) -> None:
    """Cache profile dict."""
    try:
        r = get_redis()
        await r.set(
            f"mnemo:profile:{user_id}",
            json.dumps(profile),
            ex=ttl_seconds,
        )
    except Exception:
        pass


async def invalidate_profile_cache(user_id: str) -> None:
    """Remove cached profile after update."""
    try:
        r = get_redis()
        await r.delete(f"mnemo:profile:{user_id}")
    except Exception:
        pass

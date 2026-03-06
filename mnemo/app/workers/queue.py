"""Enqueue extraction job (called from ingest route)."""

from __future__ import annotations

from arq import create_pool
from arq.connections import RedisSettings

from mnemo.app.core.config import settings


async def enqueue_extraction(episode_id: str, user_id: str) -> bool:
    """Enqueue run_extraction job. Returns True if enqueued, False on error."""
    try:
        redis_settings = RedisSettings.from_dsn(settings.redis_url)
        pool = await create_pool(redis_settings)
        await pool.enqueue_job("run_extraction", episode_id, user_id)
        await pool.close()
        return True
    except Exception:
        return False

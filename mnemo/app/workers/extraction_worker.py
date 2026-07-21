"""ARQ worker: run extraction and conflict resolution in background."""

from __future__ import annotations

import logging

from arq.connections import RedisSettings, create_pool

from mnemo.app.core.config import settings
from mnemo.app.db.sqlite import async_session_factory
from mnemo.app.services.conflict.resolver import rebuild_missing_qdrant_points
from mnemo.app.services.extraction.service import process_episode_extraction
from mnemo.app.db.qdrant import get_qdrant_client

logger = logging.getLogger(__name__)


async def run_extraction(ctx: dict, episode_id: str, user_id: str) -> None:
    """Load episode, extract facts, resolve and store semantic edges."""
    try:
        async with async_session_factory() as db:
            count = await process_episode_extraction(db, episode_id, user_id)
            await db.commit()
            logger.info(
                "extraction complete episode=%s user=%s facts_stored=%d",
                episode_id,
                user_id,
                count,
            )
    except Exception:
        logger.exception(
            "extraction worker failed episode=%s user=%s",
            episode_id,
            user_id,
        )
        raise


async def repair_qdrant_points(ctx: dict, user_id: str | None = None) -> int:
    """Rebuild active semantic edges that exist in SQLite but are missing in Qdrant."""
    async with async_session_factory() as db:
        qdrant = get_qdrant_client()
        rebuilt = await rebuild_missing_qdrant_points(db, qdrant, user_id=user_id)
        await db.commit()
        return rebuilt


async def startup(ctx: dict) -> None:
    """Worker startup."""
    ctx["redis"] = await create_pool(RedisSettings.from_dsn(settings.redis_url))


async def shutdown(ctx: dict) -> None:
    """Worker shutdown."""
    if "redis" in ctx:
        await ctx["redis"].close()


class WorkerSettings:
    functions = [run_extraction, repair_qdrant_points]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = settings.extraction_concurrency

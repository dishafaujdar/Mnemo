"""ARQ worker: run extraction and conflict resolution in background."""

from __future__ import annotations

from arq.connections import ArqRedis, RedisSettings, create_pool

from mnemo.app.core.config import settings
from mnemo.app.db.qdrant import ensure_collection, get_qdrant_client
from mnemo.app.db.sqlite import async_session_factory
from mnemo.app.services.conflict.resolver import rebuild_missing_qdrant_points, resolve_and_store
from mnemo.app.services.extraction.pipeline import extract_facts
from mnemo.app.services.memory.episodic import get_episode


async def run_extraction(ctx: dict, episode_id: str, user_id: str) -> None:
    """Load episode, extract facts, resolve and store semantic edges."""
    async with async_session_factory() as db:
        episode = await get_episode(db, episode_id)
        if episode is None:
            return
        content = episode.content or ""
        facts = await extract_facts(content, settings.spacy_model)
        print(f"facts",facts)
        if not facts:
            await db.commit()
            return
        qdrant = get_qdrant_client()
        await ensure_collection(qdrant)
        await resolve_and_store(facts, user_id, episode_id, db, qdrant)
        await db.commit()


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

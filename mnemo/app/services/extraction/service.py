"""Shared extraction + storage path for ingest (sync) and worker (async)."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.core.config import settings
from mnemo.app.db.qdrant import ensure_collection, get_qdrant_client
from mnemo.app.models.extraction import REVIEW_CONFIRMED, TripletFact
from mnemo.app.services.conflict.resolver import (
    reconcile_active_edges,
    resolve_and_store,
    sync_profile_from_active_edges,
)
from mnemo.app.services.extraction.pipeline import extract_facts
from mnemo.app.services.extraction.judge import judge_action, ACTION_STORE
from mnemo.app.services.extraction.validators import dedupe_batch_facts, validate_fact_against_source
from mnemo.app.services.memory.episodic import get_episode

logger = logging.getLogger(__name__)


def filter_storable_facts(facts: list[TripletFact], source_text: str = "") -> list[TripletFact]:
    """Keep facts that pass judge, ontology, and source-text gates."""
    storable: list[TripletFact] = []
    for fact in facts:
        if not fact.relation or not fact.object.strip():
            continue
        if judge_action(fact.confidence) != ACTION_STORE:
            continue
        if fact.review_status != REVIEW_CONFIRMED:
            continue
        if source_text and not validate_fact_against_source(source_text, fact):
            continue
        storable.append(fact)
    return dedupe_batch_facts(storable, source_text=source_text)


async def process_episode_extraction(
    db: AsyncSession,
    episode_id: str,
    user_id: str,
) -> int:
    """Extract facts from an episode, store confirmed ones, sync profile. Returns count stored."""
    episode = await get_episode(db, episode_id)
    if episode is None:
        logger.warning("Episode not found: %s", episode_id)
        return 0

    content = episode.content or ""
    if not content.strip():
        return 0

    try:
        facts = await extract_facts(content, user_id)
    except Exception:
        logger.exception("Extraction failed for episode=%s user=%s", episode_id, user_id)
        raise

    storable = filter_storable_facts(facts, source_text=content)
    logger.info(
        "episode=%s user=%s extracted=%d storable=%d",
        episode_id,
        user_id,
        len(facts),
        len(storable),
    )

    qdrant = get_qdrant_client()
    await ensure_collection(qdrant)

    if storable:
        await resolve_and_store(storable, user_id, episode_id, db, qdrant)
    else:
        retired = await reconcile_active_edges(db, user_id, qdrant)
        if retired:
            logger.info("episode=%s user=%s reconciled=%d legacy conflicts", episode_id, user_id, retired)

    await sync_profile_from_active_edges(db, user_id)
    return len(storable)

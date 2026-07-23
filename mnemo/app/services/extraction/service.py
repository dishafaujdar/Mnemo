"""Shared extraction + storage path for ingest (sync) and worker (async)."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.core.config import settings
from mnemo.app.db.qdrant import ensure_collection, get_qdrant_client
from mnemo.app.models.extraction import REVIEW_CONFIRMED, REVIEW_FUZZY, TripletFact
from mnemo.app.services.conflict.resolver import (
    reconcile_active_edges,
    resolve_and_store,
    sync_profile_from_active_edges,
)
from mnemo.app.services.extraction.pipeline import extract_facts
from mnemo.app.services.extraction.grounding import validate_grounding
from mnemo.app.services.extraction.needs_review import log_to_needs_review
from mnemo.app.services.extraction.relation_guards import validate_lives_in
from mnemo.app.services.extraction.validators import dedupe_batch_facts, validate_fact_against_source
from mnemo.app.services.memory.episodic import get_episode

logger = logging.getLogger(__name__)


async def gate_facts(
    db: AsyncSession,
    user_id: str,
    episode_id: str,
    facts: list[TripletFact],
    source_text: str = "",
) -> list[TripletFact]:
    """Steps 3–4: grounding, relation guards, confidence gate, dedupe."""
    storable: list[TripletFact] = []
    for fact in facts:
        if not fact.relation or not fact.object.strip():
            continue

        if fact.temporal_status == "future":
            await log_to_needs_review(
                db, user_id=user_id, episode_id=episode_id, fact=fact, rejection_reason="future_fact_skipped"
            )
            continue

        if source_text and not validate_grounding(fact, source_text):
            await log_to_needs_review(
                db, user_id=user_id, episode_id=episode_id, fact=fact, rejection_reason="ungrounded"
            )
            continue

        if not validate_lives_in(fact):
            await log_to_needs_review(
                db,
                user_id=user_id,
                episode_id=episode_id,
                fact=fact,
                rejection_reason="LIVES_IN object failed location validation",
            )
            continue

        if fact.confidence < settings.confidence_store_threshold:
            await log_to_needs_review(
                db, user_id=user_id, episode_id=episode_id, fact=fact, rejection_reason="low_confidence"
            )
            continue

        if fact.review_status not in {REVIEW_CONFIRMED, REVIEW_FUZZY}:
            continue
        if source_text and not validate_fact_against_source(source_text, fact):
            await log_to_needs_review(
                db,
                user_id=user_id,
                episode_id=episode_id,
                fact=fact,
                rejection_reason="failed_source_validation",
            )
            continue
        storable.append(fact)

    return dedupe_batch_facts(storable, source_text=source_text)


async def process_episode_extraction(
    db: AsyncSession,
    episode_id: str,
    user_id: str,
) -> int:
    episode = await get_episode(db, episode_id)
    if episode is None:
        logger.warning("Episode not found: %s", episode_id)
        return 0

    content = episode.content or ""
    if not content.strip():
        return 0

    try:
        extraction = await extract_facts(content, user_id)
    except Exception:
        logger.exception("Extraction failed for episode=%s user=%s", episode_id, user_id)
        raise

    storable = await gate_facts(db, user_id, episode_id, extraction.facts, source_text=content)
    logger.info(
        "episode=%s user=%s extracted=%d storable=%d retract_others=%s",
        episode_id,
        user_id,
        len(extraction.facts),
        len(storable),
        extraction.retract_others_in_category,
    )

    qdrant = get_qdrant_client()
    await ensure_collection(qdrant)

    if storable:
        await resolve_and_store(
            storable,
            user_id,
            episode_id,
            db,
            qdrant,
            retract_others_in_category=extraction.retract_others_in_category,
        )
    else:
        retired = await reconcile_active_edges(db, user_id, qdrant)
        if retired:
            logger.info("episode=%s user=%s reconciled=%d legacy conflicts", episode_id, user_id, retired)

    await sync_profile_from_active_edges(db, user_id)
    return len(storable)

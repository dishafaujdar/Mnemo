"""Episodic store: immutable raw turn storage."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.db.models import Episode
from mnemo.app.models.memory import IngestMessage


def _gen_id() -> str:
    return str(uuid4())


async def store_turn(
    session: AsyncSession,
    user_id: str,
    messages: list[IngestMessage],
    session_id: str | None = None,
    metadata: dict | None = None,
) -> str:
    """
    Store each message as an immutable episode row.
    Returns the id of the first (user) message episode for extraction provenance.
    """
    now = datetime.utcnow()
    first_episode_id: str | None = None
    for msg in messages:
        episode_id = _gen_id()
        if first_episode_id is None:
            first_episode_id = episode_id
        episode = Episode(
            id=episode_id,
            user_id=user_id,
            role=msg.role,
            content=msg.content,
            created_at=now,
            session_id=session_id,
            metadata_=metadata,
        )
        session.add(episode)
    await session.flush()
    return first_episode_id or _gen_id()


async def get_episode(session: AsyncSession, episode_id: str) -> Episode | None:
    """Load a single episode by id."""
    result = await session.execute(select(Episode).where(Episode.id == episode_id))
    return result.scalars().first()

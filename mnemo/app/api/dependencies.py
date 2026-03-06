"""FastAPI dependencies: auth, DB sessions, etc."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.core.security import require_api_key
from mnemo.app.db.sqlite import async_session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


__all__ = ["require_api_key", "get_session"]

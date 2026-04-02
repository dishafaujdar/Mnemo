"""User profile CRUD."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from mnemo.app.api.dependencies import get_session, require_api_key
from mnemo.app.db.models import ProfileFact
from mnemo.app.services.memory.profile import get_profile as get_profile_service
from sqlalchemy import func, select


router = APIRouter(prefix="/memory", tags=["profile"])


@router.get("/profile")
async def get_profile_route(
    user_id: str,
    _api_key: str = Depends(require_api_key),
    session: AsyncSession = Depends(get_session),
):
    """Persistent user facts."""
    facts = await get_profile_service(session, user_id)
    r = await session.execute(
        select(func.max(ProfileFact.updated_at)).where(ProfileFact.user_id == user_id)
    )
    last_updated = r.scalar() or None
    return {"user_id": user_id, "facts": facts, "last_updated": last_updated}

"""API key validation for Mnemo."""

from fastapi import Header, HTTPException, status

from mnemo.app.core.config import settings


def require_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """Validate X-API-Key header; raise 401 if missing or invalid."""
    if not settings.mnemo_api_key:
        return x_api_key or ""
    if not x_api_key or x_api_key != settings.mnemo_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key

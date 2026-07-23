"""FastAPI app factory, lifespan, middleware."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mnemo.app.api.routes import memory, profile, search

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure Qdrant collection exists. Shutdown: close connections."""
    from mnemo.app.db.qdrant import ensure_collection, get_qdrant_client
    from mnemo.app.services.extraction.llm_client import verify_llm_connection
    try:
        qdrant = get_qdrant_client()
        await ensure_collection(qdrant)
    except Exception:
        pass
    await verify_llm_connection()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mnemo",
        description="Memory middleware REST API for AI agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(memory.router)
    app.include_router(search.router)
    app.include_router(profile.router)

    @app.get("/health")
    async def health():
        """Health check; no auth required."""
        return {"status": "ok"}

    return app


app = create_app()

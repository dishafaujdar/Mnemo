"""FastAPI app factory, lifespan, middleware."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from mnemo.app.api.routes import memory, profile, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure Qdrant collection exists. Shutdown: close connections."""
    from mnemo.app.db.qdrant import ensure_collection, get_qdrant_client
    try:
        qdrant = get_qdrant_client()
        await ensure_collection(qdrant)
    except Exception:
        pass
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

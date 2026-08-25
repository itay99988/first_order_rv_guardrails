"""
FastAPI application entry point.

Assembles the app with CORS, lifespan, and router mounting.
In production (Docker), serves the built frontend static files.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import get_config
from backend.store.db import DatabaseStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup, close on shutdown."""
    config = get_config()
    db = DatabaseStore(config.database_path)
    await db.initialize()
    app.state.db = db
    app.state.config = config
    yield
    await db.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="DejaVuGuard",
        description="Runtime Verification for LLM Conversations",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and mount routers
    from backend.routers.chat import router as chat_router
    from backend.routers.playbooks import router as playbooks_router
    from backend.routers.policies import router as policies_router
    from backend.routers.rules import router as rules_router
    from backend.routers.settings import router as settings_router

    app.include_router(settings_router, prefix="/api")
    app.include_router(policies_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")
    app.include_router(playbooks_router, prefix="/api")
    app.include_router(rules_router, prefix="/api")

    @app.get("/api/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    # Serve built frontend static files in production (Docker)
    static_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if static_dir.is_dir():
        from fastapi.responses import FileResponse

        # Serve JS/CSS bundles
        app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")

        # SPA catch-all: serve static files if they exist, otherwise index.html
        @app.get("/{path:path}")
        async def spa_fallback(path: str):
            file_path = static_dir / path
            if path and file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(static_dir / "index.html"))

    return app


app = create_app()

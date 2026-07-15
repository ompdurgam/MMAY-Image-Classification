"""
app/application.py
FastAPI app factory — creates and configures the application instance.
Lifespan (startup/shutdown) lives here, not in main.py.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import batch_predict, health, predict
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.model_store import set_model
from app.services.model_loader import load_keras_model

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: validate config + load model.  Shutdown: release model."""
    cfg = get_settings()
    setup_logging(cfg.LOG_LEVEL)
    logger.info("MMAY API starting up …")

    try:
        cfg.validate()                          # fail fast on bad config
        model = load_keras_model(cfg.MODEL_PATH)
        set_model(model)                        # store in model_store
        logger.info(
            "Model ready. Confidence threshold: %.0f%%",
            cfg.CONFIDENCE_THRESHOLD * 100,
        )
    except Exception as exc:
        # Log the full error — but DO NOT crash the process so that the
        # /health endpoint can report "degraded" and ops can investigate.
        logger.error("Startup error — model unavailable: %s", exc, exc_info=True)

    yield

    set_model(None)
    logger.info("MMAY API shut down.")


def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title       = "MMAY Image Verification API",
        description = (
            "Verifies construction-stage photographs for the "
            "**Mukhyamantri Avas Yojana** scheme."
        ),
        version     = "2.0.0",
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ── CORS ────────────────────────────────────────────────────────────────────
    # Origins are read from the ALLOWED_ORIGINS env var (see config.py).
    # Default is localhost only — set ALLOWED_ORIGINS=* only for local dev.
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = cfg.ALLOWED_ORIGINS,
        allow_methods     = ["GET", "POST"],
        allow_headers     = ["Content-Type", "Authorization"],
        allow_credentials = False,
    )

    # ── Global exception handler ────────────────────────────────────────────────
    # Returns a generic message — never expose stack traces to clients.
    @app.exception_handler(Exception)
    async def _safe_exception_handler(request: Request, exc: Exception):
        # Log the full traceback server-side for debugging
        logger.error(
            "Unhandled exception on %s %s",
            request.method,
            request.url.path,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred. Please try again later."},
        )

    # ── Routers ─────────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(batch_predict.router)

    return app
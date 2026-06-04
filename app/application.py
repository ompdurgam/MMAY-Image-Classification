"""
app/application.py
FastAPI app factory — creates and configures the application instance.
Lifespan (startup/shutdown) lives here, not in main.py.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, predict
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
        # Log but don't crash — /health will report degraded
        logger.error("Startup error: %s", exc)

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
        version     = "1.0.0",
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],   # restrict to known domains in production
        allow_methods  = ["GET", "POST"],
        allow_headers  = ["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(predict.router)

    return app
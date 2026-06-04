"""
app/api/dependencies.py
FastAPI dependency functions — one place for all Depends() providers.
"""
from __future__ import annotations

from fastapi import HTTPException, status

from app.core.config import Settings, get_settings
from app.core.model_store import get_model, is_model_loaded


def require_model():
    """
    FastAPI dependency — injects the loaded Keras model.
    Returns 503 if model failed to load at startup.
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model is not available. Please try again later.",
        )
    return get_model()


def require_settings() -> type[Settings]:
    """FastAPI dependency — injects the Settings class."""
    return get_settings()
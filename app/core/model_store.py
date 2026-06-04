"""
app/core/model_store.py
Single global holder for the Keras model.
All code that needs the model goes through here — no bare globals elsewhere.
"""
from __future__ import annotations

from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

_model = None


def set_model(model) -> None:
    global _model
    _model = model
    logger.info("Model registered in model_store.")


def get_model():
    return _model


def is_model_loaded() -> bool:
    return _model is not None
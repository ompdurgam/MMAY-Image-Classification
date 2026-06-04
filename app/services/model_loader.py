"""
app/services/model_loader.py
Responsible only for loading the Keras model from disk.
"""
from __future__ import annotations

from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


def load_keras_model(model_path: str):
    """
    Load and return a Keras model from *model_path*.
    Raises FileNotFoundError if the file is missing.
    Raises RuntimeError if TensorFlow / Keras cannot load the file.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")

    try:
        from tensorflow import keras  # noqa: F401 – late import keeps startup fast
        model = keras.models.load_model(str(path))
        logger.info("Loaded model from '%s'.", path)
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load Keras model: {exc}") from exc
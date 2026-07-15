"""
app/core/config.py
All settings read from environment / .env via os.getenv.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()


def _parse_json_map(raw: str, env_var: str) -> dict:
    """
    Parse a JSON string from an environment variable.
    Raises a clear ValueError (not a bare JSONDecodeError) on bad input so
    the startup validator can surface a friendly message.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Environment variable '{env_var}' contains invalid JSON: {exc}"
        ) from exc


class Settings:
    # ── Model ──────────────────────────────────────────────────────────────────
    MODEL_PATH: str = os.getenv("MODEL_PATH", "model/MMAY_EfficientNetV2M_final.keras")
    IMG_HEIGHT: int = int(os.getenv("IMG_HEIGHT", "300"))
    IMG_WIDTH:  int = int(os.getenv("IMG_WIDTH",  "300"))

    # ── Inference ──────────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

    # ── Mappings ───────────────────────────────────────────────────────────────
    _LEVEL_CLASS_MAP_RAW: str = os.getenv(
        "LEVEL_CLASS_MAP",
        '{"2":"plinth","3":"roof_cast","4":"completion"}',
    )
    _CLASS_INDEX_MAP_RAW: str = os.getenv(
        "CLASS_INDEX_MAP",
        '{"0":"completion","1":"plinth","2":"roof_cast"}',
    )

    # ── Server ─────────────────────────────────────────────────────────────────
    APP_HOST:  str = os.getenv("APP_HOST",  "0.0.0.0")
    APP_PORT:  int = int(os.getenv("APP_PORT",  "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info").lower()

    # ── CORS ───────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.  Default is localhost only.
    # Set ALLOWED_ORIGINS=* in .env only for local development.
    _ALLOWED_ORIGINS_RAW: str = os.getenv("ALLOWED_ORIGINS", "http://localhost")
    ALLOWED_ORIGINS: List[str] = [
        o.strip() for o in _ALLOWED_ORIGINS_RAW.split(",") if o.strip()
    ]

    # ── Upload guards ──────────────────────────────────────────────────────────
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    _ALLOWED_IMAGE_TYPES_RAW: str = os.getenv(
        "ALLOWED_IMAGE_TYPES", "image/jpeg,image/png,image/webp"
    )

    # ── Batch endpoint — filesystem sandbox ────────────────────────────────────
    # ALL batch image paths must resolve to a file INSIDE this directory.
    # Requests that point outside this root are rejected (path-traversal guard).
    BATCH_IMAGE_ROOT: str = os.getenv("BATCH_IMAGE_ROOT", "images")

    # ── Derived (parsed once at class definition time) ─────────────────────────
    LEVEL_CLASS_MAP:  Dict[int, str] = {
        int(k): v
        for k, v in _parse_json_map(_LEVEL_CLASS_MAP_RAW, "LEVEL_CLASS_MAP").items()
    }
    CLASS_INDEX_MAP:  Dict[int, str] = {
        int(k): v
        for k, v in _parse_json_map(_CLASS_INDEX_MAP_RAW, "CLASS_INDEX_MAP").items()
    }
    ALLOWED_MIME_TYPES: List[str] = [
        t.strip() for t in _ALLOWED_IMAGE_TYPES_RAW.split(",")
    ]
    MAX_IMAGE_BYTES: int = MAX_IMAGE_SIZE_MB * 1024 * 1024
    NUM_CLASSES:     int = len(_parse_json_map(_CLASS_INDEX_MAP_RAW, "CLASS_INDEX_MAP"))

    @classmethod
    def validate(cls) -> None:
        """Called once at startup to catch mis-configuration early."""
        if not (0.0 < cls.CONFIDENCE_THRESHOLD <= 1.0):
            raise ValueError(
                f"CONFIDENCE_THRESHOLD must be in (0, 1], got {cls.CONFIDENCE_THRESHOLD}"
            )
        if not Path(cls.MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model file not found: {Path(cls.MODEL_PATH).resolve()}"
            )
        valid_levels = {"info", "debug", "warning", "error", "critical"}
        if cls.LOG_LEVEL not in valid_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of {valid_levels}, got '{cls.LOG_LEVEL}'"
            )
        # Ensure BATCH_IMAGE_ROOT directory exists (create if missing)
        batch_root = Path(cls.BATCH_IMAGE_ROOT)
        if not batch_root.exists():
            batch_root.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> type[Settings]:
    """Return the Settings class (singleton via lru_cache)."""
    return Settings
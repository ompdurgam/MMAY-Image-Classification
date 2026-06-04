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
class Settings:
    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_PATH: str = os.getenv("MODEL_PATH", "MMAY_Image_2-0.h5")
    IMG_HEIGHT: int = int(os.getenv("IMG_HEIGHT", "224"))
    IMG_WIDTH:  int = int(os.getenv("IMG_WIDTH",  "224"))
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
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    # ── Upload guards ──────────────────────────────────────────────────────────
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    _ALLOWED_IMAGE_TYPES_RAW: str = os.getenv(
        "ALLOWED_IMAGE_TYPES", "image/jpeg,image/png,image/webp"
    )
    # ── Derived (parsed once at class definition time) ─────────────────────────
    LEVEL_CLASS_MAP:  Dict[int, str] = {
        int(k): v for k, v in json.loads(_LEVEL_CLASS_MAP_RAW).items()
    }
    CLASS_INDEX_MAP:  Dict[int, str] = {
        int(k): v for k, v in json.loads(_CLASS_INDEX_MAP_RAW).items()
    }
    ALLOWED_MIME_TYPES: List[str] = [
        t.strip() for t in _ALLOWED_IMAGE_TYPES_RAW.split(",")
    ]
    MAX_IMAGE_BYTES: int = MAX_IMAGE_SIZE_MB * 1024 * 1024
    NUM_CLASSES:     int = len(json.loads(_CLASS_INDEX_MAP_RAW))
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
@lru_cache(maxsize=1)
def get_settings() -> type[Settings]:
    return Settings
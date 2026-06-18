"""
app/api/routes/health.py
"""
from pathlib import Path

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.model_store import is_model_loaded
from app.schemas.prediction import HealthResponse

router = APIRouter(tags=["Utility"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Returns service liveness, whether the ML model is ready, and the model name."""
    loaded = is_model_loaded()
    cfg = get_settings()
    model_name = Path(cfg.MODEL_PATH).stem   # e.g. "MMAY_Modelv2" from "model/MMAY_Modelv2.h5"
    return HealthResponse(
        status       = "ok" if loaded else "degraded",
        model_loaded = loaded,
        model_name   = model_name,
    )
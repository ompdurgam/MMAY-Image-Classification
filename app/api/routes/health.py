"""
app/api/routes/health.py
"""
from fastapi import APIRouter

from app.core.model_store import is_model_loaded
from app.schemas.prediction import HealthResponse

router = APIRouter(tags=["Utility"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Returns service liveness and whether the ML model is ready."""
    loaded = is_model_loaded()
    return HealthResponse(
        status       = "ok" if loaded else "degraded",
        model_loaded = loaded,
    )
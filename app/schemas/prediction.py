"""
app/schemas/prediction.py
Request / Response Pydantic models.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PredictionStatus(str, Enum):
    SUCCESS             = "success"
    MANUAL_CHECK_NEEDED = "manual_check_needed"


class PredictResponse(BaseModel):
    status:          PredictionStatus = Field(..., description="Verification outcome.")
    predicted_class:      Optional[str]    = Field(None,  description="Actual class label predicted by the model (null only if model index is unknown).")
    predicted_confidence: float            = Field(..., ge=0.0, le=100.0, description="Confidence (%) for the predicted class.")
    submitted_level: int              = Field(..., description="Construction level sent by caller.")
    expected_class:       str              = Field(..., description="Class label that maps to submitted_level.")
    expected_confidence:  float            = Field(..., ge=0.0, le=100.0, description="Confidence (%) for the expected class.")
    message:              str              = Field(..., description="Human-readable summary.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "predicted_class": "plinth",
                    "predicted_confidence": 92.3,
                    "submitted_level": 2,
                    "expected_class": "plinth",
                    "expected_confidence": 92.3,
                    "message": "Verification successful. Stage 'plinth' confirmed.",
                },
                {
                    "status": "manual_check_needed",
                    "predicted_class": "other",
                    "predicted_confidence": 55.0,
                    "submitted_level": 3,
                    "expected_class": "roof_cast",
                    "expected_confidence": 55.0,
                    "message": "Manual review required. Confidence below threshold.",
                },
            ]
        }
    }


class HealthResponse(BaseModel):
    status:       str  = Field(..., description="'ok' or 'degraded'.")
    model_loaded: bool = Field(..., description="Whether the ML model is ready.")
    version:      str  = "1.0.0"

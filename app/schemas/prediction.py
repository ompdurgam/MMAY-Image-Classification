"""
app/schemas/prediction.py
Request / Response Pydantic models.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictionStatus(str, Enum):
    SUCCESS             = "success"
    MANUAL_CHECK_NEEDED = "manual_check_needed"


# ── Single-image response (existing /predict endpoint) ───────────────────────

class PredictResponse(BaseModel):
    status:               PredictionStatus = Field(..., description="Verification outcome.")
    predicted_class:      Optional[str]    = Field(None,  description="Actual class label predicted by the model (null only if model index is unknown).")
    predicted_confidence: float            = Field(..., ge=0.0, le=100.0, description="Confidence (%) for the predicted class.")
    predicted_level:      Optional[int]    = Field(None, description="Construction level corresponding to predicted_class.")
    submitted_level:      int              = Field(..., description="Construction level sent by caller.")
    expected_class:       str              = Field(..., description="Class label that maps to submitted_level.")
    expected_confidence:  float            = Field(..., ge=0.0, le=100.0, description="Confidence (%) for the expected class.")
    image_id:             str              = Field(..., description="Caller-supplied image identifier, echoed back in the response.")
    message:              str              = Field(..., description="Human-readable summary.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "success",
                    "predicted_class": "plinth",
                    "predicted_confidence": 92.3,
                    "predicted_level": 2,
                    "submitted_level": 2,
                    "expected_class": "plinth",
                    "expected_confidence": 92.3,
                    "image_id": "23456P21",
                    "message": "Verification successful. Stage 'plinth' confirmed.",
                },
                {
                    "status": "manual_check_needed",
                    "predicted_class": "plinth",
                    "predicted_confidence": 55.0,
                    "predicted_level": 2,
                    "submitted_level": 3,
                    "expected_class": "roof_cast",
                    "expected_confidence": 55.0,
                    "image_id": "78901R03",
                    "message": "Manual review required. Confidence below threshold.",
                },
            ]
        }
    }


# ── Batch request / response (/predict/batch) ───────────────────────────────

class BatchImageItem(BaseModel):
    """Single image entry in a batch request."""
    image_id: str = Field(..., description="Caller-supplied image identifier, echoed back in each result.")
    image:    str = Field(..., description="File path to the image (absolute or relative to the server).")


class BatchPredictRequest(BaseModel):
    """
    Batch verification request.
    All images in the batch share the same construction level.
    """
    label:  int                = Field(..., description="Construction level: 2 (plinth), 3 (roof_cast), 4 (completion).")
    images: List[BatchImageItem] = Field(..., min_length=1, max_length=10, description="List of images to verify (1–10).")

    @field_validator("images")
    @classmethod
    def reject_empty_images(cls, v: List[BatchImageItem]) -> List[BatchImageItem]:
        if not v:
            raise ValueError("At least one image is required.")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "label": 2,
                    "images": [
                        {"image_id": "image_1", "image": "/images/image_1.jpg"},
                        {"image_id": "image_2", "image": "/images/image_2.jpg"},
                        {"image_id": "image_3", "image": "/images/image_3.jpg"},
                    ],
                }
            ]
        }
    }


class BatchImageResult(BaseModel):
    """Per-image result within a batch response."""
    status:               PredictionStatus = Field(..., description="Verification outcome.")
    predicted_class:      Optional[str]    = Field(None, description="Actual class label predicted by the model.")
    predicted_confidence: float            = Field(..., ge=0.0, le=100.0, description="Confidence (%) for the predicted class.")
    predicted_level:      Optional[int]    = Field(None, description="Construction level corresponding to predicted_class.")
    submitted_level:      int              = Field(..., description="Construction level sent by caller.")
    expected_class:       str              = Field(..., description="Class label that maps to submitted_level.")
    expected_confidence:  float            = Field(..., ge=0.0, le=100.0, description="Confidence (%) for the expected class.")
    image_id:             str              = Field(..., description="Caller-supplied image identifier.")
    message:              str              = Field(..., description="Human-readable summary.")


class BatchPredictResponse(BaseModel):
    """Wraps the per-image results for a batch prediction."""
    results: List[BatchImageResult] = Field(..., description="Per-image verification results.")


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str  = Field(..., description="'ok' or 'degraded'.")
    model_loaded: bool = Field(..., description="Whether the ML model is ready.")
    model_name:   str  = Field(..., description="Name of the loaded ML model file (without extension).")
    version:      str  = "2.0.0"

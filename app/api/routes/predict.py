"""
app/api/routes/predict.py
The route does NO business logic — it only:
  1. Validates inputs (via services)
  2. Calls services
  3. Maps the result to a response schema
"""
from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import require_model, require_settings
from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.prediction import PredictResponse, PredictionStatus
from app.services.image_validator import validate_and_read_image
from app.services.predictor import make_decision, run_inference
from app.services.preprocessing import decode_and_preprocess

router = APIRouter(tags=["Prediction"])
logger = get_logger(__name__)


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Verify construction stage image",
    responses={
        200: {"description": "Verification result"},
        400: {"description": "Empty or corrupt image"},
        413: {"description": "Image too large"},
        415: {"description": "Unsupported MIME type"},
        422: {"description": "Unknown construction level"},
        503: {"description": "Model unavailable"},
    },
)
async def verify_image(
    level: Annotated[
        int,
        Form(description="Construction level: 2 (plinth), 3 (roof_cast), 4 (completion)."),
    ],
    image: Annotated[
        UploadFile,
        File(description="Site photograph — JPEG / PNG / WebP."),
    ],
    model=Depends(require_model),
    cfg: type[Settings] = Depends(require_settings),
) -> PredictResponse:
    """
    ### MMAY Stage Verification

    | Level | Expected stage |
    |-------|----------------|
    | 2     | plinth         |
    | 3     | roof_cast      |
    | 4     | completion     |
    """

    # ── 1. Validate level ────────────────────────────────────────────────────
    if level not in cfg.LEVEL_CLASS_MAP:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Level {level} is not recognised. "
                f"Valid levels: {sorted(cfg.LEVEL_CLASS_MAP.keys())}."
            ),
        )
    expected_class: str = cfg.LEVEL_CLASS_MAP[level]

    # ── 2. Validate & read image bytes ───────────────────────────────────────
    image_bytes = await validate_and_read_image(
        image              = image,
        allowed_mime_types = cfg.ALLOWED_MIME_TYPES,
        max_image_bytes    = cfg.MAX_IMAGE_BYTES,
        max_image_size_mb  = cfg.MAX_IMAGE_SIZE_MB,
    )

    # ── 3. Pre-process ───────────────────────────────────────────────────────
    try:
        img_array = decode_and_preprocess(image_bytes, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
    except ValueError as exc:
        logger.warning("Image preprocessing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    # ── 4. Inference ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    predicted_label, confidence, all_confidences = run_inference(
        model, img_array, cfg.CLASS_INDEX_MAP,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Confidence the model assigned to the *expected* class
    expected_conf = all_confidences.get(expected_class, 0.0)

    logger.info(
        "level=%d | expected=%s | predicted=%s | conf=%.3f | expected_conf=%.3f | %.1f ms",
        level, expected_class, predicted_label, confidence, expected_conf, elapsed_ms,
    )

    # ── 5. Decision ───────────────────────────────────────────────────────────────
    result = make_decision(
        predicted_label      = predicted_label,
        confidence           = confidence,
        expected_class       = expected_class,
        expected_confidence  = expected_conf,
        confidence_threshold = cfg.CONFIDENCE_THRESHOLD,
    )

    # ── 6. Map to response schema ────────────────────────────────────────────
    if result.success:
        return PredictResponse(
            status               = PredictionStatus.SUCCESS,
            predicted_class      = result.predicted_class,
            predicted_confidence = result.confidence,
            submitted_level      = level,
            expected_class       = result.expected_class,
            expected_confidence  = result.expected_confidence,
            message              = f"Verification successful. {result.reason}",
        )

    return PredictResponse(
        status               = PredictionStatus.MANUAL_CHECK_NEEDED,
        predicted_class      = result.predicted_class,   # may be None only if index not in CLASS_INDEX_MAP
        predicted_confidence = result.confidence,
        submitted_level      = level,
        expected_class       = result.expected_class,
        expected_confidence  = result.expected_confidence,
        message              = f"Manual review required. {result.reason}",
    )
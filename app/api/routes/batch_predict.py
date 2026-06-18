"""
app/api/routes/batch_predict.py
Batch verification endpoint — accepts multiple image file-paths
under the same construction level and returns per-image results.

The route does NO business logic — it only:
  1. Validates inputs (via services)
  2. Calls services
  3. Maps the results to a response schema
"""
from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import require_model, require_settings
from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.prediction import (
    BatchImageResult,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionStatus,
)
from app.services.predictor import make_decision, run_inference
from app.services.preprocessing import decode_and_preprocess

router = APIRouter(tags=["Prediction"])
logger = get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _reverse_level_lookup(
    level_class_map: Dict[int, str],
    class_name: Optional[str],
) -> Optional[int]:
    """Return the construction level for a given class name, or None."""
    if class_name is None:
        return None
    for lvl, name in level_class_map.items():
        if name == class_name:
            return lvl
    return None


def _read_and_validate_file(
    file_path: str,
    allowed_mime_types: list[str],
    max_image_bytes: int,
    max_image_size_mb: int,
    image_id: str,
) -> bytes:
    """
    Read an image from disk and apply the same validation guards
    that the single-upload endpoint uses (MIME type, size, non-empty).

    Raises HTTPException on any violation.
    """
    path = Path(file_path)

    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image '{image_id}': file not found at '{file_path}'.",
        )
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image '{image_id}': path '{file_path}' is not a regular file.",
        )

    # MIME type check (based on file extension)
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type not in allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Image '{image_id}': unsupported type '{mime_type}'. "
                f"Allowed: {', '.join(allowed_mime_types)}."
            ),
        )

    # Read bytes
    image_bytes = path.read_bytes()

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image '{image_id}': file at '{file_path}' is empty.",
        )

    if len(image_bytes) > max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image '{image_id}': file size "
                f"({len(image_bytes) / 1024 / 1024:.1f} MB) "
                f"exceeds the limit of {max_image_size_mb} MB."
            ),
        )

    return image_bytes


# ── Route ────────────────────────────────────────────────────────────────────

@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch verify construction stage images",
    responses={
        200: {"description": "Per-image verification results"},
        400: {"description": "Empty, corrupt, or missing image file"},
        413: {"description": "Image too large"},
        415: {"description": "Unsupported file type"},
        422: {"description": "Unknown construction level or invalid payload"},
        503: {"description": "Model unavailable"},
    },
)
async def batch_verify_images(
    body: BatchPredictRequest,
    model=Depends(require_model),
    cfg: type[Settings] = Depends(require_settings),
) -> BatchPredictResponse:
    """
    ### Batch MMAY Stage Verification

    Accepts a JSON body containing a construction **label** and a list of
    **image file paths**.  Each image is independently verified against the
    expected construction stage.

    | Level | Expected stage |
    |-------|----------------|
    | 2     | plinth         |
    | 3     | roof_cast      |
    | 4     | completion     |
    """

    # ── 1. Validate level ────────────────────────────────────────────────
    if body.label not in cfg.LEVEL_CLASS_MAP:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Level {body.label} is not recognised. "
                f"Valid levels: {sorted(cfg.LEVEL_CLASS_MAP.keys())}."
            ),
        )
    expected_class: str = cfg.LEVEL_CLASS_MAP[body.label]

    # ── 2. Check for duplicate image IDs ─────────────────────────────────
    seen_ids = set()
    for item in body.images:
        if item.image_id in seen_ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Duplicate image_id '{item.image_id}' in request.",
            )
        seen_ids.add(item.image_id)

    # ── 3. Process each image ────────────────────────────────────────────
    results: list[BatchImageResult] = []
    batch_t0 = time.perf_counter()

    for item in body.images:
        # 3a. Read & validate file from disk
        image_bytes = _read_and_validate_file(
            file_path         = item.image,
            allowed_mime_types= cfg.ALLOWED_MIME_TYPES,
            max_image_bytes   = cfg.MAX_IMAGE_BYTES,
            max_image_size_mb = cfg.MAX_IMAGE_SIZE_MB,
            image_id          = item.image_id,
        )

        # 3b. Pre-process
        try:
            img_array = decode_and_preprocess(image_bytes, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
        except ValueError as exc:
            logger.warning("Image '%s' preprocessing failed: %s", item.image_id, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image '{item.image_id}': {exc}",
            )

        # 3c. Inference
        t0 = time.perf_counter()
        predicted_label, confidence, all_confidences = run_inference(
            model, img_array, cfg.CLASS_INDEX_MAP,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        expected_conf = all_confidences.get(expected_class, 0.0)

        logger.info(
            "batch | id=%s | level=%d | expected=%s | predicted=%s | "
            "conf=%.3f | expected_conf=%.3f | %.1f ms",
            item.image_id, body.label, expected_class,
            predicted_label, confidence, expected_conf, elapsed_ms,
        )

        # 3d. Decision
        result = make_decision(
            predicted_label      = predicted_label,
            confidence           = confidence,
            expected_class       = expected_class,
            expected_confidence  = expected_conf,
            confidence_threshold = cfg.CONFIDENCE_THRESHOLD,
        )

        # 3e. Resolve predicted_level from predicted_class
        predicted_level = _reverse_level_lookup(cfg.LEVEL_CLASS_MAP, result.predicted_class)

        # 3f. Map to response schema
        if result.success:
            results.append(BatchImageResult(
                status               = PredictionStatus.SUCCESS,
                predicted_class      = result.predicted_class,
                predicted_confidence = round(result.confidence * 100, 2),
                predicted_level      = predicted_level,
                submitted_level      = body.label,
                expected_class       = result.expected_class,
                expected_confidence  = round(result.expected_confidence * 100, 2),
                image_id             = item.image_id,
                message              = f"Verification successful. {result.reason}",
            ))
        else:
            results.append(BatchImageResult(
                status               = PredictionStatus.MANUAL_CHECK_NEEDED,
                predicted_class      = result.predicted_class,
                predicted_confidence = round(result.confidence * 100, 2),
                predicted_level      = predicted_level,
                submitted_level      = body.label,
                expected_class       = result.expected_class,
                expected_confidence  = round(result.expected_confidence * 100, 2),
                image_id             = item.image_id,
                message              = f"Manual review required. {result.reason}",
            ))

    batch_elapsed = (time.perf_counter() - batch_t0) * 1000
    logger.info(
        "Batch complete: %d images in %.1f ms", len(results), batch_elapsed,
    )

    return BatchPredictResponse(results=results)

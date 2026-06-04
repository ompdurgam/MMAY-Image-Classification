"""
app/services/predictor.py
Inference + verification decision — the two concerns are kept separate.

FIX: Previously, `predict()` returned (None, confidence, False) when below
threshold, but the caller still checked `above_threshold` AND
`predicted_label == expected_class` as independent conditions.
This created a logical inconsistency: a None label could never equal any
class string, so the `above_threshold` flag was redundant and misleading.

Now the responsibilities are split cleanly:
  - `run_inference()`  → always returns the raw top label + confidence
  - `make_decision()`  → applies the threshold + match check and returns
                         a clear VerificationResult

The caller never has to interpret a partially-null tuple.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Data class returned to the route ─────────────────────────────────────────

@dataclass
class VerificationResult:
    success:         bool
    predicted_class: str | None   # present only on success
    confidence:      float
    expected_class:  str
    reason:          str          # always populated — explains the outcome


# ── Step 1: raw inference ─────────────────────────────────────────────────────

def run_inference(
    model,
    image_array: np.ndarray,
    class_index_map: Dict[int, str],
) -> tuple[str | None, float]:
    """
    Run model.predict and return (top_label, confidence).
    Returns (None, confidence) if the predicted index is not in class_index_map.
    """
    probabilities: np.ndarray = model.predict(image_array, verbose=0)[0]
    top_index:     int        = int(np.argmax(probabilities))
    confidence:    float      = float(probabilities[top_index])
    top_label:     str | None = class_index_map.get(top_index)

    logger.debug("Raw inference: index=%d label=%s conf=%.4f", top_index, top_label, confidence)
    return top_label, confidence


# ── Step 2: business decision ─────────────────────────────────────────────────

def make_decision(
    predicted_label:      str | None,
    confidence:           float,
    expected_class:       str,
    confidence_threshold: float,
) -> VerificationResult:
    """
    Decide verification outcome based on class-label match and confidence.

    Decision table
    ──────────────────────────────────────────────────────────────
    predicted_label != expected_class   → MANUAL CHECK NEEDED (wrong stage)
    predicted_label == expected_class AND
      confidence < threshold            → MANUAL CHECK NEEDED (low confidence)
    predicted_label == expected_class AND
      confidence >= threshold           → SUCCESS
    ──────────────────────────────────────────────────────────────
    Note: unknown label (None) is treated as a mismatch.
    """
    label_match = predicted_label == expected_class   # None != any str → False

    if not label_match:
        return VerificationResult(
            success         = False,
            predicted_class = predicted_label,
            confidence      = round(confidence, 4),
            expected_class  = expected_class,
            reason          = (
                f"Predicted stage '{predicted_label}' does not match "
                f"expected stage '{expected_class}'."
            ),
        )

    if confidence < confidence_threshold:
        return VerificationResult(
            success         = False,
            predicted_class = predicted_label,
            confidence      = round(confidence, 4),
            expected_class  = expected_class,
            reason          = (
                f"Stage '{predicted_label}' matched, but confidence ({confidence:.2f}) "
                f"is below the required threshold ({confidence_threshold:.2f})."
            ),
        )

    return VerificationResult(
        success         = True,
        predicted_class = predicted_label,
        confidence      = round(confidence, 4),
        expected_class  = expected_class,
        reason          = f"Stage '{predicted_label}' confirmed.",
    )

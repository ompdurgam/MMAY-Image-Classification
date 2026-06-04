"""
app/services/image_validator.py
Validates uploaded image files before any ML processing.

FIX: Previous code checked content_type BEFORE reading bytes.
content_type is a client-supplied header and can be None or spoofed.
Correct order:
  1. Read bytes                  (required before any other check)
  2. Check not empty             (empty upload is a client error)
  3. Check file size             (avoid holding huge payloads in memory)
  4. Check MIME type header      (fast reject for obviously wrong types)
"""
from __future__ import annotations

from typing import List

from fastapi import HTTPException, UploadFile, status


async def validate_and_read_image(
    image:              UploadFile,
    allowed_mime_types: List[str],
    max_image_bytes:    int,
    max_image_size_mb:  int,
) -> bytes:
    """
    Read and validate the uploaded image.
    Returns raw bytes on success; raises HTTPException on any violation.
    """
    # Step 1 – read all bytes first
    image_bytes: bytes = await image.read()

    # Step 2 – reject empty uploads
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty.",
        )

    # Step 3 – reject oversized uploads
    if len(image_bytes) > max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image size ({len(image_bytes) / 1024 / 1024:.1f} MB) "
                f"exceeds the limit of {max_image_size_mb} MB."
            ),
        )

    # Step 4 – reject unsupported MIME types
    # content_type may be None if client omits Content-Type header
    content_type: str = image.content_type or ""
    if content_type not in allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported image type '{content_type}'. "
                f"Allowed: {', '.join(allowed_mime_types)}."
            ),
        )

    return image_bytes
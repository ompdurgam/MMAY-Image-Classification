"""
app/services/image_validator.py
Validates uploaded image files before any ML processing.

Validation order:
  1. Read bytes                  (required before any other check)
  2. Check not empty             (empty upload is a client error)
  3. Check file size             (avoid holding huge payloads in memory)
  4. Check MIME type header      (fast reject for obviously wrong content-type)
  5. Magic-byte validation       (Pillow actually decodes the header bytes —
                                  this catches spoofed Content-Type headers)
"""
from __future__ import annotations

import io
from typing import List

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError


# Map of MIME types to their canonical PIL format identifiers.
# Only formats listed here pass the magic-byte check.
_MIME_TO_PIL_FORMAT: dict[str, str] = {
    "image/jpeg": "JPEG",
    "image/png":  "PNG",
    "image/webp": "WEBP",
}


def _verify_magic_bytes(image_bytes: bytes, content_type: str) -> None:
    """
    Verify the actual file content matches the declared content type.

    Uses Pillow's Image.open() + verify() which reads only the image header
    (magic bytes) without fully decoding the image — fast and safe.

    Raises HTTPException(415) if the file content does not match, or if the
    file is not a valid image regardless of the declared type.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()   # Pillow reads the header; raises on corrupt/fake images
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File content is not a recognised image format.",
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file appears corrupt or truncated.",
        )

    # Re-open after verify() (verify() closes the internal file pointer)
    try:
        reopened = Image.open(io.BytesIO(image_bytes))
        actual_format = reopened.format  # e.g. "JPEG", "PNG", "WEBP"
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file appears corrupt or truncated.",
        )

    expected_format = _MIME_TO_PIL_FORMAT.get(content_type)
    if expected_format and actual_format != expected_format:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File content does not match the declared type '{content_type}' "
                f"(detected: {actual_format or 'unknown'})."
            ),
        )


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

    # Step 4 – reject unsupported MIME types (header check)
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

    # Step 5 – magic-byte validation: confirm the file IS what it claims to be.
    # This prevents spoofed Content-Type attacks (e.g. .exe sent as image/jpeg).
    _verify_magic_bytes(image_bytes, content_type)

    return image_bytes
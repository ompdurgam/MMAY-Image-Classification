"""
app/services/preprocessing.py
Image decoding, resizing, and normalisation.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError


def decode_and_preprocess(
    image_bytes: bytes,
    img_height:  int,
    img_width:   int,
) -> np.ndarray:
    """
    Decode *image_bytes*, resize to (img_height × img_width), normalise to [0, 1].

    Returns ndarray of shape (1, H, W, 3).
    Raises ValueError for corrupt / non-image data.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot identify image file: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Image decode error: {exc}") from exc

    img   = img.resize((img_width, img_height), Image.BILINEAR)
    arr   = np.array(img, dtype=np.float32)
    arr   = (arr / 127.5) - 1.0  # MobileNetV2 expects [-1, 1]
    return np.expand_dims(arr, axis=0)   # (1, H, W, 3)
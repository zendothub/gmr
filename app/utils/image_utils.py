"""Image utility functions for crop extraction and frame processing."""

import asyncio
import io
import uuid
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from app.config import get_settings


def extract_crop(frame: np.ndarray, bbox: dict, padding_pct: float = 0.10) -> Optional[np.ndarray]:
    """
    Extract a crop from a frame using bounding box coordinates.

    Args:
        frame: The full video frame (numpy array)
        bbox: dict with x1, y1, x2, y2
        padding_pct: Padding as a fraction of bbox dimensions (e.g. 0.10 = 10% on all sides)
                     Applied as a percentage of width (for horizontal) and height (for vertical).

    Returns:
        Cropped image as numpy array, or None if invalid
    """
    try:
        h, w = frame.shape[:2]
        bbox_h = int(bbox["y2"]) - int(bbox["y1"])
        bbox_w = int(bbox["x2"]) - int(bbox["x1"])
        pad_y = int(bbox_h * padding_pct)
        pad_x = int(bbox_w * padding_pct)

        x1 = max(0, int(bbox["x1"]) - pad_x)
        y1 = max(0, int(bbox["y1"]) - pad_y)
        x2 = min(w, int(bbox["x2"]) + pad_x)
        y2 = min(h, int(bbox["y2"]) + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop
    except Exception as e:
        logger.error(f"Failed to extract crop: {e}")
        return None


def resize_crop(crop: np.ndarray, target_size: Tuple[int, int] = (128, 256)) -> np.ndarray:
    """Resize a crop to a standard size for ReID model input."""
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)


def save_image(image: np.ndarray, directory: str, prefix: str = "img") -> Optional[str]:
    """
    Encode image to JPEG bytes and upload to MinIO.

    Args:
        image: numpy array image
        directory: MinIO object-name prefix (e.g. "crops", "snapshots")
        prefix: Filename prefix

    Returns:
        MinIO object key on success, or None on failure.
    """
    try:
        from app.modules.storage.minio_client import upload_image

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        object_name = f"{directory}/{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"

        result = upload_image(image, object_name)
        if result is None:
            logger.error(f"MinIO upload failed for object_name={object_name}")
            return None
        logger.debug(f"Image uploaded to MinIO: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to upload image: {e}")
        return None


async def save_image_async(image: np.ndarray, directory: str, prefix: str = "img") -> Optional[str]:
    """Async wrapper: run JPEG encode + MinIO upload in a background thread.

    Offloads the synchronous ``cv2.imencode`` and MinIO HTTP PUT from the
    FastAPI event-loop thread so that high-throughput frame processing does
    not starve HTTP request handlers (login, debug, analytics).
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, save_image, image, directory, prefix)


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 80) -> Optional[bytes]:
    """Convert a frame to JPEG bytes for streaming."""
    try:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()
    except Exception as e:
        logger.error(f"Failed to encode frame: {e}")
        return None

"""Image utility functions for crop extraction and frame processing."""

import io
import uuid
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from app.config import get_settings


def extract_crop(frame: np.ndarray, bbox: dict, padding: int = 10) -> Optional[np.ndarray]:
    """
    Extract a crop from a frame using bounding box coordinates.

    Args:
        frame: The full video frame (numpy array)
        bbox: dict with x1, y1, x2, y2
        padding: Extra pixels around the bbox

    Returns:
        Cropped image as numpy array, or None if invalid
    """
    try:
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox["x1"]) - padding)
        y1 = max(0, int(bbox["y1"]) - padding)
        x2 = min(w, int(bbox["x2"]) + padding)
        y2 = min(h, int(bbox["y2"]) + padding)

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


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 80) -> Optional[bytes]:
    """Convert a frame to JPEG bytes for streaming."""
    try:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()
    except Exception as e:
        logger.error(f"Failed to encode frame: {e}")
        return None

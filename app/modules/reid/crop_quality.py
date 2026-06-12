"""Crop quality assessment for ReID."""

import cv2
import numpy as np
from loguru import logger


def assess_crop_quality(crop: np.ndarray) -> float:
    """
    Assess the quality of a person crop for ReID.

    Factors:
    - Image sharpness (Laplacian variance)
    - Brightness distribution
    - Minimum size
    - Aspect ratio

    Args:
        crop: Person crop image (BGR)

    Returns:
        Quality score between 0.0 and 1.0
    """
    try:
        if crop is None or crop.size == 0:
            return 0.0

        h, w = crop.shape[:2]

        # Minimum size check
        if h < 50 or w < 25:
            return 0.0

        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Sharpness score (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 500.0)

        # Brightness score
        mean_brightness = np.mean(gray)
        if mean_brightness < 30 or mean_brightness > 240:
            brightness_score = 0.3
        elif mean_brightness < 60 or mean_brightness > 200:
            brightness_score = 0.6
        else:
            brightness_score = 1.0

        # Size score
        pixel_count = h * w
        size_score = min(1.0, pixel_count / (128 * 256))

        # Aspect ratio score (person crops should be taller than wide)
        aspect = h / w if w > 0 else 0
        if 1.5 <= aspect <= 3.5:
            aspect_score = 1.0
        elif 1.0 <= aspect <= 4.5:
            aspect_score = 0.7
        else:
            aspect_score = 0.4

        # Weighted combination
        quality = (
            0.35 * sharpness
            + 0.25 * brightness_score
            + 0.20 * size_score
            + 0.20 * aspect_score
        )

        return round(min(1.0, max(0.0, quality)), 3)

    except Exception as e:
        logger.error(f"Crop quality assessment failed: {e}")
        return 0.0

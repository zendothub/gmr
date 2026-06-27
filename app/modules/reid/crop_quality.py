"""Crop quality assessment for ReID with enhanced keypoint-based formula."""

import cv2
import numpy as np
from loguru import logger


# ReID quality gate constants
REID_MIN_VISIBLE_KEYPOINT_RATIO = 0.50  # Minimum torso keypoint visibility
TORSO_KP_INDICES = [5, 6, 11, 12]  # left_shoulder, right_shoulder, left_hip, right_hip


def check_torso_keypoints(keypoints_data, confidence_threshold: float = 0.3) -> tuple:
    """
    Check torso keypoint visibility for body ReID quality gate.
    
    Args:
        keypoints_data: YOLO-Pose keypoints result (from pose model prediction)
        confidence_threshold: Minimum confidence for a keypoint to be considered visible
        
    Returns:
        Tuple of (visibility_ratio, passed_gate)
    """
    try:
        if keypoints_data is None:
            return 0.0, False
        
        # Handle different keypoint data formats
        if hasattr(keypoints_data, 'conf'):
            kp_conf = keypoints_data.conf
            if kp_conf is None or kp_conf.shape[0] == 0:
                return 0.0, False
            person_kp = kp_conf[0].cpu().numpy()
        elif isinstance(keypoints_data, (np.ndarray, list)):
            # Direct numpy array or list of confidences
            person_kp = np.array(keypoints_data) if isinstance(keypoints_data, list) else keypoints_data
        else:
            return 0.0, False
        
        # Check 4 critical torso keypoints: shoulders (5,6) and hips (11,12)
        visible_count = 0
        for idx in TORSO_KP_INDICES:
            if idx < len(person_kp) and person_kp[idx] >= confidence_threshold:
                visible_count += 1
        
        visibility_ratio = visible_count / len(TORSO_KP_INDICES)
        passed_gate = visibility_ratio >= REID_MIN_VISIBLE_KEYPOINT_RATIO
        
        return visibility_ratio, passed_gate
        
    except Exception as e:
        logger.debug(f"Torso keypoint check failed: {e}")
        return 0.0, False


def assess_crop_quality(crop: np.ndarray, keypoint_visibility_ratio: float = None) -> float:
    """
    Enhanced crop quality assessment with weighted formula from reid_logic_explanation.md.
    
    Formula: keypoints(0.50) + sharpness(0.20) + size(0.15) + aspect(0.10) + brightness(0.05)
    
    This implements the Body Quality Gate from the ReID logic documentation:
    - If keypoint visibility ratio < 0.50, quality = 0.0 (immediate reject)
    - Enhanced weighted formula prioritizes keypoints (50% weight)
    - Threshold gate: quality < 0.50 rejects body ReID

    Args:
        crop: Person crop image (BGR)
        keypoint_visibility_ratio: Torso keypoint visibility ratio (0.0-1.0)
                                   from YOLO-Pose torso keypoints check

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

        # Body Quality Gate: If keypoint ratio is below threshold, immediately return 0.0
        if keypoint_visibility_ratio is not None and keypoint_visibility_ratio < REID_MIN_VISIBLE_KEYPOINT_RATIO:
            logger.debug(
                f"Crop rejected by torso keypoint gate: "
                f"visibility_ratio={keypoint_visibility_ratio:.2f} < {REID_MIN_VISIBLE_KEYPOINT_RATIO}"
            )
            return 0.0

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 1. Keypoints score (0.50 weight - highest priority)
        if keypoint_visibility_ratio is not None:
            keypoints_score = keypoint_visibility_ratio
        else:
            # If no keypoints provided, assume 0 (will result in low quality)
            keypoints_score = 0.0

        # 2. Sharpness score (0.20 weight) - Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500.0)

        # 3. Size score (0.15 weight)
        pixel_count = h * w
        size_score = min(1.0, pixel_count / (128 * 256))

        # 4. Aspect ratio score (0.10 weight) - person crops should be taller than wide
        aspect = h / w if w > 0 else 0
        if 1.5 <= aspect <= 3.5:
            aspect_score = 1.0
        elif 1.0 <= aspect <= 4.5:
            aspect_score = 0.7
        else:
            aspect_score = 0.4

        # 5. Brightness score (0.05 weight)
        mean_brightness = np.mean(gray)
        if mean_brightness < 30 or mean_brightness > 240:
            brightness_score = 0.3
        elif mean_brightness < 60 or mean_brightness > 200:
            brightness_score = 0.6
        else:
            brightness_score = 1.0

        # Enhanced weighted combination as per documentation
        quality = (
            0.50 * keypoints_score
            + 0.20 * sharpness_score
            + 0.15 * size_score
            + 0.10 * aspect_score
            + 0.05 * brightness_score
        )

        return round(min(1.0, max(0.0, quality)), 3)

    except Exception as e:
        logger.error(f"Enhanced crop quality assessment failed: {e}")
        return 0.0

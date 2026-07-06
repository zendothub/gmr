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


def assess_crop_quality(
    crop: np.ndarray,
    keypoint_visibility_ratio: float = None,
    yolo_confidence: float = None
) -> float:
    """
    Enhanced crop quality assessment with weighted formula prioritizing keypoints and YOLO confidence.
    
    Formula: keypoints(0.35) + yolo_conf(0.25) + sharpness(0.15) + size(0.10) + aspect(0.10) + brightness(0.05)
    
    If keypoints are None, we dynamically assign keypoints weight to YOLO confidence to prevent score capping.

    Args:
        crop: Person crop image (BGR)
        keypoint_visibility_ratio: Torso keypoint visibility ratio (0.0-1.0)
        yolo_confidence: YOLO person detection confidence (0.0-1.0)

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

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 1. Keypoints score & YOLO confidence score
        has_kp = keypoint_visibility_ratio is not None
        has_yolo = yolo_confidence is not None

        # Build dynamic weights to prevent score caps if some metrics are missing
        w_kp = 0.35
        w_yolo = 0.25
        w_sharp = 0.15
        w_size = 0.10
        w_aspect = 0.10
        w_bright = 0.05

        if not has_kp:
            # Redistribute keypoints weight (0.35) to YOLO confidence
            w_yolo += w_kp
            w_kp = 0.0
            keypoints_score = 0.0
        else:
            keypoints_score = keypoint_visibility_ratio

        if not has_yolo:
            # Re-normalize if YOLO confidence is missing (unlikely but safe)
            w_kp += w_yolo
            w_yolo = 0.0
            yolo_score = 0.0
        else:
            yolo_score = yolo_confidence

        # 2. Sharpness score - Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 500.0)

        # 3. Size score
        pixel_count = h * w
        size_score = min(1.0, pixel_count / (128 * 256))

        # 4. Aspect ratio score - person crops should be taller than wide
        aspect = h / w if w > 0 else 0
        if 1.5 <= aspect <= 3.5:
            aspect_score = 1.0
        elif 1.0 <= aspect <= 4.5:
            aspect_score = 0.7
        else:
            aspect_score = 0.4

        # 5. Brightness score
        mean_brightness = np.mean(gray)
        if mean_brightness < 30 or mean_brightness > 240:
            brightness_score = 0.3
        elif mean_brightness < 60 or mean_brightness > 200:
            brightness_score = 0.6
        else:
            brightness_score = 1.0

        # Weighted combination
        quality = (
            w_kp * keypoints_score
            + w_yolo * yolo_score
            + w_sharp * sharpness_score
            + w_size * size_score
            + w_aspect * aspect_score
            + w_bright * brightness_score
        )

        return round(min(1.0, max(0.0, quality)), 3)

    except Exception as e:
        logger.error(f"Enhanced crop quality assessment failed: {e}")
        return 0.0


def assess_face_quality(face_result) -> float:
    """
    Assess face quality based on face detection score and keypoint visibility.
    In InsightFace, kps array of shape (5, 2) lists:
    0: left_eye, 1: right_eye, 2: nose, 3: left_mouth_corner, 4: right_mouth_corner
    """
    try:
        if face_result is None:
            return 0.0
        
        score = face_result.face_score
        
        # Robust check: are both eyes detected and separated horizontally?
        if face_result.kps is not None and len(face_result.kps) >= 2:
            left_eye = face_result.kps[0]
            right_eye = face_result.kps[1]
            
            # Distance between eyes should be a positive, healthy number in horizontal plane
            eye_dist = abs(right_eye[0] - left_eye[0])
            if eye_dist > 5:
                # Up weight slightly if eyes are clearly visible and well-structured
                return round(min(1.0, score * 1.1), 3)
                
        return round(score, 3)
    except Exception as e:
        logger.debug(f"Face quality assessment failed: {e}")
        return 0.0

"""Upload API routes — image quality assessment for identity suitability."""

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from loguru import logger

from app.dependencies import get_current_user
from app.core.db.models.user import User
from app.config import get_settings
from app.modules.upload.schemas import (
    ImageQualityResponse,
    FaceQualityResult,
    CropQualityResult,
)

router = APIRouter(prefix="/api/v2/upload", tags=["Upload / Quality Check"])


@router.post("/check-quality", response_model=ImageQualityResponse)
async def check_image_quality(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload an image (JPEG/PNG) to check whether it is suitable for identity
    creation or body ReID.

    The endpoint runs:
      - **InsightFace** face detection + quality assessment (frontality, eye spread, det score)
      - **Crop quality** analysis (sharpness, size, aspect ratio, brightness)

    **Returns:**
    - `face.detected` — whether a valid face was found
    - `face.quality_score` — composite score (detection × frontality)
    - `face.frontality_score` — how frontal the face is (0=profile, 1=frontal)
    - `good_for_identity` — true if face quality ≥ `FACE_IDENTITY_MIN_SCORE` (0.60)
    - `good_for_body_reid` — true if body crop quality ≥ `REID_CROP_QUALITY_THRESHOLD` (0.30)
    - `reasons` — signals that passed their thresholds
    - `failures` — signals that failed (helpful for the UI to show what's wrong)
    - `thresholds` — all relevant thresholds from config for transparency
    """
    settings = get_settings()

    # ── Read & decode the uploaded image ──────────────────────────────
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h, w = img.shape[:2]
    if h < 50 or w < 25:
        raise HTTPException(status_code=400, detail="Image too small (min 50×25)")

    # ── Face analysis ─────────────────────────────────────────────────
    from app.modules.reid.insightface_analyzer import get_shared_analyzer

    analyzer = get_shared_analyzer()
    face_result = analyzer.analyze(img)

    reasons: list[str] = []
    failures: list[str] = []

    if face_result is not None and face_result.face_score >= settings.FACE_MIN_DET_SCORE:
        face_quality = face_result.face_quality
        frontality = face_result.frontality_score
        eye_spread = face_result.eye_spread
        det_score = face_result.face_score
        face_age = face_result.age

        face_detected = True
        reasons.append("face_detected")

        if face_quality >= settings.FACE_IDENTITY_MIN_SCORE:
            reasons.append("quality_above_threshold")
        else:
            failures.append(
                f"face_quality ({face_quality:.2f}) < FACE_IDENTITY_MIN_SCORE ({settings.FACE_IDENTITY_MIN_SCORE})"
            )

        if eye_spread >= settings.FACE_MIN_EYE_SPREAD:
            reasons.append("frontal_enough")
        else:
            failures.append(
                f"eye_spread ({eye_spread:.2f}) < FACE_MIN_EYE_SPREAD ({settings.FACE_MIN_EYE_SPREAD})"
            )

        if det_score >= settings.FACE_MIN_DET_SCORE:
            reasons.append("detection_confidence_ok")
        else:
            failures.append(
                f"det_score ({det_score:.2f}) < FACE_MIN_DET_SCORE ({settings.FACE_MIN_DET_SCORE})"
            )

    else:
        face_detected = False
        face_quality = 0.0
        frontality = 0.0
        eye_spread = 0.0
        det_score = 0.0
        face_age = None
        failures.append("no_face_detected")

    # ── Body crop quality ─────────────────────────────────────────────
    from app.modules.reid.crop_quality import assess_crop_quality

    body_quality = assess_crop_quality(img)
    good_for_body = body_quality >= settings.REID_CROP_QUALITY_THRESHOLD
    if good_for_body:
        reasons.append("body_crop_quality_ok")
    else:
        failures.append(
            f"body_crop_quality ({body_quality:.2f}) < REID_CROP_QUALITY_THRESHOLD ({settings.REID_CROP_QUALITY_THRESHOLD})"
        )

    # Build body-crop detail (re-use the per-component scores)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(1.0, laplacian_var / 500.0)
    pixel_count = h * w
    size_score = min(1.0, pixel_count / (128 * 256))
    aspect = h / w if w > 0 else 0
    if 1.5 <= aspect <= 3.5:
        aspect_score = 1.0
    elif 1.0 <= aspect <= 4.5:
        aspect_score = 0.7
    else:
        aspect_score = 0.4
    mean_brightness = np.mean(gray)
    if mean_brightness < 30 or mean_brightness > 240:
        brightness_score = 0.3
    elif mean_brightness < 60 or mean_brightness > 200:
        brightness_score = 0.6
    else:
        brightness_score = 1.0

    good_for_identity = (
        face_detected
        and face_quality >= settings.FACE_IDENTITY_MIN_SCORE
        and eye_spread >= settings.FACE_MIN_EYE_SPREAD
        and det_score >= settings.FACE_MIN_DET_SCORE
    )

    return ImageQualityResponse(
        face=FaceQualityResult(
            detected=face_detected,
            quality_score=round(face_quality, 3),
            frontality_score=round(frontality, 3),
            eye_spread=round(eye_spread, 3),
            det_score=round(det_score, 3),
            age=face_age,
        ),
        body_crop=CropQualityResult(
            quality_score=round(body_quality, 3),
            sharpness_score=round(sharpness_score, 3),
            size_score=round(size_score, 3),
            aspect_score=round(aspect_score, 3),
            brightness_score=round(brightness_score, 3),
        ),
        good_for_identity=good_for_identity,
        good_for_body_reid=good_for_body,
        reasons=reasons,
        failures=failures,
        thresholds={
            "FACE_IDENTITY_MIN_SCORE": settings.FACE_IDENTITY_MIN_SCORE,
            "FACE_MIN_DET_SCORE": settings.FACE_MIN_DET_SCORE,
            "FACE_MIN_EYE_SPREAD": settings.FACE_MIN_EYE_SPREAD,
            "REID_CROP_QUALITY_THRESHOLD": settings.REID_CROP_QUALITY_THRESHOLD,
            "FACE_FRONTALITY_WEIGHT": settings.FACE_FRONTALITY_WEIGHT,
        },
    )
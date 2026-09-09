"""Upload quality-check schemas."""
from typing import Optional, List
from pydantic import BaseModel


class ImageQualityReason(BaseModel):
    """A single signal that contributed to the quality decision."""
    name: str
    passed: bool
    value: Optional[float] = None
    threshold: Optional[float] = None


class FaceQualityResult(BaseModel):
    """Face analysis result."""
    detected: bool
    quality_score: float
    frontality_score: float
    eye_spread: float
    det_score: float
    age: Optional[int] = None


class CropQualityResult(BaseModel):
    """Body crop quality result (when a full-body image is provided)."""
    quality_score: float
    sharpness_score: float
    size_score: float
    aspect_score: float
    brightness_score: float


class ImageQualityResponse(BaseModel):
    """Response for the image quality check endpoint."""
    face: FaceQualityResult
    body_crop: Optional[CropQualityResult] = None
    good_for_identity: bool
    good_for_body_reid: bool
    reasons: List[str] = []
    failures: List[str] = []
    thresholds: dict
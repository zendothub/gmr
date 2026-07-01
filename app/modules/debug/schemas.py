"""Debug detection schemas."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


class PersonDebugResponse(BaseModel):
    """Single debug record response."""
    id: UUID
    camera_id: UUID
    camera_name: Optional[str] = None
    store_id: Optional[UUID] = None
    store_name: Optional[str] = None
    occurred_at: datetime
    
    # Outcome
    reid_attempted: bool
    reid_success: bool
    person_identity_id: Optional[UUID] = None
    
    # Track metrics
    bbox_height_px: Optional[float] = None
    bbox_width_px: Optional[float] = None
    detection_confidence: Optional[float] = None
    track_total_frames: Optional[int] = None
    
    # Crop
    crop_path: Optional[str] = None
    body_crop_path: Optional[str] = None
    
    # Quality
    quality_score: Optional[float] = None
    quality_passed: bool = False
    keypoint_visibility_ratio: Optional[float] = None
    keypoint_gate_passed: bool = False
    
    # Face
    face_detected: bool = False
    face_score: Optional[float] = None
    face_crop_path: Optional[str] = None
    face_age: Optional[int] = None
    face_gender: Optional[str] = None
    
    # ReID
    reid_score: Optional[float] = None
    reid_confident: bool = False
    reid_frame_count: int = 0
    
    # Failure
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    
    class Config:
        from_attributes = True


class DebugSummary(BaseModel):
    """Summary statistics for debug data."""
    total: int
    detected: int
    not_detected: int
    detection_rate: float
    failure_by_stage: dict


class DebugListResponse(BaseModel):
    """Paginated debug records with summary."""
    summary: DebugSummary
    records: List[PersonDebugResponse]
    total: int
    page: int
    limit: int

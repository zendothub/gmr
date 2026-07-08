"""Debug detection schemas."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


# --- Active Tracks Realtime Debug Schemas ---

class ActiveTrackDemographics(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    age_group: Optional[str] = None


class ActiveTrackResponse(BaseModel):
    camera_id: UUID
    camera_name: Optional[str] = None
    local_track_id: int
    track_session_id: Optional[UUID] = None
    person_identity_id: Optional[UUID] = None
    started_at: datetime
    last_seen_at: datetime
    age_seconds: float
    total_frames: int
    stability_score: float
    reid_attempted: bool
    reid_resolved: bool
    reid_confident: bool
    reid_score: float
    reid_frame_count: int
    best_crop_quality: float
    best_crop_path: Optional[str] = None
    current_crop_quality: float
    current_crop_path: Optional[str] = None
    face_crop_path: Optional[str] = None
    current_face_crop_path: Optional[str] = None
    current_face_score: float = 0.0
    identity_face_crop_path: Optional[str] = None
    face_score: Optional[float] = None
    demographics: Optional[ActiveTrackDemographics] = None


class ActiveTracksRealtimeResponse(BaseModel):
    total_active_tracks: int
    total_identified_tracks: int
    total_inactive_tracks: int
    active_tracks: List[ActiveTrackResponse]


# --- Paginated Unique Persons & Tracks Debug Schemas ---

class PersonTrackDetail(BaseModel):
    track_session_id: UUID
    camera_name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: float
    body_crop_path: Optional[str] = None
    face_crop_path: Optional[str] = None


class PaginatedTracksResponse(BaseModel):
    total_count: int
    page: int
    size: int
    tracks: List[PersonTrackDetail]


class UniquePersonItem(BaseModel):
    id: UUID
    gender: Optional[str] = None
    age_group: Optional[str] = None
    estimated_age: Optional[int] = None
    best_face_score: Optional[float] = None
    face_crop_path: Optional[str] = None
    first_seen_at: datetime
    last_seen_at: datetime
    visit_count: int
    total_tracks: int
    total_days: int
    is_staff: bool = False
    total_purchases: int = 0


class PaginatedUniquePersonsResponse(BaseModel):
    total_count: int
    page: int
    size: int
    persons: List[UniquePersonItem]

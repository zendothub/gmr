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


# --- Track Sessions Browser (assigned + unassigned) ---

class TrackSessionPersonSummary(BaseModel):
    """Lightweight person card for assigned track expand."""
    id: UUID
    gender: Optional[str] = None
    age_group: Optional[str] = None
    estimated_age: Optional[int] = None
    is_staff: bool = False
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    visit_count: int = 0
    total_tracks: int = 0
    total_days: int = 0
    face_crop_path: Optional[str] = None
    best_face_score: Optional[float] = None
    total_purchases: int = 0


class TrackSessionDebugItem(BaseModel):
    track_session_id: UUID
    camera_id: UUID
    camera_name: str
    local_track_id: int
    person_identity_id: Optional[UUID] = None
    started_at: datetime
    ended_at: Optional[datetime] = None
    last_seen_at: datetime
    duration_seconds: float
    total_frames: int
    is_active: bool
    gender: Optional[str] = None
    age_group: Optional[str] = None
    avg_confidence: Optional[float] = None
    stability_score: Optional[float] = None
    body_crop_path: Optional[str] = None
    # From bbox_history debug object; null/N/A if legacy array or missing
    best_crop_quality: Optional[float] = None
    torso_visibility_ratio: Optional[float] = None
    face_crop_path: Optional[str] = None
    best_face_score: Optional[float] = None
    has_billing: bool = False
    billing_count: int = 0
    person: Optional[TrackSessionPersonSummary] = None


class PaginatedTrackSessionsResponse(BaseModel):
    total_count: int
    page: int
    size: int
    tracks: List[TrackSessionDebugItem]


# --- Identity merge audit (dedup) ---

class IdentityMergeEventItem(BaseModel):
    id: UUID
    merged_at: datetime
    source: str
    job_run_id: Optional[UUID] = None
    job_run_at: Optional[datetime] = None
    winner_person_id: Optional[UUID] = None
    loser_person_id: UUID
    face_similarity: Optional[float] = None
    winner_face_score: Optional[float] = None
    loser_face_score: Optional[float] = None
    winner_first_seen_at: Optional[datetime] = None
    loser_first_seen_at: Optional[datetime] = None
    loser_visit_count: int = 0
    loser_track_count: int = 0
    winner_visit_count_before: Optional[int] = None
    winner_face_crop_path: Optional[str] = None
    loser_face_crop_path: Optional[str] = None
    winner_still_exists: bool = False
    winner_is_staff: Optional[bool] = None
    winner_gender: Optional[str] = None
    winner_visit_count_now: Optional[int] = None
    metadata_json: Optional[dict] = None


class PaginatedIdentityMergeEventsResponse(BaseModel):
    total_count: int
    page: int
    size: int
    items: List[IdentityMergeEventItem]


# --- Fragmented track / billing visit repair audit ---

class FragmentedTrackEventItem(BaseModel):
    id: UUID
    occurred_at: datetime
    event_type: str
    job_run_id: Optional[UUID] = None
    job_run_at: Optional[datetime] = None
    person_identity_id: Optional[UUID] = None
    camera_id: Optional[UUID] = None
    camera_name: Optional[str] = None
    zone_id: Optional[UUID] = None
    billing_interaction_id: Optional[UUID] = None
    primary_track_session_id: Optional[UUID] = None
    fragment_session_ids: Optional[List[str]] = None
    fragment_count: int = 1
    sum_dwell_seconds: Optional[float] = None
    dwell_threshold: Optional[float] = None
    stitch_gap_seconds: Optional[float] = None
    stitch_reason: Optional[str] = None
    body_median: Optional[float] = None
    face_max: Optional[float] = None
    person_is_staff: Optional[bool] = None
    person_gender: Optional[str] = None
    person_face_crop_path: Optional[str] = None
    metadata_json: Optional[dict] = None


class PaginatedFragmentedTrackEventsResponse(BaseModel):
    total_count: int
    page: int
    size: int
    items: List[FragmentedTrackEventItem]

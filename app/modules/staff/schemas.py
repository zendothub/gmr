"""Staff module Pydantic schemas."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


class StaffRegisterResponse(BaseModel):
    person_identity_id: UUID
    name: Optional[str] = None
    # "created" = new identity, "linked" = existing identity matched by face and marked staff
    action: str
    match_similarity: Optional[float] = None
    faces_stored: int
    faces_rejected: int
    # Why a photo was skipped, e.g. "image 2: duplicate of an already stored face"
    rejected: List[str] = []


class StaffResponse(BaseModel):
    person_identity_id: UUID
    name: Optional[str] = None
    is_staff: bool
    registered: bool
    face_count: int
    registration_face_count: int
    first_seen_at: datetime
    last_seen_at: datetime
    face_crop_path: Optional[str] = None


class StaffListResponse(BaseModel):
    items: List[StaffResponse]
    total: int


class ImageCheck(BaseModel):
    name: str
    # "pass" | "fail" — any fail means register rejects the photo
    status: str
    message: str


class ImageQualityReport(BaseModel):
    index: int
    filename: Optional[str] = None
    ok: bool
    face_count: int
    metrics: dict
    checks: List[ImageCheck]


class ExistingMatch(BaseModel):
    person_identity_id: UUID
    similarity: float
    name: Optional[str] = None
    is_staff: bool


class StaffImageCheckResponse(BaseModel):
    # True = every image passes the "fail" checks and all show the same person
    ok: bool
    images: List[ImageQualityReport]
    # None when fewer than 2 usable faces
    same_person: Optional[bool] = None
    # Registering these photos would link to this existing identity (action="linked")
    existing_match: Optional[ExistingMatch] = None

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

"""Events module Pydantic schemas."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


class EventResponse(BaseModel):
    id: UUID
    camera_id: UUID
    rule_id: Optional[UUID] = None
    zone_id: Optional[UUID] = None
    person_identity_id: Optional[UUID] = None
    track_session_id: Optional[UUID] = None
    event_type: str
    severity: str
    description: Optional[str] = None
    metadata_json: Optional[dict] = None
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    occurred_at: datetime
    is_acknowledged: bool
    is_false_positive: bool
    acknowledged_by: Optional[UUID] = None

    class Config:
        from_attributes = True


class EventListResponse(BaseModel):
    items: List[EventResponse]
    total: int
    page: int
    page_size: int
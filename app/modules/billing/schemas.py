"""Billing module Pydantic schemas."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


class BillingInteractionResponse(BaseModel):
    id: UUID
    camera_id: UUID
    person_identity_id: Optional[UUID] = None
    track_session_id: Optional[UUID] = None
    zone_id: Optional[UUID] = None
    entered_at: datetime
    exited_at: Optional[datetime] = None
    dwell_seconds: Optional[float] = None
    interaction_type: str
    metadata_json: Optional[dict] = None

    class Config:
        from_attributes = True


class BillingInteractionListResponse(BaseModel):
    items: List[BillingInteractionResponse]
    total: int
    page: int
    page_size: int
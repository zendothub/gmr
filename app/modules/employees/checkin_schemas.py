"""Pydantic schemas for employee check-in events."""

from datetime import datetime, date
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel


class CheckInResponse(BaseModel):
    """Single check-in event in the feed."""
    id: UUID
    employee_id: UUID
    employee_name: str
    emp_code: str
    checked_in_at: datetime
    check_in_date: date
    shift_slot_id: Optional[UUID] = None
    shift_label: Optional[str] = None
    camera_id: Optional[UUID] = None
    face_crop_path: Optional[str] = None
    face_crop_url: Optional[str] = None   # presigned MinIO URL — populated by the router
    status: str                           # "on_time" or "late"
    created_at: datetime

    model_config = {"from_attributes": True}


class CheckInListResponse(BaseModel):
    """Paginated list of check-in events (descending by checked_in_at)."""
    items: List[CheckInResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

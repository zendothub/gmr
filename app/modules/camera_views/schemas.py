"""Camera View Pydantic schemas."""

from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class CameraViewCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    view_type: str = "full_frame"
    polygon: Optional[Dict[str, Any]] = Field(
        None,
        description='Polygon JSON: {"points": [[x1,y1], [x2,y2], ...]}'
    )
    is_default: bool = False
    is_active: bool = True


class CameraViewUpdate(BaseModel):
    name: Optional[str] = None
    view_type: Optional[str] = None
    polygon: Optional[Dict[str, Any]] = None
    is_default: Optional[bool] = None
    is_active: Optional[bool] = None


class CameraViewResponse(BaseModel):
    id: UUID
    camera_id: UUID
    name: str
    view_type: str
    polygon: Optional[Dict[str, Any]]
    is_default: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

"""Zone Pydantic schemas."""

from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class ZoneCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    zone_type: str = Field(..., description="entry_line, exit_line, billing_zone, queue_zone, product_zone, ignore_zone, restricted_zone")
    shape: str = Field(default="polygon", description="polygon or line")
    polygon: Optional[Dict[str, Any]] = Field(None, description='{"points": [[x1,y1], [x2,y2], ...]}')
    line_config: Optional[Dict[str, Any]] = Field(None, description='{"start": [x1,y1], "end": [x2,y2]}')
    color: Optional[str] = "#FF0000"
    is_active: bool = True


class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    zone_type: Optional[str] = None
    shape: Optional[str] = None
    polygon: Optional[Dict[str, Any]] = None
    line_config: Optional[Dict[str, Any]] = None
    color: Optional[str] = None
    is_active: Optional[bool] = None


class ZoneResponse(BaseModel):
    id: UUID
    camera_view_id: UUID
    name: str
    zone_type: str
    shape: str
    polygon: Optional[Dict[str, Any]]
    line_config: Optional[Dict[str, Any]]
    color: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

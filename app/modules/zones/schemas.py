"""Zone Pydantic schemas.

A zone is bound on a specific camera's stream: the operator views the live
stream, selects polygon points, names the zone and picks a zone type. One
camera can have many zones.
"""

from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class ZoneCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    zone_type: str = Field(
        ...,
        description=(
            "entry_line, exit_line, billing_zone, queue_zone, product_zone, "
            "ignore_zone, restricted_zone, medicine_pickup_zone"
        ),
    )
    shape: str = Field(default="polygon", description="polygon or line")
    polygon: Optional[Dict[str, Any]] = Field(
        None, description='Selected polygon points, e.g. {"points": [[x1,y1], [x2,y2], ...]}'
    )
    is_active: bool = True


class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    zone_type: Optional[str] = None
    shape: Optional[str] = None
    polygon: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class ZoneResponse(BaseModel):
    id: UUID
    camera_id: Optional[UUID]
    name: str
    zone_type: str
    shape: str
    polygon: Optional[Dict[str, Any]]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

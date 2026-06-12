"""Rule Pydantic schemas."""

from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class RuleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    rule_type: str = Field(..., description="line_crossing, zone_dwell, billing_interaction, queue_count, possible_purchase, restricted_zone")
    zone_id: Optional[UUID] = None
    camera_id: Optional[UUID] = None
    config: Optional[Dict[str, Any]] = None
    cooldown_seconds: int = Field(default=30, ge=0)
    severity: str = "info"
    dwell_threshold_seconds: Optional[int] = None
    count_threshold: Optional[int] = None
    is_enabled: bool = True


class RuleUpdate(BaseModel):
    name: Optional[str] = None
    rule_type: Optional[str] = None
    zone_id: Optional[UUID] = None
    camera_id: Optional[UUID] = None
    config: Optional[Dict[str, Any]] = None
    cooldown_seconds: Optional[int] = None
    severity: Optional[str] = None
    dwell_threshold_seconds: Optional[int] = None
    count_threshold: Optional[int] = None
    is_enabled: Optional[bool] = None


class RuleResponse(BaseModel):
    id: UUID
    name: str
    rule_type: str
    zone_id: Optional[UUID]
    camera_id: Optional[UUID]
    config: Optional[Dict[str, Any]]
    cooldown_seconds: int
    severity: str
    dwell_threshold_seconds: Optional[int]
    count_threshold: Optional[int]
    is_enabled: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

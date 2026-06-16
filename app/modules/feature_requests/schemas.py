"""Feature Requests module Pydantic schemas."""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


class FeatureRequestCreate(BaseModel):
    """Payload to submit a new feature request from admin dashboard."""

    title: str = Field(..., min_length=1, max_length=500, description="Title of the feature request")
    description: str = Field(..., min_length=1, description="Detailed description of the feature")


class FeatureRequestUpdate(BaseModel):
    """Fields updatable by a developer after reviewing the request."""

    status: Optional[str] = Field(default=None, pattern=r"^(queued|in_progress|live)$")
    forecast_message: Optional[str] = Field(default=None, description="e.g. 'Will be live after 48 hours'")


class FeatureRequestResponse(BaseModel):
    """Public response schema for a single feature request."""

    id: UUID
    title: str
    description: str
    status: str
    forecast_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class FeatureRequestListResponse(BaseModel):
    """Paginated list of feature requests."""

    items: List[FeatureRequestResponse]
    total: int
    page: int
    page_size: int
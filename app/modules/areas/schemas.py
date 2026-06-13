"""Area Pydantic schemas - only a name is required."""

from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class AreaCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Proper name e.g. 'Main Entry'")


class AreaUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)


class AreaResponse(BaseModel):
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

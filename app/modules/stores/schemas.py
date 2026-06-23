"""Stores module Pydantic schemas."""

from typing import Optional
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, Field


class StoreCreate(BaseModel):
    """Schema for creating a new store."""
    name: str = Field(..., min_length=1, max_length=255)
    category: str = Field(..., min_length=1, max_length=100)
    status: str = Field(default="active", pattern="^(active|inactive)$")
    terminal: Optional[str] = Field(None, max_length=100)
    level: Optional[str] = Field(None, max_length=100)
    zone_gate: Optional[str] = Field(None, max_length=100, description="Physical gate/zone label e.g. 'Gate B4'")
    description: Optional[str] = Field(None)


class StoreUpdate(BaseModel):
    """Schema for updating an existing store."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    status: Optional[str] = Field(None, pattern="^(active|inactive)$")
    terminal: Optional[str] = Field(None, max_length=100)
    level: Optional[str] = Field(None, max_length=100)
    zone_gate: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# Lookup schemas — Category, Level, StoreZone
# ---------------------------------------------------------------------------

class StoreCategoryCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None)


class StoreCategoryUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None)


class StoreCategoryResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class StoreLevelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None)


class StoreLevelUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None)


class StoreLevelResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class StoreZoneCreate(BaseModel):
    """Physical airport zone/gate label — NOT a camera detection zone."""
    name: str = Field(..., min_length=1, max_length=100)
    terminal: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None)


class StoreZoneUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    terminal: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None)


class StoreZoneResponse(BaseModel):
    id: UUID
    name: str
    terminal: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class StoreResponse(BaseModel):
    """Schema for store responses."""
    id: UUID
    name: str
    category: str
    status: str
    terminal: Optional[str] = None
    level: Optional[str] = None
    zone_gate: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

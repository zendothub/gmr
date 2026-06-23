"""Users module Pydantic schemas."""

from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


class RoleResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """Schema for creating a user by an admin."""
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)
    status: str = Field(default="active", pattern="^(active|inactive)$")
    role: str = Field(default="VIEWER", pattern="^(SUPER_ADMIN|ADMIN|VIEWER)$")


class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[str] = Field(None, min_length=3, max_length=255)
    password: Optional[str] = Field(None, min_length=6, max_length=128)
    status: Optional[str] = Field(None, pattern="^(active|inactive)$")
    role: Optional[str] = Field(None, pattern="^(SUPER_ADMIN|ADMIN|VIEWER)$")


class UserDetailResponse(BaseModel):
    id: UUID
    name: str
    email: str
    status: str
    roles: List[RoleResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True
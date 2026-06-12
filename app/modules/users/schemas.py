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
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=128)
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_superuser: bool = False
    store_id: Optional[UUID] = None
    role_names: List[str] = Field(default_factory=list)


class UserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None
    store_id: Optional[UUID] = None
    password: Optional[str] = Field(None, min_length=6, max_length=128)


class UserDetailResponse(BaseModel):
    id: UUID
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool
    is_superuser: bool
    store_id: Optional[UUID] = None
    roles: List[RoleResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class RoleCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    description: Optional[str] = None
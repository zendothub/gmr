"""Auth Pydantic schemas."""

from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class UserSignup(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=128)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: UUID
    username: str


class RefreshTokenResponse(BaseModel):
    """Returned by the refresh endpoint: a freshly rotated token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"



class UserResponse(BaseModel):
    id: UUID
    username: str
    full_name: Optional[str] = None
    roles: List[str] = []

    @field_validator("roles", mode="before")
    @classmethod
    def extract_role_names(cls, v):
        """Convert Role ORM objects to plain role name strings."""
        if v and hasattr(v[0], "name"):
            return [r.name for r in v]
        return v or []

    class Config:
        from_attributes = True


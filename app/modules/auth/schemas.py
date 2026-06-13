"""Auth Pydantic schemas."""

from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


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
    is_superuser: bool

    class Config:
        from_attributes = True


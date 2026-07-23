"""Auth Pydantic schemas."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class UserLogin(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: UUID
    email: str
    session_count: int = 0


# ── Device session schemas ───────────────────────────────────────────

class DeviceSessionResponse(BaseModel):
    id: UUID
    device_hash: str
    device_label: str
    user_agent: str | None = None
    ip_address: str | None = None
    login_at: datetime
    last_active_at: datetime
    is_current_device: bool = False

    class Config:
        from_attributes = True


class SessionsListResponse(BaseModel):
    total_active_sessions: int
    sessions: list[DeviceSessionResponse]


class RevokeSessionResponse(BaseModel):
    message: str
class RefreshTokenResponse(BaseModel):
    """Returned by the refresh endpoint: a freshly rotated token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: UUID
    name: str
    email: str
    status: str
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
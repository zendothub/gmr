"""Users module Pydantic schemas."""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class UserInvite(BaseModel):
    """Schema for inviting (creating) a new user by Super Admin.

    Status is always set to 'active' automatically — not required from client.
    Role is limited to ADMIN or VIEWER; only one SUPER_ADMIN exists.
    """
    name: str = Field(..., min_length=1, max_length=255, description="Full name")
    email: str = Field(..., min_length=3, max_length=255, description="Email address")
    password: str = Field(..., min_length=6, max_length=128, description="Initial password")
    role: str = Field(
        ...,
        pattern="^(ADMIN|VIEWER)$",
        description="Role to assign: ADMIN or VIEWER",
    )


# Keep backward-compatible alias
UserCreate = UserInvite


class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[str] = Field(None, min_length=3, max_length=255)
    password: Optional[str] = Field(None, min_length=6, max_length=128)
    status: Optional[str] = Field(None, pattern="^(active|inactive)$")
    role: Optional[str] = Field(None, pattern="^(ADMIN|VIEWER)$")


class UserStatusUpdate(BaseModel):
    """Toggle a user's active/inactive status."""
    status: str = Field(..., pattern="^(active|inactive)$")


class UpdateProfile(BaseModel):
    """Authenticated user updates their own profile."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)


class UpdatePassword(BaseModel):
    """Authenticated user changes their own password."""
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6, max_length=128)
    confirm_new_password: str = Field(..., min_length=6, max_length=128)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class UserListItem(BaseModel):
    """Compact user record for list views (User Management table).

    Returns: name, email, status, role, password (plain) — no stores, no last_login.
    """
    id: UUID
    name: str
    email: str
    status: str
    role: str  # first role name as a plain string, e.g. "ADMIN"
    password: Optional[str] = None  # plain text password for admin visibility

    class Config:
        from_attributes = True


class UserDetailResponse(BaseModel):
    """Full user response (used for create / get-by-id)."""
    id: UUID
    name: str
    email: str
    status: str
    role: str  # first role name

    class Config:
        from_attributes = True

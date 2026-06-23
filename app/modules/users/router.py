"""Users API routes — User Management for Super Admin."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.users.schemas import (
    UserInvite,
    UserAdd,
    UserUpdate,
    UserStatusUpdate,
    UserDetailResponse,
    UserListItem,
    UpdateProfile,
    UpdatePassword,
)
from app.modules.users.service import UserService, _first_role_name

router = APIRouter(prefix="/api/users", tags=["Users"])


# ---------------------------------------------------------------------------
# Profile endpoints  (authenticated user manages their own account)
# ---------------------------------------------------------------------------

@router.put("/profile", response_model=UserDetailResponse)
async def update_profile(
    data: UpdateProfile,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update the authenticated user's profile details (name)."""
    user = await UserService.update_profile(db, current_user, data)
    return UserDetailResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        status=user.status.value if hasattr(user.status, "value") else user.status,
        role=_first_role_name(user),
    )


@router.put("/profile/password")
async def update_password(
    data: UpdatePassword,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Change the authenticated user's password.

    Body: `{ "current_password": "...", "new_password": "...", "confirm_new_password": "..." }`
    """
    return await UserService.update_password(db, current_user, data)


# ---------------------------------------------------------------------------
# User Management  (Super Admin — add, invite, list, toggle status, delete)
# ---------------------------------------------------------------------------

@router.post("/invite", response_model=UserDetailResponse, status_code=201)
async def invite_user(
    data: UserInvite,
    db: AsyncSession = Depends(get_db),
):
    """**Invite User** — Super Admin sets the password explicitly.

    - **name**: full name
    - **email**: email address
    - **password**: password chosen by admin (min 6 chars)
    - **role**: `ADMIN` or `VIEWER`

    No auth required. Status is automatically `active`.
    """
    user = await UserService.create_user(db, data)
    return UserDetailResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        status=user.status.value if hasattr(user.status, "value") else user.status,
        role=_first_role_name(user),
    )


@router.post("", response_model=UserDetailResponse, status_code=201)
async def add_user(
    data: UserAdd,
    db: AsyncSession = Depends(get_db),
):
    """**Add User** — password is auto-generated and stored for admin to view.

    - **name**: full name
    - **email**: email address
    - **role**: `ADMIN` or `VIEWER`

    No auth required. Status is automatically `active`.
    Password is auto-generated and returned in the list endpoint (`GET /api/users`).
    """
    user = await UserService.create_user_auto_password(db, data)
    return UserDetailResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        status=user.status.value if hasattr(user.status, "value") else user.status,
        role=_first_role_name(user),
    )


@router.get("", response_model=List[UserListItem])
async def list_users(
    role: Optional[str] = Query(
        None,
        pattern="^(ADMIN|VIEWER)$",
        description="Filter by role tab: ADMIN or VIEWER. Omit for all (excl. SUPER_ADMIN).",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List admins and viewers.

    - `GET /api/users` → all non-super-admin users
    - `GET /api/users?role=ADMIN` → admins tab
    - `GET /api/users?role=VIEWER` → viewers tab

    Returns: `id`, `name`, `email`, `status`, `role`
    """
    users = await UserService.get_users(db, role_filter=role)
    return [
        UserListItem(
            id=u.id,
            name=u.name,
            email=u.email,
            status=u.status.value if hasattr(u.status, "value") else u.status,
            role=_first_role_name(u),
            password=u.password_plain,  # plain text password set by admin
        )
        for u in users
    ]


@router.patch("/{user_id}/status", response_model=UserListItem)
async def toggle_user_status(
    user_id: UUID,
    data: UserStatusUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Activate or deactivate a user.

    Body: `{ "status": "active" }` or `{ "status": "inactive" }`
    """
    user = await UserService.update_user_status(db, user_id, data)
    return UserListItem(
        id=user.id,
        name=user.name,
        email=user.email,
        status=user.status.value if hasattr(user.status, "value") else user.status,
        role=_first_role_name(user),
    )


@router.get("/{user_id}", response_model=UserDetailResponse)
async def get_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single user by ID."""
    user = await UserService.get_user(db, user_id)
    return UserDetailResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        status=user.status.value if hasattr(user.status, "value") else user.status,
        role=_first_role_name(user),
    )


@router.put("/{user_id}", response_model=UserDetailResponse)
async def update_user(
    user_id: UUID,
    data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update a user's name, email, password, role or status."""
    user = await UserService.update_user(db, user_id, data)
    return UserDetailResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        status=user.status.value if hasattr(user.status, "value") else user.status,
        role=_first_role_name(user),
    )


@router.delete("/{user_id}")
async def delete_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a user permanently."""
    return await UserService.delete_user(db, user_id)

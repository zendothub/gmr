"""Users API routes (admin)."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, require_role
from app.core.db.models.user import User
from app.modules.users.schemas import (
    UserCreate,
    UserUpdate,
    UserDetailResponse,
    RoleCreate,
    RoleResponse,
)
from app.modules.users.service import UserService

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.post("", response_model=UserDetailResponse, status_code=201)
async def create_user(
    data: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Create a new user (admin only)."""
    user = await UserService.create_user(db, data)
    return UserDetailResponse.model_validate(user)


@router.get("", response_model=List[UserDetailResponse])
async def list_users(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """List all users (admin only)."""
    users = await UserService.get_users(db)
    return [UserDetailResponse.model_validate(u) for u in users]


@router.get("/roles", response_model=List[RoleResponse])
async def list_roles(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """List all roles (admin only)."""
    roles = await UserService.get_roles(db)
    return [RoleResponse.model_validate(r) for r in roles]


@router.post("/roles", response_model=RoleResponse, status_code=201)
async def create_role(
    data: RoleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Create a new role (admin only)."""
    role = await UserService.create_role(db, data)
    return RoleResponse.model_validate(role)


@router.get("/{user_id}", response_model=UserDetailResponse)
async def get_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Get a user by ID (admin only)."""
    user = await UserService.get_user(db, user_id)
    return UserDetailResponse.model_validate(user)


@router.put("/{user_id}", response_model=UserDetailResponse)
async def update_user(
    user_id: UUID,
    data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Update a user (admin only)."""
    user = await UserService.update_user(db, user_id, data)
    return UserDetailResponse.model_validate(user)


@router.delete("/{user_id}")
async def delete_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Delete a user (admin only)."""
    return await UserService.delete_user(db, user_id)


@router.post("/{user_id}/roles/{role_name}", response_model=UserDetailResponse)
async def assign_role(
    user_id: UUID,
    role_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Assign a role to a user (admin only)."""
    user = await UserService.assign_role(db, user_id, role_name)
    return UserDetailResponse.model_validate(user)

"""Users API routes."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.modules.users.schemas import (
    UserCreate,
    UserUpdate,
    UserDetailResponse,
)
from app.modules.users.service import UserService

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.post("", response_model=UserDetailResponse, status_code=201)
async def create_user(
    data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new user."""
    user = await UserService.create_user(db, data)
    return UserDetailResponse.model_validate(user)


@router.get("", response_model=List[UserDetailResponse])
async def list_users(
    db: AsyncSession = Depends(get_db),
):
    """List all users."""
    users = await UserService.get_users(db)
    return [UserDetailResponse.model_validate(u) for u in users]


@router.get("/{user_id}", response_model=UserDetailResponse)
async def get_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a user by ID."""
    user = await UserService.get_user(db, user_id)
    return UserDetailResponse.model_validate(user)


@router.put("/{user_id}", response_model=UserDetailResponse)
async def update_user(
    user_id: UUID,
    data: UserUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a user."""
    user = await UserService.update_user(db, user_id, data)
    return UserDetailResponse.model_validate(user)


@router.delete("/{user_id}")
async def delete_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a user."""
    return await UserService.delete_user(db, user_id)
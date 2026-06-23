"""Auth API routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.auth.schemas import (
    UserLogin,
    TokenResponse,
    RefreshTokenResponse,
    RefreshRequest,
    UserResponse,
)
from app.modules.auth.service import AuthService

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(data: UserLogin, db: AsyncSession = Depends(get_db)):
    """Authenticate with email and password to get access + refresh tokens."""
    return await AuthService.login(db, data)


@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh(data: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Rotate tokens: exchange a valid refresh token for a new access + refresh pair."""
    return await AuthService.refresh_access_token(db, data.refresh_token)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user profile."""
    return UserResponse.model_validate(current_user)
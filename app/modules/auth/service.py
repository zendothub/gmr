"""Auth service - handles login and token management."""

from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from fastapi import HTTPException, status
from loguru import logger

from jose import JWTError

from app.core.db.models.user import User
from app.utils.encryption import (
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.modules.auth.schemas import (
    UserLogin,
    TokenResponse,
    RefreshTokenResponse,
)


class AuthService:

    @staticmethod
    async def login(db: AsyncSession, data: UserLogin) -> TokenResponse:
        """Authenticate user by email and return access + refresh tokens."""
        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.email == data.email)
        )
        user = result.scalar_one_or_none()

        if not user or not verify_password(data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        if user.status.value != "active":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive",
            )

        access_token = create_access_token(data={"sub": str(user.id)})
        refresh_token = create_refresh_token(data={"sub": str(user.id)})
        logger.info(f"User logged in: {user.email}")
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=user.id,
            email=user.email,
        )

    @staticmethod
    async def refresh_access_token(db: AsyncSession, refresh_token: str) -> RefreshTokenResponse:
        """Exchange a valid refresh token for a freshly rotated token pair."""
        invalid = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = decode_token(refresh_token)
        except JWTError:
            raise invalid

        if payload.get("type") != "refresh":
            raise invalid

        user_id = payload.get("sub")
        if not user_id:
            raise invalid

        user = await AuthService.get_user_by_id(db, UUID(user_id))
        if user is None:
            raise invalid

        access_token = create_access_token(data={"sub": str(user.id)})
        new_refresh_token = create_refresh_token(data={"sub": str(user.id)})
        logger.info(f"Token pair refreshed: {user.email}")
        return RefreshTokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
        )

    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        result = await db.execute(
            select(User).options(selectinload(User.roles)).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
"""Auth service - handles signup, login, token creation."""

from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status
from loguru import logger

from jose import JWTError

from app.core.db.models.user import User
from app.utils.encryption import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.modules.auth.schemas import (
    UserSignup,
    UserLogin,
    TokenResponse,
    RefreshTokenResponse,
)



class AuthService:

    @staticmethod
    async def signup(db: AsyncSession, data: UserSignup) -> User:
        """Register a new user."""
        # Check if username exists
        result = await db.execute(select(User).where(User.username == data.username))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists",
            )

        user = User(
            username=data.username,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
            is_superuser=False,
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)
        logger.info(f"New user registered: {user.username} (id={user.id})")
        return user

    @staticmethod
    async def login(db: AsyncSession, data: UserLogin) -> TokenResponse:
        """Authenticate user and return both access and refresh tokens."""
        result = await db.execute(select(User).where(User.username == data.username))
        user = result.scalar_one_or_none()

        if not user or not verify_password(data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        access_token = create_access_token(data={"sub": str(user.id)})
        refresh_token = create_refresh_token(data={"sub": str(user.id)})
        logger.info(f"User logged in: {user.username}")
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=user.id,
            username=user.username,
        )

    @staticmethod
    async def refresh_access_token(db: AsyncSession, refresh_token: str) -> RefreshTokenResponse:
        """Exchange a valid refresh token for a freshly rotated token pair.

        Returns a NEW access token and a NEW refresh token (refresh-token
        rotation), so the client always moves forward with fresh credentials.
        """
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
        logger.info(f"Token pair refreshed: {user.username}")
        return RefreshTokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
        )


    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()


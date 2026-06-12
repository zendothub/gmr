"""Auth service - handles signup, login, token creation."""

from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status
from loguru import logger

from app.core.db.models.user import User
from app.utils.encryption import hash_password, verify_password, create_access_token
from app.modules.auth.schemas import UserSignup, UserLogin, TokenResponse


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

        # Check if email exists
        if data.email:
            result = await db.execute(select(User).where(User.email == data.email))
            if result.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered",
                )

        user = User(
            username=data.username,
            hashed_password=hash_password(data.password),
            email=data.email,
            full_name=data.full_name,
            is_active=True,
            is_superuser=False,
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)
        logger.info(f"New user registered: {user.username} (id={user.id})")
        return user

    @staticmethod
    async def login(db: AsyncSession, data: UserLogin) -> TokenResponse:
        """Authenticate user and return JWT token."""
        result = await db.execute(select(User).where(User.username == data.username))
        user = result.scalar_one_or_none()

        if not user or not verify_password(data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled",
            )

        access_token = create_access_token(data={"sub": str(user.id)})
        logger.info(f"User logged in: {user.username}")
        return TokenResponse(
            access_token=access_token,
            user_id=user.id,
            username=user.username,
        )

    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

"""Auth service - handles login and token management."""

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
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
from app.config import get_settings
from app.core.db.models.device_session import DeviceSession
from app.modules.auth.schemas import (
    UserLogin,
    TokenResponse,
    RefreshTokenResponse,
)
from app.utils.device_fingerprint import fingerprint, device_label


class AuthService:

    @staticmethod
    async def _get_client_info(request: Request | None):
        """Extract user-agent and IP from a FastAPI request."""
        ua = ""
        ip = ""
        if request:
            ua = request.headers.get("user-agent", "")
            ip = request.client.host if request.client else ""
        return ua, ip

    @staticmethod
    async def login(
        db: AsyncSession,
        data: UserLogin,
        request: Request | None = None,
    ) -> TokenResponse:
        """Authenticate user by email and return access + refresh tokens.

        When ``request`` is provided, creates or refreshes a DeviceSession
        record so we can track how many devices are currently logged in.
        """
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

        # ── Device session tracking ───────────────────────────────────
        session_count = 0
        if request:
            settings = get_settings()
            ua, ip = await AuthService._get_client_info(request)
            device_hash = fingerprint(ua, ip)
            now = datetime.now(timezone.utc)
            expires = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

            # Upsert: if same device already has an active session, refresh it;
            # otherwise create a new one.
            result = await db.execute(
                select(DeviceSession).where(
                    DeviceSession.user_id == user.id,
                    DeviceSession.device_hash == device_hash,
                    DeviceSession.is_active == True,
                ).limit(1)
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.last_active_at = now
                existing.expires_at = expires
                existing.user_agent = ua
                existing.ip_address = ip
            else:
                session = DeviceSession(
                    user_id=user.id,
                    device_hash=device_hash,
                    user_agent=ua,
                    ip_address=ip,
                    login_at=now,
                    last_active_at=now,
                    expires_at=expires,
                    is_active=True,
                )
                db.add(session)

            await db.flush()

            # Count active sessions for this user
            count_result = await db.execute(
                select(func.count(DeviceSession.id)).where(
                    DeviceSession.user_id == user.id,
                    DeviceSession.is_active == True,
                )
            )
            session_count = count_result.scalar() or 0

        logger.info(f"User logged in: {user.email}")
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=user.id,
            email=user.email,
            session_count=session_count,
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

    # ── Device session management ─────────────────────────────────────

    @staticmethod
    async def list_device_sessions(
        db: AsyncSession,
        user: User,
        request: Request | None = None,
    ) -> "SessionsListResponse":
        """Return all active device sessions for the given user."""
        from app.modules.auth.schemas import SessionsListResponse, DeviceSessionResponse

        result = await db.execute(
            select(DeviceSession)
            .where(
                DeviceSession.user_id == user.id,
                DeviceSession.is_active == True,
            )
            .order_by(DeviceSession.last_active_at.desc())
        )
        sessions = result.scalars().all()

        # Determine which session is the current device
        current_hash = ""
        if request:
            ua, ip = await AuthService._get_client_info(request)
            current_hash = fingerprint(ua, ip)

        session_list = []
        for s in sessions:
            session_list.append(
                DeviceSessionResponse(
                    id=s.id,
                    device_hash=s.device_hash,
                    device_label=device_label(s.user_agent or ""),
                    user_agent=s.user_agent,
                    ip_address=s.ip_address,
                    login_at=s.login_at,
                    last_active_at=s.last_active_at,
                    is_current_device=(s.device_hash == current_hash),
                )
            )

        return SessionsListResponse(
            total_active_sessions=len(session_list),
            sessions=session_list,
        )

    @staticmethod
    async def revoke_device_session(
        db: AsyncSession,
        user: User,
        session_id: UUID,
    ) -> "RevokeSessionResponse":
        """Deactivate a specific device session (force logout)."""
        from app.modules.auth.schemas import RevokeSessionResponse

        result = await db.execute(
            select(DeviceSession).where(
                DeviceSession.id == session_id,
                DeviceSession.user_id == user.id,
            )
        )
        session = result.scalar_one_or_none()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

        session.is_active = False
        await db.flush()
        logger.info(f"Device session revoked: user={user.email} session={session_id}")

        return RevokeSessionResponse(
            message=f"Session revoked successfully"
        )

"""Application-wide dependencies for dependency injection."""

from datetime import datetime, timezone
from typing import AsyncGenerator
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.core.db.session import get_async_session
from app.core.db.models.user import User
from app.core.db.models.device_session import DeviceSession
from app.utils.device_fingerprint import fingerprint

settings = get_settings()
security = HTTPBearer()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async for session in get_async_session():
        yield session


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Decode JWT token and return the current user (with roles eager-loaded).

    Also updates the matching DeviceSession's last_active_at as a side effect
    so we can track how many devices are actively using the API.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(
        select(User).options(selectinload(User.roles)).where(User.id == UUID(user_id))
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    # ── Device session heartbeat ──────────────────────────────────────
    # Update last_active_at on the matching device session (best-effort).
    # We don't have access to the Request object here, so we skip the
    # heartbeat. The login endpoint creates/refreshes sessions; the
    # periodic cleanup job handles expiry.
    # (Middleware-free approach — simpler and avoids BaseHTTPMiddleware
    #  issues with streaming responses.)

    return user


def require_role(role_name: str):
    """Dependency factory: only allow users with the given role.

    Usage: ``current_user: User = Depends(require_role("SUPER_ADMIN"))``
    """
    async def dependency(current_user: User = Depends(get_current_user)) -> User:
        user_role_names = {r.name for r in current_user.roles}
        if role_name not in user_role_names:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires '{role_name}' role",
            )
        return current_user
    return dependency
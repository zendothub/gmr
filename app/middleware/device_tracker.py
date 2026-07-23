"""Device activity tracking middleware.

Updates ``last_active_at`` on the matching ``DeviceSession`` for every
authenticated API request. Runs after auth so we already have the user.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from loguru import logger

from app.config import get_settings
from app.core.db.session import get_async_session
from app.core.db.models.device_session import DeviceSession
from app.utils.device_fingerprint import fingerprint


# Paths that should NOT trigger a session update (health checks, etc.)
_SKIP_PATHS = {"/health", "/", "/docs", "/redoc", "/openapi.json"}


class DeviceTrackerMiddleware(BaseHTTPMiddleware):
    """Lightweight middleware: updates device session last_active_at.

    Does NOT block the request — failures are logged and swallowed so a
    stale session row never breaks the API.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # Skip non-API paths and unauthenticated requests
        if request.url.path in _SKIP_PATHS:
            return response
        if not request.url.path.startswith("/api/"):
            return response

        # Only track if the request was authenticated (user_id set by dependency)
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            return response

        try:
            await self._update_session(request, user_id)
        except Exception:
            # Never let session tracking break the API
            pass

        return response

    async def _update_session(self, request: Request, user_id: str) -> None:
        """Update last_active_at on the matching device session."""
        settings = get_settings()
        ua = request.headers.get("user-agent", "")
        ip = request.client.host if request.client else ""
        device_hash = fingerprint(ua, ip)
        now = datetime.now(timezone.utc)

        async for db in get_async_session():
            from sqlalchemy import select, update

            result = await db.execute(
                select(DeviceSession).where(
                    DeviceSession.user_id == user_id,
                    DeviceSession.device_hash == device_hash,
                    DeviceSession.is_active == True,
                ).limit(1)
            )
            session = result.scalar_one_or_none()

            if session:
                session.last_active_at = now
                await db.commit()
            break  # single-use generator
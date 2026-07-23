"""DeviceSession model — tracks authenticated devices for a user account."""

import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class DeviceSession(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One row per authenticated device session.

    A device is identified by a SHA-256 hash of (User-Agent + /24 IP prefix).
    Sessions are created on login and updated on every authenticated request.
    Stale sessions are cleaned up periodically.
    """

    __tablename__ = "device_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    device_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    user_agent: Mapped[str] = mapped_column(
        String(512), nullable=True
    )
    ip_address: Mapped[str] = mapped_column(
        String(45), nullable=True  # IPv6 max length
    )
    login_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default="true", nullable=False
    )

    __table_args__ = (
        Index("ix_device_sessions_active_user", "is_active", "user_id"),
        Index("ix_device_sessions_active_expires", "is_active", "expires_at"),
    )
"""StreamViewerSession model — tracks devices actively watching a camera live feed."""

import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class StreamViewerSession(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One row per device actively watching a camera's live stream.

    Created when a device calls `POST /api/cameras/{id}/stream/start` and
    ended when it calls `/stop` or when the StreamManager idle reaper tears
    down the publisher.
    """

    __tablename__ = "stream_viewer_sessions"

    device_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("device_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    camera_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    device_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    ip_address: Mapped[str] = mapped_column(
        String(45), nullable=True
    )
    user_agent: Mapped[str] = mapped_column(
        String(512), nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index(
            "ix_stream_viewer_active",
            "camera_id",
            "ended_at",
        ),
        Index(
            "ix_stream_viewer_user_active",
            "user_id",
            "ended_at",
        ),
    )
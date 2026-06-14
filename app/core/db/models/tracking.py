"""Track session and observation models."""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class TrackSession(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "track_sessions"

    camera_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    local_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    person_identity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("person_identities.id"), nullable=True, index=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    bbox_history: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True, default=list)
    avg_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stability_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    # Demographics
    gender: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    age_group: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    best_crop_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    camera: Mapped["Camera"] = relationship("Camera", back_populates="track_sessions")
    person_identity: Mapped[Optional["PersonIdentity"]] = relationship(
        "PersonIdentity", back_populates="track_sessions"
    )
    observations: Mapped[List["TrackObservation"]] = relationship(
        "TrackObservation", back_populates="track_session", cascade="all, delete-orphan"
    )


class TrackObservation(Base, UUIDPrimaryKeyMixin):
    __tablename__ = "track_observations"

    track_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("track_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    bbox: Mapped[dict] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    zone_ids: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    track_session: Mapped["TrackSession"] = relationship("TrackSession", back_populates="observations")

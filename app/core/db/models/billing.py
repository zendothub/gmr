"""Billing interaction model."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class BillingInteraction(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "billing_interactions"

    camera_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    person_identity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("person_identities.id"), nullable=True, index=True
    )
    track_session_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("track_sessions.id"), nullable=True
    )
    zone_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id"), nullable=True
    )
    entered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    exited_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    dwell_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    interaction_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="billing_counter"
    )
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    camera: Mapped["Camera"] = relationship("Camera", back_populates="billing_interactions")
    person_identity: Mapped[Optional["PersonIdentity"]] = relationship("PersonIdentity")

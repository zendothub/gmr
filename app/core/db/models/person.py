"""Person identity and embedding models using pgvector."""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class PersonIdentity(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "person_identities"

    label: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    visit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    is_anonymous: Mapped[bool] = mapped_column(default=True, nullable=False)

    embeddings: Mapped[List["PersonEmbedding"]] = relationship(
        "PersonEmbedding", back_populates="person_identity", cascade="all, delete-orphan"
    )
    track_sessions: Mapped[List["TrackSession"]] = relationship(
        "TrackSession", back_populates="person_identity"
    )


class PersonEmbedding(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "person_embeddings"

    person_identity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("person_identities.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    embedding = mapped_column(Vector(512), nullable=False)
    camera_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id"), nullable=True
    )
    crop_quality: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    crop_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    person_identity: Mapped["PersonIdentity"] = relationship(
        "PersonIdentity", back_populates="embeddings"
    )

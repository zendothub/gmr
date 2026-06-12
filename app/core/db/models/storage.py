"""Storage objects model."""

import uuid
from datetime import datetime
from typing import Optional
import enum

from sqlalchemy import String, Integer, ForeignKey, DateTime, func, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class StorageType(str, enum.Enum):
    SNAPSHOT = "snapshot"
    CROP = "crop"
    CLIP = "clip"
    REPORT = "report"


class StorageObject(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "storage_objects"

    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_type: Mapped[str] = mapped_column(
        SAEnum(StorageType, name="storage_type_enum", create_constraint=True),
        nullable=False,
    )
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    camera_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id"), nullable=True
    )
    event_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("events.id"), nullable=True
    )
    person_identity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("person_identities.id"), nullable=True
    )
    captured_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

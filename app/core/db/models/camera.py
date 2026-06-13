"""Camera, CameraView, and Zone models."""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, Boolean, Integer, Float, ForeignKey, Text, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class CameraRole(str, enum.Enum):
    ENTRY_GATE = "entry_gate"
    BILLING_COUNTER = "billing_counter"
    QUEUE = "queue"
    PRODUCT_SHELF = "product_shelf"
    GENERAL = "general"


class CameraStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ViewType(str, enum.Enum):
    FULL_FRAME = "full_frame"
    ENTRY_GATE_VIEW = "entry_gate_view"
    BILLING_COUNTER_VIEW = "billing_counter_view"
    QUEUE_VIEW = "queue_view"
    PRODUCT_SHELF_VIEW = "product_shelf_view"
    IGNORE_AREA = "ignore_area"


class ZoneType(str, enum.Enum):
    ENTRY_LINE = "entry_line"
    EXIT_LINE = "exit_line"
    BILLING_ZONE = "billing_zone"
    QUEUE_ZONE = "queue_zone"
    PRODUCT_ZONE = "product_zone"
    IGNORE_ZONE = "ignore_zone"
    RESTRICTED_ZONE = "restricted_zone"


class ZoneShape(str, enum.Enum):
    POLYGON = "polygon"
    LINE = "line"


class Camera(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "cameras"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    rtsp_url: Mapped[str] = mapped_column(String(500), nullable=False)
    store_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("stores.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(
        SAEnum(CameraRole, name="camera_role_enum", create_constraint=True),
        nullable=False,
        default=CameraRole.GENERAL,
    )
    status: Mapped[str] = mapped_column(
        SAEnum(CameraStatus, name="camera_status_enum", create_constraint=True),
        nullable=False,
        default=CameraStatus.INACTIVE,
    )
    fps_target: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    resolution: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, default="1920x1080")
    detection_model: Mapped[str] = mapped_column(String(100), nullable=False, default="yolov8n")
    reid_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    demographic_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    frame_rotation: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # None, 90, 180, 270
    location_description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    store: Mapped["Store"] = relationship("Store", back_populates="cameras")
    views: Mapped[List["CameraView"]] = relationship("CameraView", back_populates="camera", cascade="all, delete-orphan")
    track_sessions: Mapped[List["TrackSession"]] = relationship("TrackSession", back_populates="camera")
    events: Mapped[List["Event"]] = relationship("Event", back_populates="camera")


class CameraView(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "camera_views"

    camera_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    view_type: Mapped[str] = mapped_column(
        SAEnum(ViewType, name="view_type_enum", create_constraint=True),
        nullable=False,
        default=ViewType.FULL_FRAME,
    )
    polygon: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    camera: Mapped["Camera"] = relationship("Camera", back_populates="views")
    zones: Mapped[List["Zone"]] = relationship("Zone", back_populates="camera_view", cascade="all, delete-orphan")


class Zone(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "zones"

    camera_view_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("camera_views.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    zone_type: Mapped[str] = mapped_column(
        SAEnum(ZoneType, name="zone_type_enum", create_constraint=True),
        nullable=False,
    )
    shape: Mapped[str] = mapped_column(
        SAEnum(ZoneShape, name="zone_shape_enum", create_constraint=True),
        nullable=False,
        default=ZoneShape.POLYGON,
    )
    polygon: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    line_config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    color: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, default="#FF0000")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    camera_view: Mapped["CameraView"] = relationship("CameraView", back_populates="zones")
    rules: Mapped[List["Rule"]] = relationship("Rule", back_populates="zone")

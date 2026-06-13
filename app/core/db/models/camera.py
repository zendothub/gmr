"""Camera and Zone models.

Cameras are statically mounted (no PTZ / moving), so there is no per-camera
ROI/"view" concept - detection runs on the full frame and is filtered by zones.

A camera has NO role/type of its own - the role belongs at the *zone* level
because one camera can cover multiple areas (entry + billing + exit can all be
visible in the same frame, each drawn as a separate zone with its own ZoneType).
"""


import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, Boolean, Integer, Float, ForeignKey, Text, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class CameraStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ZoneType(str, enum.Enum):

    ENTRY_LINE = "entry_line"
    EXIT_LINE = "exit_line"
    BILLING_ZONE = "billing_zone"
    QUEUE_ZONE = "queue_zone"
    PRODUCT_ZONE = "product_zone"
    IGNORE_ZONE = "ignore_zone"
    RESTRICTED_ZONE = "restricted_zone"
    # Where medicine is taken / dispensed (Apollo pharmacy pickup point)
    MEDICINE_PICKUP_ZONE = "medicine_pickup_zone"



class ZoneShape(str, enum.Enum):
    POLYGON = "polygon"
    LINE = "line"


class Camera(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "cameras"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Source RTSP the backend pulls from (camera/NVR).
    rtsp_url: Mapped[str] = mapped_column(String(500), nullable=False)
    # MediaMTX path the backend republishes into; the browser pulls the feed back
    # (WebRTC/HLS) from this path. Stable & deterministic per camera id.
    stream_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

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

    # Area is chosen from a dropdown when the camera is added (independent entity).
    area_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("areas.id", ondelete="SET NULL"), nullable=True
    )

    area: Mapped[Optional["Area"]] = relationship("Area", back_populates="cameras")

    track_sessions: Mapped[List["TrackSession"]] = relationship("TrackSession", back_populates="camera")

    events: Mapped[List["Event"]] = relationship("Event", back_populates="camera")

    # A camera can have MANY zones (each zone is bound on this camera's stream).
    zones: Mapped[List["Zone"]] = relationship(
        "Zone", back_populates="camera", cascade="all, delete-orphan"
    )



class Zone(Base, UUIDPrimaryKeyMixin, TimestampMixin):

    __tablename__ = "zones"

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
    # A zone is bound on a specific camera's stream (camera -> many zones).
    camera_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), nullable=True
    )
    # The selected polygon points drawn on the camera frame.
    polygon: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


    camera: Mapped[Optional["Camera"]] = relationship("Camera", back_populates="zones")
    rules: Mapped[List["Rule"]] = relationship("Rule", back_populates="zone")


"""Rule configuration model."""

import uuid
from typing import Optional
import enum

from sqlalchemy import String, Boolean, Integer, Float, ForeignKey, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class RuleType(str, enum.Enum):
    LINE_CROSSING = "line_crossing"
    ZONE_DWELL = "zone_dwell"
    BILLING_INTERACTION = "billing_interaction"
    QUEUE_COUNT = "queue_count"
    POSSIBLE_PURCHASE = "possible_purchase"
    RESTRICTED_ZONE = "restricted_zone"


class Rule(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "rules"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    rule_type: Mapped[str] = mapped_column(
        SAEnum(RuleType, name="rule_type_enum", create_constraint=True),
        nullable=False,
    )
    zone_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("zones.id", ondelete="SET NULL"), nullable=True
    )
    camera_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True, default=dict)
    cooldown_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="info")
    dwell_threshold_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    count_threshold: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    zone: Mapped[Optional["Zone"]] = relationship("Zone", back_populates="rules")
    camera: Mapped[Optional["Camera"]] = relationship("Camera")

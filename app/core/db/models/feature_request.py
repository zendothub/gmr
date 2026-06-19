"""FeatureRequest model - stores feature requests submitted from admin dashboard."""

import enum
from sqlalchemy import String, Text, Boolean, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.base import Base, UUIDPrimaryKeyMixin, TimestampMixin


class FeatureStatus(str, enum.Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    LIVE = "live"


class FeaturePriority(str, enum.Enum):
    LOW = "low"
    HIGH = "high"


class FeatureRequest(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "feature_requests"

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        SAEnum(
            FeatureStatus,
            name="feature_status_enum",
            values_callable=lambda obj: [e.value for e in obj],
            create_constraint=True,
        ),
        nullable=False,
        default=FeatureStatus.QUEUED,
    )
    forecast_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # New fields
    priority: Mapped[str] = mapped_column(
        SAEnum(
            FeaturePriority,
            name="feature_priority_enum",
            values_callable=lambda obj: [e.value for e in obj],
            create_constraint=True,
        ),
        nullable=False,
        default=FeaturePriority.LOW,
        server_default=FeaturePriority.LOW.value,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

"""Store model - retail outlets across the airport."""

import enum
from typing import Optional

from sqlalchemy import String, Enum, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class StoreStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class Store(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "stores"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[StoreStatus] = mapped_column(
        Enum(StoreStatus, name="store_status", create_type=True),
        default=StoreStatus.ACTIVE,
        server_default="active",
        nullable=False,
    )
    terminal: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    level: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # Zone/Gate is a physical location label (e.g. "Gate B4"), NOT related to camera zones
    zone_gate: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Aggregated counters — updated by analytics/billing jobs
    footfall_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    purchase_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    camera_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)

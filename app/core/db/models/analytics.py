"""Daily analytics summary model."""

import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import String, Integer, Float, ForeignKey, Date, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class DailyAnalyticsSummary(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "daily_analytics_summary"

    # Single-pharmacy deployment: analytics are aggregated globally (one row per day).
    summary_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)

    total_footfall: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unique_visitors: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    returning_visitors: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_dwell_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_queue_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    avg_queue_wait_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_billing_interactions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    zone_occupancy: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    hourly_footfall: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)


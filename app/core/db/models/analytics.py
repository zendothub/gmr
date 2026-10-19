"""Daily analytics summary model."""

import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, String, Integer, Float, ForeignKey, Date, DateTime, func
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


class DailyAnalyticsFrozen(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Immutable IST-day analytics snapshot. Dashboard past days read only this."""

    __tablename__ = "daily_analytics_frozen"

    day: Mapped[date] = mapped_column(Date, nullable=False, unique=True, index=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="scheduled")
    is_partial: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    unique_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unique_customers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unique_staff: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    night_only_persons: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    adjusted_unique: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    adjustment_factor: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    factor_model_version: Mapped[str] = mapped_column(String(64), nullable=False, default="none")
    adjusted_customers: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    customer_factor: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    purchases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    billing_interactions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_events: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tracks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_dwell_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    demographics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    hourly: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    per_camera: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    zone_counts: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    person_ids: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)


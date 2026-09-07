"""Employee attendance models: ShiftSlot, Employee, AttendanceRecord."""

import uuid
from datetime import datetime, time, date
from typing import Optional, List

from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, Date, Time, func, Boolean, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class AttendanceStatus(str, enum.Enum):
    present = "present"
    absent = "absent"
    late = "late"
    half_day = "half_day"


class ShiftSlot(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Predefined shift time windows (e.g. 11AM-7PM, 7PM-4AM)."""

    __tablename__ = "shift_slots"

    label: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    # Stored as Python time objects (HH:MM:SS)
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    end_time: Mapped[time] = mapped_column(Time, nullable=False)
    # True when end_time < start_time (crosses midnight e.g. 7PM → 4AM)
    crosses_midnight: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    employees: Mapped[List["Employee"]] = relationship(
        "Employee", back_populates="shift_slot"
    )
    attendance_records: Mapped[List["AttendanceRecord"]] = relationship(
        "AttendanceRecord", back_populates="shift_slot"
    )


class Employee(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Registered employee — links a business identity (emp_id/name/shift) to a
    face-based PersonIdentity from the AI pipeline."""

    __tablename__ = "employees"

    emp_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # FK to the AI pipeline's person_identities table (face + body embeddings).
    # May be null briefly during manual registration before face is confirmed.
    person_identity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("person_identities.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    shift_slot_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("shift_slots.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Latest face crop path stored in MinIO (updated on each registration refresh).
    face_crop_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    shift_slot: Mapped[Optional["ShiftSlot"]] = relationship(
        "ShiftSlot", back_populates="employees"
    )
    attendance_records: Mapped[List["AttendanceRecord"]] = relationship(
        "AttendanceRecord", back_populates="employee", cascade="all, delete-orphan"
    )


class AttendanceRecord(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """One row per employee per calendar date.

    Written/upserted by the camera worker whenever an employee's
    person_identity is recognised. Fully materialised — never recomputed from
    track_sessions — so exported sheets are permanently consistent even after
    identity-dedup merges.
    """

    __tablename__ = "attendance_records"

    employee_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("employees.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Calendar date the attendance row belongs to.
    # For night-shift employees the date is the *start* date of the shift
    # (e.g. 2026-09-07 for the 7PM-2026-09-07 → 4AM-2026-09-08 window).
    attendance_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Snapshot of the shift at recording time (kept even if employee later changes slot).
    shift_slot_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("shift_slots.id", ondelete="SET NULL"),
        nullable=True,
    )

    # First and last camera detection within the shift window.
    first_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Derived from first_seen_at / last_seen_at; stored for cheap reporting.
    total_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    status: Mapped[AttendanceStatus] = mapped_column(
        SAEnum(AttendanceStatus, name="attendance_status"),
        nullable=False,
        default=AttendanceStatus.absent,
    )

    # Which camera first/last detected the employee.
    check_in_camera_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )
    check_out_camera_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True
    )

    employee: Mapped["Employee"] = relationship(
        "Employee", back_populates="attendance_records"
    )
    shift_slot: Mapped[Optional["ShiftSlot"]] = relationship(
        "ShiftSlot", back_populates="attendance_records"
    )

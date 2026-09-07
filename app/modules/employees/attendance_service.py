"""Attendance upsert service — called from the camera worker.

When the camera pipeline identifies a person (assigns a person_identity_id to
a track session), this module checks if that identity belongs to a registered
employee and — if so — upserts an AttendanceRecord for the appropriate date.

Design constraints:
  • One row per (employee, attendance_date). UNIQUE(employee_id, attendance_date).
  • first_seen_at is set on first detection and NEVER overwritten.
  • last_seen_at is always updated to the latest detection time.
  • total_hours is recomputed on each update.
  • status: present if within shift window, late if first_seen_at > shift start.
  • Night-shift: attendance_date = shift START date (even if detection is after midnight).
  • Immune to identity-dedup merges — once written, timestamps don't move.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, date, timedelta, time
from typing import Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.attendance import (
    Employee,
    AttendanceRecord,
    AttendanceStatus,
    ShiftSlot,
)
from app.core.db.session import AsyncSessionLocal
from app.utils.time_utils import utc_now

# How many minutes after shift start before someone is marked "late".
LATE_THRESHOLD_MINUTES = 15


# ---------------------------------------------------------------------------
# Public entry point — called from camera_worker after identity is assigned
# ---------------------------------------------------------------------------

async def record_employee_detection(
    person_identity_id: UUID,
    camera_id: UUID,
    detected_at: datetime,
) -> None:
    """
    Async-safe upsert of an AttendanceRecord.

    Can be awaited directly inside the camera worker coroutine.
    Uses its own DB session to avoid interfering with the worker's session.
    """
    try:
        async with AsyncSessionLocal() as db:
            await _upsert_attendance(db, person_identity_id, camera_id, detected_at)
    except Exception as e:
        logger.warning(f"[Attendance] record_employee_detection failed: {e}")


# ---------------------------------------------------------------------------
# Internal upsert logic
# ---------------------------------------------------------------------------

async def _upsert_attendance(
    db: AsyncSession,
    person_identity_id: UUID,
    camera_id: UUID,
    detected_at: datetime,
) -> None:
    # 1. Find the employee linked to this person_identity
    employee: Optional[Employee] = (
        await db.execute(
            select(Employee).where(
                Employee.person_identity_id == person_identity_id,
                Employee.is_active.is_(True),
            )
        )
    ).scalar_one_or_none()

    if employee is None:
        return  # Not a registered employee, ignore

    # 2. Load shift slot
    shift: Optional[ShiftSlot] = None
    if employee.shift_slot_id:
        shift = (
            await db.execute(
                select(ShiftSlot).where(ShiftSlot.id == employee.shift_slot_id)
            )
        ).scalar_one_or_none()

    # 3. Compute the attendance_date
    attendance_date = _compute_attendance_date(detected_at, shift)

    # 4. Check if detection is within the shift window (if shift defined)
    if shift and not _is_within_shift(detected_at, shift, attendance_date):
        logger.debug(
            f"[Attendance] Detection for emp={employee.emp_id} at {detected_at} "
            f"is outside shift window '{shift.label}' — skipping."
        )
        return

    # 5. Fetch existing record
    existing: Optional[AttendanceRecord] = (
        await db.execute(
            select(AttendanceRecord).where(
                AttendanceRecord.employee_id == employee.id,
                AttendanceRecord.attendance_date == attendance_date,
            )
        )
    ).scalar_one_or_none()

    if existing is None:
        # ── First detection today — create record ────────────────────────
        status = _compute_status(detected_at, shift, attendance_date)
        record = AttendanceRecord(
            employee_id=employee.id,
            attendance_date=attendance_date,
            shift_slot_id=employee.shift_slot_id,
            first_seen_at=detected_at,
            last_seen_at=detected_at,
            total_hours=0.0,
            status=status,
            check_in_camera_id=camera_id,
            check_out_camera_id=camera_id,
        )
        db.add(record)
        logger.info(
            f"[Attendance] NEW record: emp={employee.emp_id} date={attendance_date} "
            f"first_seen={detected_at} status={status.value}"
        )
    else:
        # ── Update last_seen and recompute total_hours ───────────────────
        # first_seen_at is NEVER overwritten (immutable check-in time)
        if detected_at > (existing.last_seen_at or detected_at):
            existing.last_seen_at = detected_at
            existing.check_out_camera_id = camera_id

        if existing.first_seen_at and existing.last_seen_at:
            delta = (existing.last_seen_at - existing.first_seen_at).total_seconds()
            existing.total_hours = round(delta / 3600, 4)

        logger.debug(
            f"[Attendance] UPDATED: emp={employee.emp_id} date={attendance_date} "
            f"last_seen={detected_at} total_hours={existing.total_hours}"
        )

    await db.commit()


# ---------------------------------------------------------------------------
# Shift-window helpers
# ---------------------------------------------------------------------------

def _compute_attendance_date(detected_at: datetime, shift: Optional[ShiftSlot]) -> date:
    """
    Determine which calendar date this detection belongs to.

    For day shifts: it's simply the date of `detected_at`.
    For night shifts (crosses_midnight=True, e.g. 7PM→4AM):
      - If the detection time is BEFORE the shift end (e.g. 2AM), it belongs to
        the PREVIOUS day's shift date.
      - If the detection time is AFTER or AT the shift start (e.g. 7:30PM), it
        belongs to TODAY's shift date.
    """
    if shift is None or not shift.crosses_midnight:
        return detected_at.date()

    # Night shift: end_time < start_time (e.g. 19:00 → 04:00)
    t = detected_at.time().replace(tzinfo=None)
    if t < shift.end_time:
        # Early morning portion (e.g. 01:00 → belongs to previous day's shift)
        return (detected_at - timedelta(days=1)).date()
    return detected_at.date()


def _is_within_shift(detected_at: datetime, shift: ShiftSlot, attendance_date: date) -> bool:
    """Return True if `detected_at` falls inside the shift window."""
    tz = detected_at.tzinfo

    if not shift.crosses_midnight:
        # Normal day shift
        shift_start = datetime.combine(attendance_date, shift.start_time, tzinfo=tz) if tz else \
                      datetime.combine(attendance_date, shift.start_time)
        shift_end = datetime.combine(attendance_date, shift.end_time, tzinfo=tz) if tz else \
                    datetime.combine(attendance_date, shift.end_time)
        return shift_start <= detected_at <= shift_end
    else:
        # Night shift spanning two dates
        shift_start = datetime.combine(attendance_date, shift.start_time, tzinfo=tz) if tz else \
                      datetime.combine(attendance_date, shift.start_time)
        next_day = attendance_date + timedelta(days=1)
        shift_end = datetime.combine(next_day, shift.end_time, tzinfo=tz) if tz else \
                    datetime.combine(next_day, shift.end_time)
        return shift_start <= detected_at <= shift_end


def _compute_status(
    first_seen_at: datetime,
    shift: Optional[ShiftSlot],
    attendance_date: date,
) -> AttendanceStatus:
    """Determine present vs late for a new attendance record."""
    if shift is None:
        return AttendanceStatus.present

    tz = first_seen_at.tzinfo
    shift_start = datetime.combine(attendance_date, shift.start_time, tzinfo=tz) if tz else \
                  datetime.combine(attendance_date, shift.start_time)

    late_cutoff = shift_start + timedelta(minutes=LATE_THRESHOLD_MINUTES)
    if first_seen_at > late_cutoff:
        return AttendanceStatus.late

    return AttendanceStatus.present

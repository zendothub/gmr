"""Backfill today's attendance records that were missed due to the UTC vs IST timezone bug.

Bug (now fixed in attendance_service.py):
  - detected_at was UTC but shift start/end times are IST.
  - _is_within_shift compared UTC 06:15 against IST 09:00 → always False → no record created.

This script:
  1. Finds all active employees with a shift slot assigned.
  2. For today's date, checks if they have attendance_records.
  3. If missing, looks up their track_sessions from today (in IST timezone) and
     creates attendance_records with the correct IST-derived attendance_date.

Usage:
    cd /path/to/project
    python -m danger.backfill_todays_attendance              # backfill today
    python -m danger.backfill_todays_attendance --date=2026-09-09  # backfill specific date
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, date, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc


async def backfill():
    import argparse

    parser = argparse.ArgumentParser(description="Backfill attendance for a specific date.")
    parser.add_argument("--date", type=str, default=None, help="Date to backfill (YYYY-MM-DD). Defaults to today in IST.")
    args = parser.parse_args()

    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        target_date = datetime.now(IST).date()

    print(f"[Backfill] Target date (IST): {target_date}")

    from app.core.db.session import AsyncSessionLocal
    from sqlalchemy import select
    from app.core.db.models.attendance import Employee, AttendanceRecord, AttendanceStatus, ShiftSlot
    from app.core.db.models.tracking import TrackSession

    async with AsyncSessionLocal() as db:
        # 1. Load all active employees with shift slots
        employees = (
            await db.execute(
                select(Employee)
                .where(Employee.is_active.is_(True))
                .where(Employee.shift_slot_id.isnot(None))
            )
        ).scalars().all()

        print(f"[Backfill] Found {len(employees)} active employees with shifts.")

        # 2. Load all shift slots
        slot_ids = {e.shift_slot_id for e in employees}
        slots = (
            await db.execute(select(ShiftSlot).where(ShiftSlot.id.in_(slot_ids)))
        ).scalars().all()
        slot_map = {s.id: s for s in slots}

        # 3. Build IST day boundaries for the target date
        day_start_ist = datetime.combine(target_date, datetime.min.time(), tzinfo=IST)
        day_end_ist = datetime.combine(target_date, datetime.max.time(), tzinfo=IST)
        # Convert to UTC for DB query
        day_start_utc = day_start_ist.astimezone(UTC)
        day_end_utc = day_end_ist.astimezone(UTC)

        created = 0
        updated = 0
        skipped_no_sessions = 0
        skipped_outside_shift = 0

        for emp in employees:
            shift = slot_map.get(emp.shift_slot_id)
            if shift is None:
                print(f"  [Skip] emp={emp.emp_id}: no shift slot found")
                continue

            # Check if attendance record already exists for this date
            existing = (
                await db.execute(
                    select(AttendanceRecord).where(
                        AttendanceRecord.employee_id == emp.id,
                        AttendanceRecord.attendance_date == target_date,
                    )
                )
            ).scalar_one_or_none()

            if existing:
                print(f"  [Exists] emp={emp.emp_id}: already has attendance (status={existing.status.value})")
                continue

            # Look for track sessions for this employee's person_identity within the day range
            if emp.person_identity_id is None:
                print(f"  [Skip] emp={emp.emp_id}: no person_identity linked")
                skipped_no_sessions += 1
                continue

            sessions = (
                await db.execute(
                    select(TrackSession).where(
                        TrackSession.person_identity_id == emp.person_identity_id,
                        TrackSession.started_at >= day_start_utc,
                        TrackSession.started_at <= day_end_utc,
                    ).order_by(TrackSession.started_at.asc())
                )
            ).scalars().all()

            if not sessions:
                print(f"  [Skip] emp={emp.emp_id}: no track sessions found in IST day {target_date}")
                skipped_no_sessions += 1
                continue

            # Find first and last seen timestamps
            first_seen_utc = min(s.started_at for s in sessions)
            last_seen_utc = max(s.last_seen_at or s.started_at for s in sessions)

            # Convert to IST for shift window check
            first_seen_ist = first_seen_utc.astimezone(IST) if first_seen_utc.tzinfo else first_seen_utc.replace(tzinfo=UTC).astimezone(IST)
            last_seen_ist = last_seen_utc.astimezone(IST) if last_seen_utc.tzinfo else last_seen_utc.replace(tzinfo=UTC).astimezone(IST)

            # Check if within shift window
            from app.modules.employees.attendance_service import _is_within_shift, _compute_status
            if not _is_within_shift(first_seen_ist, shift, target_date):
                print(f"  [Skip] emp={emp.emp_id}: first_seen_ist={first_seen_ist.time()} outside shift {shift.start_time}-{shift.end_time}")
                skipped_outside_shift += 1
                continue

            # Determine status
            status = _compute_status(first_seen_ist, shift, target_date)

            # Get camera_id from first session
            camera_id = sessions[0].camera_id

            # Calculate total hours
            delta = (last_seen_utc - first_seen_utc).total_seconds()
            total_hours = round(delta / 3600, 4)

            record = AttendanceRecord(
                employee_id=emp.id,
                attendance_date=target_date,
                shift_slot_id=emp.shift_slot_id,
                first_seen_at=first_seen_utc,
                last_seen_at=last_seen_utc,
                total_hours=total_hours,
                status=status,
                check_in_camera_id=camera_id,
                check_out_camera_id=camera_id,
            )
            db.add(record)
            created += 1
            print(f"  [Created] emp={emp.emp_id}: status={status.value} first={first_seen_ist} last={last_seen_ist} hours={total_hours}")

        await db.commit()
        print(f"\n[Backfill] Complete: {created} created, {updated} updated, {skipped_no_sessions} no-sessions, {skipped_outside_shift} outside-shift")


if __name__ == "__main__":
    asyncio.run(backfill())
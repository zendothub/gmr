#!/usr/bin/env python3
"""Backfill employee_checkins from existing attendance_records.

For every attendance_record with status present/late that does NOT already
have a corresponding employee_checkins row, create one.

Usage:
    cd /path/to/retail-ai-platform
    .venv/bin/python danger/backfill_checkins_from_attendance.py

Safe to run multiple times — skips rows that already exist (unique constraint
on employee_id + check_in_date).
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, and_, not_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.db.session import AsyncSessionLocal
from app.core.db.models.attendance import (
    AttendanceRecord,
    AttendanceStatus,
    Employee,
    EmployeeCheckIn,
    ShiftSlot,
)


async def backfill():
    async with AsyncSessionLocal() as db:
        # Find all attendance records with present/late status
        result = await db.execute(
            select(AttendanceRecord)
            .where(
                AttendanceRecord.status.in_([
                    AttendanceStatus.present,
                    AttendanceStatus.late,
                ])
            )
            .where(AttendanceRecord.first_seen_at.isnot(None))
            .order_by(AttendanceRecord.attendance_date.asc())
        )
        records = result.scalars().all()
        print(f"Found {len(records)} attendance records with present/late status")

        # Cache employees and shift slots
        emp_cache: dict = {}
        slot_cache: dict = {}

        created = 0
        skipped = 0

        for rec in records:
            # Check if checkin already exists
            existing = (
                await db.execute(
                    select(EmployeeCheckIn).where(
                        EmployeeCheckIn.employee_id == rec.employee_id,
                        EmployeeCheckIn.check_in_date == rec.attendance_date,
                    )
                )
            ).scalar_one_or_none()

            if existing:
                skipped += 1
                continue

            # Load employee if not cached
            if rec.employee_id not in emp_cache:
                emp = (
                    await db.execute(
                        select(Employee).where(Employee.id == rec.employee_id)
                    )
                ).scalar_one_or_none()
                emp_cache[rec.employee_id] = emp
            emp = emp_cache[rec.employee_id]

            if emp is None:
                skipped += 1
                continue

            # Load shift slot if not cached
            shift_label = None
            if rec.shift_slot_id:
                if rec.shift_slot_id not in slot_cache:
                    slot = (
                        await db.execute(
                            select(ShiftSlot).where(ShiftSlot.id == rec.shift_slot_id)
                        )
                    ).scalar_one_or_none()
                    slot_cache[rec.shift_slot_id] = slot
                slot = slot_cache[rec.shift_slot_id]
                shift_label = slot.label if slot else None

            checkin_status = "late" if rec.status == AttendanceStatus.late else "on_time"

            checkin = EmployeeCheckIn(
                employee_id=rec.employee_id,
                employee_name=emp.name,
                emp_code=emp.emp_id,
                checked_in_at=rec.first_seen_at,
                check_in_date=rec.attendance_date,
                shift_slot_id=rec.shift_slot_id,
                shift_label=shift_label,
                camera_id=rec.check_in_camera_id,
                face_crop_path=emp.face_crop_path,
                status=checkin_status,
            )
            db.add(checkin)
            created += 1

            # Batch commit every 100 rows
            if created % 100 == 0:
                await db.commit()
                print(f"  ... committed {created} check-in events so far")

        await db.commit()
        print(f"\nDone! Created: {created}, Skipped (already exist): {skipped}")


if __name__ == "__main__":
    asyncio.run(backfill())

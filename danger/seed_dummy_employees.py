#!/usr/bin/env python3
"""Seed dummy employees with full data chain.

Creates:
  - PersonIdentity + PersonFaceEmbedding (with random face images from randomuser.me → MinIO)
  - Employees linked to PersonIdentity (random shift slot from DB)
  - AttendanceRecords for past 30 days (mix of present/late/absent)
  - LeaveRequests for employees on leave days
  - EmployeeCheckIn events for present/late days

Usage:
    cd /path/to/retail-ai-platform
    .venv/bin/python danger/seed_dummy_employees.py
"""

import asyncio
import io
import os
import random
import sys
import uuid
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import requests
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.session import AsyncSessionLocal
from app.core.db.models.attendance import (
    AttendanceRecord,
    AttendanceStatus,
    Employee,
    EmployeeCheckIn,
    Gender,
    ShiftSlot,
)
from app.core.db.models.leave import LeaveRequest, LeaveType
from app.core.db.models.person import PersonIdentity, PersonFaceEmbedding

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

# ---------------------------------------------------------------------------
# Dummy employee data
# ---------------------------------------------------------------------------
DUMMY_EMPLOYEES = [
    {"emp_id": "EMP001", "name": "Rahul Sharma", "gender": "MALE"},
    {"emp_id": "EMP002", "name": "Priya Patel", "gender": "FEMALE"},
    {"emp_id": "EMP003", "name": "Amit Kumar", "gender": "MALE"},
    {"emp_id": "EMP004", "name": "Sneha Gupta", "gender": "FEMALE"},
    {"emp_id": "EMP005", "name": "Vikram Singh", "gender": "MALE"},
    {"emp_id": "EMP006", "name": "Neha Verma", "gender": "FEMALE"},
    {"emp_id": "EMP007", "name": "Rohit Jain", "gender": "MALE"},
    {"emp_id": "EMP008", "name": "Kavita Mishra", "gender": "FEMALE"},
    {"emp_id": "EMP009", "name": "Suresh Yadav", "gender": "MALE"},
    {"emp_id": "EMP010", "name": "Anjali Reddy", "gender": "FEMALE"},
    {"emp_id": "EMP011", "name": "Deepak Tiwari", "gender": "MALE"},
    {"emp_id": "EMP012", "name": "Pooja Nair", "gender": "FEMALE"},
    {"emp_id": "EMP013", "name": "Manoj Pandey", "gender": "MALE"},
    {"emp_id": "EMP014", "name": "Ritu Chauhan", "gender": "FEMALE"},
    {"emp_id": "EMP015", "name": "Karan Mehta", "gender": "MALE"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_random_face(gender: str) -> bytes | None:
    """Download a random face image from randomuser.me."""
    try:
        g = "male" if gender == "MALE" else "female"
        resp = requests.get(
            f"https://randomuser.me/api/?gender={g}&inc=picture&noinfo",
            timeout=10,
        )
        data = resp.json()
        pic_url = data["results"][0]["picture"]["large"]
        img_resp = requests.get(pic_url, timeout=10)
        if img_resp.status_code == 200:
            return img_resp.content
    except Exception as e:
        logger.warning(f"Failed to download face image: {e}")
    return None


def _upload_to_minio(image_bytes: bytes, prefix: str) -> str | None:
    """Upload image bytes to MinIO, return the object path."""
    try:
        from app.modules.storage.minio_client import get_client, BUCKET_PREFIX

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        uid = uuid.uuid4().hex[:8]
        object_name = f"crops/{prefix}_{ts}_{uid}.jpg"

        client = get_client()
        client.put_object(
            BUCKET_PREFIX,
            object_name,
            io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="image/jpeg",
        )
        return object_name
    except Exception as e:
        logger.warning(f"MinIO upload failed: {e}")
        return None


def _random_embedding() -> list:
    """Generate a random 512-dim unit vector (dummy face embedding)."""
    vec = np.random.randn(512).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def _random_checkin_time(shift: ShiftSlot, att_date: date, is_late: bool) -> datetime:
    """Generate a plausible check-in time for a given shift and date."""
    base_dt = datetime.combine(att_date, shift.start_time, tzinfo=IST)

    if is_late:
        # 16-60 minutes late
        offset = timedelta(minutes=random.randint(16, 60))
    else:
        # 0-15 minutes early to on-time
        offset = timedelta(minutes=random.randint(-15, 10))

    return (base_dt + offset).astimezone(UTC)


def _random_checkout_time(checkin: datetime, shift: ShiftSlot) -> datetime:
    """Generate a plausible check-out time (4-9 hours after check-in)."""
    hours = random.uniform(4.0, 9.0)
    return checkin + timedelta(hours=hours)


# ---------------------------------------------------------------------------
# Main seed
# ---------------------------------------------------------------------------

async def seed():
    async with AsyncSessionLocal() as db:
        # 1. Load existing shift slots and camera
        slots = (await db.execute(
            select(ShiftSlot).where(ShiftSlot.is_active.is_(True))
        )).scalars().all()
        if not slots:
            print("ERROR: No active shift_slots found. Create some first.")
            return

        # Get a camera ID (use first available, or None)
        from app.core.db.models.camera import Camera
        camera = (await db.execute(select(Camera).limit(1))).scalar_one_or_none()
        camera_id = camera.id if camera else None

        print(f"Found {len(slots)} shift slots, camera_id={camera_id}")
        print(f"Shift slots: {[f'{s.label} ({s.start_time}-{s.end_time})' for s in slots]}")
        print()

        # Check for existing dummy employees (skip if already seeded)
        existing = (await db.execute(
            select(Employee).where(Employee.emp_id.in_([e["emp_id"] for e in DUMMY_EMPLOYEES]))
        )).scalars().all()
        if existing:
            print(f"WARNING: {len(existing)} dummy employees already exist. Skipping creation.")
            print("  To re-seed, delete them first.")
            return

        created_employees = []
        today = date.today()
        start_date = today - timedelta(days=30)

        # 2. Create each dummy employee
        for i, emp_data in enumerate(DUMMY_EMPLOYEES):
            print(f"\n--- Creating {emp_data['name']} ({emp_data['emp_id']}) ---")

            # Download face image
            img_bytes = _download_random_face(emp_data["gender"])
            face_crop_path = None
            if img_bytes:
                face_crop_path = _upload_to_minio(img_bytes, "emp_face")
                if face_crop_path:
                    print(f"  ✓ Face image uploaded: {face_crop_path}")
                else:
                    print(f"  ✗ Face upload failed (will continue without image)")
            else:
                print(f"  ✗ Face download failed (will continue without image)")

            # Create PersonIdentity
            person = PersonIdentity(
                label=emp_data["name"],
                is_anonymous=False,
                face_crop_path=face_crop_path,
                best_face_score=round(random.uniform(0.75, 0.98), 2),
                gender=emp_data["gender"],
                estimated_age=random.randint(22, 45),
                first_seen_at=datetime.combine(start_date, time(10, 0), tzinfo=UTC),
                last_seen_at=datetime.combine(today, time(10, 0), tzinfo=UTC),
            )
            db.add(person)
            await db.flush()  # get person.id
            print(f"  ✓ PersonIdentity created: {person.id}")

            # Create PersonFaceEmbedding (dummy 512-dim vector)
            face_emb = PersonFaceEmbedding(
                person_identity_id=person.id,
                embedding=_random_embedding(),
                face_score=round(random.uniform(0.70, 0.95), 2),
                face_crop_path=face_crop_path,
                captured_at=datetime.combine(start_date, time(10, 0), tzinfo=UTC),
            )
            db.add(face_emb)

            # Create Employee
            shift = random.choice(slots)
            employee = Employee(
                emp_id=emp_data["emp_id"],
                name=emp_data["name"],
                gender=Gender(emp_data["gender"]),
                weekends=["SATURDAY", "SUNDAY"],
                person_identity_id=person.id,
                shift_slot_id=shift.id,
                face_crop_path=face_crop_path,
            )
            db.add(employee)
            await db.flush()
            print(f"  ✓ Employee created: {employee.id} (shift: {shift.label})")

            created_employees.append((employee, shift, face_crop_path))

        await db.commit()
        print(f"\n=== Created {len(created_employees)} employees ===\n")

        # 3. Generate attendance, leaves, and check-ins for 30 days
        total_attendance = 0
        total_leaves = 0
        total_checkins = 0

        for employee, shift, face_crop_path in created_employees:
            # Decide leave days (2-5 random days in the 30-day period)
            num_leave_days = random.randint(2, 5)
            all_days = [start_date + timedelta(days=d) for d in range(31)]
            # Only pick working days (not weekends) for leave
            working_days = [d for d in all_days if d.strftime("%A").upper() not in ("SATURDAY", "SUNDAY")]
            leave_days = set(random.sample(working_days, min(num_leave_days, len(working_days))))

            # Create leave requests (group consecutive leave days)
            sorted_leave = sorted(leave_days)
            leave_groups = []
            if sorted_leave:
                group_start = sorted_leave[0]
                group_end = sorted_leave[0]
                for ld in sorted_leave[1:]:
                    if (ld - group_end).days == 1:
                        group_end = ld
                    else:
                        leave_groups.append((group_start, group_end))
                        group_start = ld
                        group_end = ld
                leave_groups.append((group_start, group_end))

            for g_start, g_end in leave_groups:
                days_count = sum(
                    1 for d_ord in range((g_end - g_start).days + 1)
                    if (g_start + timedelta(days=d_ord)).strftime("%A").upper() not in ("SATURDAY", "SUNDAY")
                )
                leave = LeaveRequest(
                    employee_id=employee.id,
                    leave_type=random.choice([LeaveType.CASUAL, LeaveType.SICK]),
                    date_from=g_start,
                    date_to=g_end,
                    days_count=float(days_count),
                    is_half_day=False,
                    reason=random.choice([
                        "Personal work", "Feeling unwell", "Family function",
                        "Doctor appointment", "Out of station", "Festival",
                    ]),
                )
                db.add(leave)
                total_leaves += 1

            # Create attendance + check-in for each day
            for day in all_days:
                weekday_name = day.strftime("%A").upper()
                is_weekend = weekday_name in ("SATURDAY", "SUNDAY")

                if is_weekend:
                    continue  # No attendance on weekends

                if day in leave_days:
                    # On leave — create absent/on_leave attendance
                    record = AttendanceRecord(
                        employee_id=employee.id,
                        attendance_date=day,
                        shift_slot_id=shift.id,
                        first_seen_at=None,
                        last_seen_at=None,
                        total_hours=None,
                        status=AttendanceStatus.on_leave,
                    )
                    db.add(record)
                    total_attendance += 1
                    continue

                # Working day — determine status
                roll = random.random()
                if roll < 0.10:
                    # 10% absent (no detection)
                    record = AttendanceRecord(
                        employee_id=employee.id,
                        attendance_date=day,
                        shift_slot_id=shift.id,
                        first_seen_at=None,
                        last_seen_at=None,
                        total_hours=None,
                        status=AttendanceStatus.absent,
                    )
                    db.add(record)
                    total_attendance += 1
                    continue

                is_late = roll < 0.35  # 25% late (of remaining 90%)
                status = AttendanceStatus.late if is_late else AttendanceStatus.present

                checkin_time = _random_checkin_time(shift, day, is_late)
                checkout_time = _random_checkout_time(checkin_time, shift)
                total_hours = round((checkout_time - checkin_time).total_seconds() / 3600, 2)

                record = AttendanceRecord(
                    employee_id=employee.id,
                    attendance_date=day,
                    shift_slot_id=shift.id,
                    first_seen_at=checkin_time,
                    last_seen_at=checkout_time,
                    total_hours=total_hours,
                    status=status,
                    check_in_camera_id=camera_id,
                    check_out_camera_id=camera_id,
                )
                db.add(record)
                total_attendance += 1

                # Create check-in event
                checkin_status = "late" if is_late else "on_time"
                checkin_event = EmployeeCheckIn(
                    employee_id=employee.id,
                    employee_name=employee.name,
                    emp_code=employee.emp_id,
                    checked_in_at=checkin_time,
                    check_in_date=day,
                    shift_slot_id=shift.id,
                    shift_label=shift.label,
                    camera_id=camera_id,
                    face_crop_path=face_crop_path,
                    status=checkin_status,
                )
                db.add(checkin_event)
                total_checkins += 1

        await db.commit()

        print(f"=== Seed Summary ===")
        print(f"  Employees created:     {len(created_employees)}")
        print(f"  Attendance records:    {total_attendance}")
        print(f"  Leave requests:        {total_leaves}")
        print(f"  Check-in events:       {total_checkins}")
        print(f"\nDone! ✓")


if __name__ == "__main__":
    asyncio.run(seed())

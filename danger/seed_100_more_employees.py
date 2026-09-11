#!/usr/bin/env python3
"""Seed 100 more dummy employees with full data chain.

Usage:
    .venv/bin/python danger/seed_100_more_employees.py
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
    AttendanceRecord, AttendanceStatus, Employee, EmployeeCheckIn, Gender, ShiftSlot,
)
from app.core.db.models.leave import LeaveRequest, LeaveType
from app.core.db.models.person import PersonIdentity, PersonFaceEmbedding

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

FIRST_NAMES_M = [
    "Aarav", "Arjun", "Aditya", "Ansh", "Arnav", "Dhruv", "Ishaan", "Kabir",
    "Krishna", "Lakshay", "Mohit", "Nakul", "Om", "Pranav", "Raghav", "Sahil",
    "Tanmay", "Utkarsh", "Varun", "Yash", "Abhinav", "Bharat", "Chirag",
    "Darshan", "Eshan", "Farhan", "Gaurav", "Hemant", "Ishan", "Jayesh",
    "Kartik", "Lokesh", "Manish", "Nikhil", "Omkar", "Piyush", "Rajat",
    "Sachin", "Tushar", "Ujjwal", "Vineet", "Waris", "Yogesh", "Zayan",
    "Akhil", "Bhuvan", "Chetan", "Dev", "Harsh", "Jatin",
]
FIRST_NAMES_F = [
    "Aadhya", "Ananya", "Avni", "Bhavna", "Charvi", "Diya", "Esha", "Falak",
    "Gauri", "Hina", "Ira", "Jiya", "Kiara", "Lavanya", "Meera", "Nisha",
    "Oviya", "Pallavi", "Ritika", "Saanvi", "Tanya", "Uma", "Vaidehi",
    "Yukta", "Zara", "Aditi", "Bhumika", "Chhavi", "Divya", "Ekta",
    "Falguni", "Garima", "Harshita", "Isha", "Juhi", "Kriti", "Lata",
    "Madhuri", "Nandini", "Payal", "Radhika", "Sakshi", "Trisha", "Urvi",
    "Vidhi", "Wafa", "Yamini", "Zoya", "Anika", "Bhavika",
]
LAST_NAMES = [
    "Sharma", "Patel", "Kumar", "Singh", "Gupta", "Verma", "Yadav", "Reddy",
    "Jain", "Mishra", "Tiwari", "Nair", "Pandey", "Chauhan", "Mehta", "Shah",
    "Agarwal", "Bose", "Chatterjee", "Das", "Dubey", "Ghosh", "Iyer",
    "Joshi", "Kapoor", "Malhotra", "Naidu", "Pillai", "Rao", "Saxena",
    "Thakur", "Trivedi", "Varma", "Chopra", "Banerjee",
]


def _gen_employees(n: int, start_idx: int):
    """Generate n employee dicts with unique emp_ids."""
    employees = []
    for i in range(n):
        gender = random.choice(["MALE", "FEMALE"])
        if gender == "MALE":
            first = random.choice(FIRST_NAMES_M)
        else:
            first = random.choice(FIRST_NAMES_F)
        last = random.choice(LAST_NAMES)
        idx = start_idx + i
        employees.append({
            "emp_id": f"EMP{idx:04d}",
            "name": f"{first} {last}",
            "gender": gender,
        })
    return employees


def _download_face(gender: str) -> bytes | None:
    try:
        g = "male" if gender == "MALE" else "female"
        resp = requests.get(
            f"https://randomuser.me/api/?gender={g}&inc=picture&noinfo", timeout=10
        )
        pic_url = resp.json()["results"][0]["picture"]["large"]
        img = requests.get(pic_url, timeout=10)
        return img.content if img.status_code == 200 else None
    except Exception:
        return None


def _upload_minio(data: bytes, prefix: str) -> str | None:
    try:
        from app.modules.storage.minio_client import get_client, BUCKET_PREFIX
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        uid = uuid.uuid4().hex[:8]
        obj = f"crops/{prefix}_{ts}_{uid}.jpg"
        get_client().put_object(BUCKET_PREFIX, obj, io.BytesIO(data), len(data), "image/jpeg")
        return obj
    except Exception:
        return None


def _rand_emb():
    v = np.random.randn(512).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _checkin_time(shift, day, late):
    base = datetime.combine(day, shift.start_time, tzinfo=IST)
    off = timedelta(minutes=random.randint(16, 60) if late else random.randint(-15, 10))
    return (base + off).astimezone(UTC)


async def seed():
    async with AsyncSessionLocal() as db:
        slots = (await db.execute(
            select(ShiftSlot).where(ShiftSlot.is_active.is_(True))
        )).scalars().all()
        if not slots:
            print("No shift slots found"); return

        from app.core.db.models.camera import Camera
        cam = (await db.execute(select(Camera).limit(1))).scalar_one_or_none()
        cam_id = cam.id if cam else None

        # Figure out next emp index
        max_emp = (await db.execute(
            select(Employee.emp_id).where(Employee.emp_id.like("EMP%")).order_by(Employee.emp_id.desc()).limit(1)
        )).scalar_one_or_none()
        start_idx = int(max_emp.replace("EMP", "")) + 1 if max_emp else 100

        emp_list = _gen_employees(100, start_idx)
        print(f"Creating 100 employees (EMP{start_idx:04d} → EMP{start_idx+99:04d})")

        today = date.today()
        start_date = today - timedelta(days=30)
        all_days = [start_date + timedelta(days=d) for d in range(31)]
        working_days = [d for d in all_days if d.weekday() < 5]

        created = []
        for i, ed in enumerate(emp_list):
            # Download face (batch of 100 — may be slow, ~1s each)
            if i % 10 == 0:
                print(f"  Progress: {i}/100 ...")
            img = _download_face(ed["gender"])
            crop = _upload_minio(img, "emp_face") if img else None

            person = PersonIdentity(
                label=ed["name"], is_anonymous=False, face_crop_path=crop,
                best_face_score=round(random.uniform(0.75, 0.98), 2),
                gender=ed["gender"], estimated_age=random.randint(22, 50),
                first_seen_at=datetime.combine(start_date, time(10, 0), tzinfo=UTC),
                last_seen_at=datetime.combine(today, time(10, 0), tzinfo=UTC),
            )
            db.add(person); await db.flush()

            db.add(PersonFaceEmbedding(
                person_identity_id=person.id, embedding=_rand_emb(),
                face_score=round(random.uniform(0.70, 0.95), 2),
                face_crop_path=crop,
                captured_at=datetime.combine(start_date, time(10, 0), tzinfo=UTC),
            ))

            shift = random.choice(slots)
            emp = Employee(
                emp_id=ed["emp_id"], name=ed["name"],
                gender=Gender(ed["gender"]), weekends=["SATURDAY", "SUNDAY"],
                person_identity_id=person.id, shift_slot_id=shift.id,
                face_crop_path=crop,
            )
            db.add(emp); await db.flush()
            created.append((emp, shift, crop))

        await db.commit()
        print(f"\n✓ Created {len(created)} employees\n")

        # Attendance, leaves, checkins
        t_att = t_leave = t_ci = 0
        for emp, shift, crop in created:
            n_leave = random.randint(1, 5)
            leave_days = set(random.sample(working_days, min(n_leave, len(working_days))))

            # Group consecutive leaves
            sl = sorted(leave_days)
            groups = []
            if sl:
                gs, ge = sl[0], sl[0]
                for ld in sl[1:]:
                    if (ld - ge).days <= 1: ge = ld
                    else: groups.append((gs, ge)); gs = ge = ld
                groups.append((gs, ge))
            for gs, ge in groups:
                dc = sum(1 for x in range((ge-gs).days+1) if (gs+timedelta(days=x)).weekday() < 5)
                db.add(LeaveRequest(
                    employee_id=emp.id, leave_type=random.choice([LeaveType.CASUAL, LeaveType.SICK]),
                    date_from=gs, date_to=ge, days_count=float(dc), is_half_day=False,
                    reason=random.choice(["Personal", "Unwell", "Family", "Doctor", "Festival"]),
                ))
                t_leave += 1

            for day in all_days:
                if day.weekday() >= 5: continue
                if day in leave_days:
                    db.add(AttendanceRecord(
                        employee_id=emp.id, attendance_date=day, shift_slot_id=shift.id,
                        status=AttendanceStatus.on_leave,
                    ))
                    t_att += 1; continue

                roll = random.random()
                if roll < 0.10:
                    db.add(AttendanceRecord(
                        employee_id=emp.id, attendance_date=day, shift_slot_id=shift.id,
                        status=AttendanceStatus.absent,
                    ))
                    t_att += 1; continue

                late = roll < 0.35
                ci = _checkin_time(shift, day, late)
                co = ci + timedelta(hours=random.uniform(4, 9))
                hrs = round((co - ci).total_seconds() / 3600, 2)
                st = AttendanceStatus.late if late else AttendanceStatus.present

                db.add(AttendanceRecord(
                    employee_id=emp.id, attendance_date=day, shift_slot_id=shift.id,
                    first_seen_at=ci, last_seen_at=co, total_hours=hrs, status=st,
                    check_in_camera_id=cam_id, check_out_camera_id=cam_id,
                ))
                t_att += 1

                db.add(EmployeeCheckIn(
                    employee_id=emp.id, employee_name=emp.name, emp_code=emp.emp_id,
                    checked_in_at=ci, check_in_date=day, shift_slot_id=shift.id,
                    shift_label=shift.label, camera_id=cam_id, face_crop_path=crop,
                    status="late" if late else "on_time",
                ))
                t_ci += 1

        await db.commit()
        print(f"=== Summary ===")
        print(f"  Employees: {len(created)}")
        print(f"  Attendance: {t_att}")
        print(f"  Leaves: {t_leave}")
        print(f"  Check-ins: {t_ci}")
        print("Done! ✓")


if __name__ == "__main__":
    asyncio.run(seed())

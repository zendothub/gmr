"""Pydantic schemas for Employee registration, updates, and responses."""

from datetime import datetime, date, time
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field

from app.modules.shift_slots.schemas import ShiftSlotResponse


# ---------------------------------------------------------------------------
# Employee
# ---------------------------------------------------------------------------

class EmployeeResponse(BaseModel):
    id: UUID
    emp_id: str
    name: str
    person_identity_id: Optional[UUID] = None
    shift_slot_id: Optional[UUID] = None
    shift_slot: Optional[ShiftSlotResponse] = None
    face_crop_path: Optional[str] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    shift_slot_id: Optional[UUID] = None
    is_active: Optional[bool] = None


# ---------------------------------------------------------------------------
# Registration — Way 1 (image upload) — form fields
# ---------------------------------------------------------------------------

class RegisterByImageForm(BaseModel):
    """Non-file fields submitted alongside the image upload."""
    emp_id: str
    name: str
    shift_slot_id: Optional[UUID] = None


class RegisterByImageResponse(BaseModel):
    """
    Returned by POST /api/v1/employees/register.

    - `already_registered=True`  → face matched an EXISTING person_identity.
      The employee record was updated with the latest face crop.
    - `already_registered=False` → new person_identity + employee created.
    """
    employee: EmployeeResponse
    already_registered: bool
    face_crop_url: Optional[str] = None
    message: str


# ---------------------------------------------------------------------------
# Registration — Way 2 (camera-detected person from debug view)
# ---------------------------------------------------------------------------

class RegisterByCameraBody(BaseModel):
    """PUT /api/v1/employees/register/by-camera — link a debug-selected person."""
    person_identity_id: UUID = Field(..., description="person_identity.id selected from the debug active-tracks view")
    emp_id: str
    name: str
    shift_slot_id: Optional[UUID] = None


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------

class AttendanceRecordResponse(BaseModel):
    id: UUID
    employee_id: UUID
    attendance_date: date
    shift_slot_id: Optional[UUID] = None
    shift_slot: Optional[ShiftSlotResponse] = None
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    total_hours: Optional[float] = None
    status: str  # present / absent / late / half_day
    check_in_camera_id: Optional[UUID] = None
    check_out_camera_id: Optional[UUID] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Attendance Report
# ---------------------------------------------------------------------------

class AttendanceReportEmployee(BaseModel):
    emp_id: str
    name: str
    shift_label: Optional[str] = None

    # For daily
    status: Optional[str] = None            # present / absent / late / half_day
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    total_hours: Optional[float] = None

    # For weekly / monthly aggregates
    total_days: Optional[int] = None
    present_days: Optional[int] = None
    absent_days: Optional[int] = None
    late_days: Optional[int] = None
    avg_hours_per_day: Optional[float] = None


class AttendanceReportSummary(BaseModel):
    total_employees: int
    present: int
    absent: int
    late: int


class AttendanceReportResponse(BaseModel):
    period: str                             # daily | weekly | monthly
    date_from: date
    date_to: date
    employees: List[AttendanceReportEmployee]
    summary: AttendanceReportSummary

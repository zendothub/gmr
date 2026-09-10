"""Pydantic schemas for Employee registration, updates, and responses."""

import enum
from datetime import datetime, date, time
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.modules.shift_slots.schemas import ShiftSlotResponse


# ---------------------------------------------------------------------------
# Gender / Weekend — shared validation
# ---------------------------------------------------------------------------

GENDER_PATTERN = "^(MALE|FEMALE|OTHER)$"

WEEKDAY_VALUES = {
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
}


def validate_weekends(values: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize + validate a list of weekday strings against WEEKDAY_VALUES.

    Raises ValueError on any unrecognized day. Used by both the Pydantic
    schemas below and the raw multipart Form fields in router.py (which
    aren't validated by a BaseModel).
    """
    if values is None:
        return None
    normalized = [v.strip().upper() for v in values]
    invalid = [v for v in normalized if v not in WEEKDAY_VALUES]
    if invalid:
        raise ValueError(
            f"Invalid weekday value(s): {invalid}. Must be one of {sorted(WEEKDAY_VALUES)}."
        )
    return normalized


# ---------------------------------------------------------------------------
# Employee
# ---------------------------------------------------------------------------

class EmployeeResponse(BaseModel):
    id: UUID
    emp_id: str
    name: str
    gender: Optional[str] = None
    weekends: List[str] = Field(default_factory=list)
    person_identity_id: Optional[UUID] = None
    shift_slot_id: Optional[UUID] = None
    shift_slot: Optional[ShiftSlotResponse] = None
    face_crop_path: Optional[str] = None
    face_crop_url: Optional[str] = None   # presigned MinIO URL, populated by router
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("gender", mode="before")
    @classmethod
    def _gender_to_value(cls, v):
        # ORM attribute is a Gender enum member — coerce to its plain value.
        return v.value if isinstance(v, enum.Enum) else v

    @field_validator("weekends", mode="before")
    @classmethod
    def _weekends_default(cls, v):
        # Defensive: the DB column is NOT NULL, but guard against None anyway
        # so a legacy/partial row never breaks serialization.
        return v if v is not None else []


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = Field(None, pattern=GENDER_PATTERN)
    weekends: Optional[List[str]] = None
    shift_slot_id: Optional[UUID] = None
    is_active: Optional[bool] = None

    @field_validator("weekends")
    @classmethod
    def _check_weekends(cls, v):
        return validate_weekends(v)


# ---------------------------------------------------------------------------
# Registration — Way 1 (image upload) — form fields
# ---------------------------------------------------------------------------

class RegisterByImageForm(BaseModel):
    """Non-file fields submitted alongside the image upload."""
    emp_id: str
    name: str
    gender: Optional[str] = Field(None, pattern=GENDER_PATTERN)
    weekends: Optional[List[str]] = None
    shift_slot_id: Optional[UUID] = None

    @field_validator("weekends")
    @classmethod
    def _check_weekends(cls, v):
        return validate_weekends(v)


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
    gender: Optional[str] = Field(None, pattern=GENDER_PATTERN)
    weekends: Optional[List[str]] = None
    shift_slot_id: Optional[UUID] = None

    @field_validator("weekends")
    @classmethod
    def _check_weekends(cls, v):
        return validate_weekends(v)


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
    status: Optional[str] = None            # present / absent / late / half_day / on_leave / weekend
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    total_hours: Optional[float] = None
    leave_type: Optional[str] = None        # CASUAL / SICK, only set when status=on_leave
    is_half_day: Optional[bool] = None      # only set when status=on_leave

    # For weekly / monthly aggregates
    total_days: Optional[int] = None
    present_days: Optional[int] = None
    absent_days: Optional[int] = None
    late_days: Optional[int] = None
    leave_days: Optional[float] = None      # fractional when a half-day leave falls in range
    avg_hours_per_day: Optional[float] = None


class AttendanceReportSummary(BaseModel):
    total_employees: int
    present: int
    absent: int
    late: int
    weekend_offs: int = 0  # employees whose weekly-off falls on the report date (daily reports only)
    on_leave: int = 0      # daily: on leave that date. weekly/monthly/custom: had any leave day in range.


class AttendanceReportResponse(BaseModel):
    period: str                             # daily | weekly | monthly | custom
    date_from: date
    date_to: date
    employees: List[AttendanceReportEmployee]
    summary: AttendanceReportSummary

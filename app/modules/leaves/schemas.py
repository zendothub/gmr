"""Pydantic schemas for Leave CRUD (casual/sick leave)."""

from datetime import date, datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

LEAVE_TYPE_PATTERN = "^(CASUAL|SICK)$"

# Combined casual+sick pool, per employee, per calendar year.
ANNUAL_LEAVE_QUOTA = 12


class LeaveCreate(BaseModel):
    emp_id: str
    leave_type: str = Field(..., pattern=LEAVE_TYPE_PATTERN)
    date_from: date
    date_to: date
    reason: Optional[str] = Field(None, max_length=500)

    @model_validator(mode="after")
    def _check_range(self) -> "LeaveCreate":
        if self.date_to < self.date_from:
            raise ValueError("date_to cannot be before date_from")
        if self.date_from.year != self.date_to.year:
            raise ValueError("leave request must fall within a single calendar year")
        return self


class LeaveUpdate(BaseModel):
    leave_type: Optional[str] = Field(None, pattern=LEAVE_TYPE_PATTERN)
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    reason: Optional[str] = Field(None, max_length=500)

    @model_validator(mode="after")
    def _check_range(self) -> "LeaveUpdate":
        if self.date_from is not None and self.date_to is not None:
            if self.date_to < self.date_from:
                raise ValueError("date_to cannot be before date_from")
            if self.date_from.year != self.date_to.year:
                raise ValueError("leave request must fall within a single calendar year")
        return self


class LeaveResponse(BaseModel):
    id: UUID
    employee_id: UUID
    emp_id: str
    employee_name: str
    leave_type: str
    date_from: date
    date_to: date
    days_count: int
    reason: Optional[str] = None
    balance_remaining: int  # ANNUAL_LEAVE_QUOTA minus days used in date_from.year (incl. this request)
    created_at: datetime
    updated_at: datetime

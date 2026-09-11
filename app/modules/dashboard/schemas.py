"""Pydantic schemas for the dashboard's current-day attendance summary."""

from datetime import date
from typing import List, Optional

from pydantic import BaseModel


class GenderCount(BaseModel):
    male: int = 0
    female: int = 0
    other: int = 0
    unspecified: int = 0  # employee record has no gender set


class EmployeeBrief(BaseModel):
    emp_id: str
    name: str
    gender: Optional[str] = None    # MALE / FEMALE / OTHER / None (unspecified)
    leave_type: Optional[str] = None  # only set on on_leave entries
    is_half_day: Optional[bool] = None  # only set on on_leave entries


class TodayDashboardResponse(BaseModel):
    date: date
    total_employees: int
    present: int          # includes late (late = was present, just tardy)
    absent: int
    on_leave: int
    late: int
    weekend_offs: int
    gender: GenderCount    # breakdown of employees present today (present + late)

    present_employees: List[EmployeeBrief] = []
    on_leave_employees: List[EmployeeBrief] = []
    absent_employees: List[EmployeeBrief] = []  # no leave AND no check-in today

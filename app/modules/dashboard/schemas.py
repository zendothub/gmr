"""Pydantic schemas for the dashboard's current-day attendance summary."""

from datetime import date

from pydantic import BaseModel


class GenderCount(BaseModel):
    male: int = 0
    female: int = 0
    other: int = 0
    unspecified: int = 0  # employee record has no gender set


class TodayDashboardResponse(BaseModel):
    date: date
    total_employees: int
    present: int          # includes late (late = was present, just tardy)
    absent: int
    on_leave: int
    late: int
    weekend_offs: int
    gender: GenderCount    # breakdown of employees present today (present + late)

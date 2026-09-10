"""Current-day attendance summary — reuses report_service's daily report so
present/absent/leave/weekend logic isn't duplicated here."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.attendance import AttendanceStatus, Employee, Gender
from app.modules.dashboard.schemas import GenderCount, TodayDashboardResponse
from app.modules.employees import report_service

# Site-local day boundary — matches how the rest of the app treats "today"
# (report timestamps, the v2 analytics dashboard's time_range=today).
IST = timezone(timedelta(hours=5, minutes=30))

_PRESENT_STATUSES = (AttendanceStatus.present.value, AttendanceStatus.late.value)


def _today_ist() -> date:
    return datetime.now(timezone.utc).astimezone(IST).date()


async def get_today_summary(db: AsyncSession) -> TodayDashboardResponse:
    today = _today_ist()
    report = await report_service.get_daily_report(db, report_date=today)

    present_emp_ids = [
        row.emp_id for row in report.employees if row.status in _PRESENT_STATUSES
    ]

    gender = GenderCount()
    if present_emp_ids:
        genders = (
            await db.execute(
                select(Employee.gender).where(Employee.emp_id.in_(present_emp_ids))
            )
        ).scalars().all()
        for g in genders:
            if g == Gender.MALE:
                gender.male += 1
            elif g == Gender.FEMALE:
                gender.female += 1
            elif g == Gender.OTHER:
                gender.other += 1
            else:
                gender.unspecified += 1

    return TodayDashboardResponse(
        date=today,
        total_employees=report.summary.total_employees,
        present=report.summary.present,
        absent=report.summary.absent,
        on_leave=report.summary.on_leave,
        late=report.summary.late,
        weekend_offs=report.summary.weekend_offs,
        gender=gender,
    )

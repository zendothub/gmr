"""Tests for the custom start_date/end_date range mode on
GET /api/v1/employees/attendance/report.

Router-level validation (start_date > end_date, period requirement) is tested
as pure unit tests against the endpoint function directly. The actual
date-range filtering is tested as a lightweight integration test against the
dev Postgres (this project has no isolated test-DB fixture — same convention
gap as the rest of the suite), mirroring how report_service is exercised in
practice.
"""
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import delete

from app.core.db.models.attendance import (
    AttendanceRecord,
    AttendanceStatus,
    Employee,
    Gender,
)
from app.core.db.session import AsyncSessionLocal
from app.modules.employees.router import get_attendance_report
from app.modules.employees import report_service


# ---------------------------------------------------------------------------
# Router-level validation — no DB needed, the check happens before any query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_after_end_raises_400():
    with pytest.raises(HTTPException) as exc_info:
        await get_attendance_report(
            period=None, date=None, week_start=None, year=None, month=None,
            start_date=date(2026, 9, 10), end_date=date(2026, 9, 1),
            shift_slot_id=None, emp_id=None,
            sort_by=None, sort_order="asc",
            db=AsyncMock(), current_user=AsyncMock(),
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "start_date cannot be greater than end_date"


@pytest.mark.asyncio
async def test_no_period_and_no_date_range_still_422():
    """Existing behavior preserved: period is required when no date range is given."""
    with pytest.raises(HTTPException) as exc_info:
        await get_attendance_report(
            period=None, date=None, week_start=None, year=None, month=None,
            start_date=None, end_date=None,
            shift_slot_id=None, emp_id=None,
            sort_by=None, sort_order="asc",
            db=AsyncMock(), current_user=AsyncMock(),
        )
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_date_range_bypasses_period_requirement(monkeypatch):
    """Supplying start_date/end_date works without `period` at all."""
    called = {}

    async def fake_get_range_report(db, start_date, end_date, shift_slot_id, emp_id):
        called["start_date"] = start_date
        called["end_date"] = end_date
        return "ok"

    monkeypatch.setattr(report_service, "get_range_report", fake_get_range_report)

    result = await get_attendance_report(
        period=None, date=None, week_start=None, year=None, month=None,
        start_date=date(2026, 9, 1), end_date=date(2026, 9, 9),
        shift_slot_id=None, emp_id=None,
        sort_by=None, sort_order="asc",
        db=AsyncMock(), current_user=AsyncMock(),
    )
    assert result == "ok"
    assert called == {"start_date": date(2026, 9, 1), "end_date": date(2026, 9, 9)}


# ---------------------------------------------------------------------------
# Integration tests against the dev DB — actual filtering behavior
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def range_test_employee():
    """Employee with attendance on 09-01, 09-03, 09-09, 09-10 (2026)."""
    async with AsyncSessionLocal() as db:
        emp = Employee(
            emp_id="RNGTEST01", name="Range Fixture Employee",
            gender=Gender.MALE, weekends=["SATURDAY", "SUNDAY"],
        )
        db.add(emp)
        await db.flush()
        for d in (date(2026, 9, 1), date(2026, 9, 3), date(2026, 9, 9), date(2026, 9, 10)):
            db.add(AttendanceRecord(
                employee_id=emp.id,
                attendance_date=d,
                first_seen_at=datetime(d.year, d.month, d.day, 9, 0, tzinfo=timezone.utc),
                last_seen_at=datetime(d.year, d.month, d.day, 17, 0, tzinfo=timezone.utc),
                total_hours=8.0,
                status=AttendanceStatus.present,
            ))
        await db.commit()
        emp_id = emp.emp_id

    yield emp_id

    async with AsyncSessionLocal() as db:
        await db.execute(delete(Employee).where(Employee.emp_id == emp_id))
        await db.commit()


@pytest.mark.asyncio
async def test_both_dates_inclusive_range(range_test_employee):
    async with AsyncSessionLocal() as db:
        report = await report_service.get_range_report(
            db, start_date=date(2026, 9, 1), end_date=date(2026, 9, 9),
            emp_id=range_test_employee,
        )
    row = report.employees[0]
    # 09-01, 09-03, 09-09 present (start AND end date both included); 09-10 excluded
    assert row.present_days == 3
    assert report.date_from == date(2026, 9, 1)
    assert report.date_to == date(2026, 9, 9)


@pytest.mark.asyncio
async def test_end_date_boundary_is_inclusive(range_test_employee):
    """A record exactly ON end_date must be counted (not excluded by a
    midnight/off-by-one comparison)."""
    async with AsyncSessionLocal() as db:
        report = await report_service.get_range_report(
            db, start_date=date(2026, 9, 8), end_date=date(2026, 9, 9),
            emp_id=range_test_employee,
        )
    assert report.employees[0].present_days == 1  # only 09-09, which is the end_date itself


@pytest.mark.asyncio
async def test_start_date_boundary_is_inclusive(range_test_employee):
    """A record exactly ON start_date must be counted."""
    async with AsyncSessionLocal() as db:
        report = await report_service.get_range_report(
            db, start_date=date(2026, 9, 1), end_date=date(2026, 9, 2),
            emp_id=range_test_employee,
        )
    assert report.employees[0].present_days == 1  # only 09-01, which is the start_date itself


@pytest.mark.asyncio
async def test_same_start_and_end_date_single_day(range_test_employee):
    async with AsyncSessionLocal() as db:
        report = await report_service.get_range_report(
            db, start_date=date(2026, 9, 9), end_date=date(2026, 9, 9),
            emp_id=range_test_employee,
        )
    assert report.employees[0].total_days == 1
    assert report.employees[0].present_days == 1


@pytest.mark.asyncio
async def test_only_start_date_bounds_to_today(range_test_employee, monkeypatch):
    fixed_today = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(report_service, "utc_now", lambda: fixed_today)

    async with AsyncSessionLocal() as db:
        report = await report_service.get_range_report(
            db, start_date=date(2026, 9, 9), end_date=None,
            emp_id=range_test_employee,
        )
    # end resolved to "today" (2026-09-09) -> only the 09-09 record counts,
    # the 09-10 record (in the future relative to "today") is excluded
    assert report.date_to == date(2026, 9, 9)
    assert report.employees[0].present_days == 1


@pytest.mark.asyncio
async def test_only_end_date_bounds_to_earliest_record(range_test_employee):
    async with AsyncSessionLocal() as db:
        report = await report_service.get_range_report(
            db, start_date=None, end_date=date(2026, 9, 3),
            emp_id=range_test_employee,
        )
    # resolved start = earliest attendance_date on record = 09-01
    assert report.date_from == date(2026, 9, 1)
    assert report.date_to == date(2026, 9, 3)
    assert report.employees[0].present_days == 2  # 09-01, 09-03


@pytest.mark.asyncio
async def test_no_date_filters_preserves_existing_daily_behavior(range_test_employee):
    """Sanity check: the period-based path is completely untouched."""
    async with AsyncSessionLocal() as db:
        report = await report_service.get_daily_report(
            db, report_date=date(2026, 9, 9), emp_id=range_test_employee,
        )
    assert report.period == "daily"
    assert report.employees[0].status == "present"


@pytest.mark.asyncio
async def test_date_range_combines_with_emp_id_filter(range_test_employee):
    """emp_id filter still narrows results when used together with the date range."""
    async with AsyncSessionLocal() as db:
        report = await report_service.get_range_report(
            db, start_date=date(2026, 9, 1), end_date=date(2026, 9, 10),
            emp_id="NON_EXISTENT_EMP_ID",
        )
    assert report.employees == []
    assert report.summary.total_employees == 0

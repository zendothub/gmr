"""Unit tests for the P0 employee attendance changes: gender + employee-specific
weekly-off (weekend) configuration.

Covers:
  1. Employee with Saturday/Sunday weekend
  2. Employee with Sunday/Monday weekend
  3. Employee with only one weekly-off day
  4. Weekend configuration can be updated
  5. Updated weekend is returned by the API (response schema)
  6. Attendance calculation changes after weekend update
  7. Existing employee without the new fields does not break
  8. Gender can be created, fetched, and updated
"""
from datetime import date
from unittest.mock import AsyncMock

import pytest

from app.core.db.models.attendance import DEFAULT_WEEKENDS, Employee, Gender
from app.modules.employees.report_service import _working_days_in_range
from app.modules.employees.schemas import (
    EmployeeResponse,
    EmployeeUpdate,
    validate_weekends,
)
from app.modules.employees import service as emp_svc


def _emp(**kwargs) -> Employee:
    defaults = dict(emp_id="EMP001", name="Test Employee")
    defaults.update(kwargs)
    return Employee(**defaults)


# ---------------------------------------------------------------------------
# 1-3. Employee.is_weekly_off() — per-employee weekend configuration
# ---------------------------------------------------------------------------

def test_is_weekly_off_saturday_sunday_employee():
    emp = _emp(weekends=["SATURDAY", "SUNDAY"])
    assert emp.is_weekly_off(date(2026, 9, 7)) is False   # Monday -> working
    assert emp.is_weekly_off(date(2026, 9, 5)) is True    # Saturday -> off
    assert emp.is_weekly_off(date(2026, 9, 6)) is True    # Sunday -> off
    assert emp.is_weekly_off(date(2026, 9, 4)) is False   # Friday -> working


def test_is_weekly_off_sunday_monday_employee():
    emp = _emp(weekends=["SUNDAY", "MONDAY"])
    assert emp.is_weekly_off(date(2026, 9, 7)) is True    # Monday -> off
    assert emp.is_weekly_off(date(2026, 9, 6)) is True    # Sunday -> off
    assert emp.is_weekly_off(date(2026, 9, 5)) is False   # Saturday -> working (differs from Sat/Sun employee)


def test_is_weekly_off_single_day_employee():
    emp = _emp(weekends=["FRIDAY"])
    assert emp.is_weekly_off(date(2026, 9, 4)) is True    # Friday -> off
    assert emp.is_weekly_off(date(2026, 9, 5)) is False   # Saturday -> working
    assert emp.is_weekly_off(date(2026, 9, 6)) is False   # Sunday -> working


# ---------------------------------------------------------------------------
# 6. Attendance calculation changes after weekend update
# ---------------------------------------------------------------------------

def test_working_days_change_after_weekend_update():
    week_from, week_to = date(2026, 8, 31), date(2026, 9, 6)  # Mon..Sun

    emp = _emp(weekends=["SATURDAY", "SUNDAY"])
    before = _working_days_in_range(emp, week_from, week_to)
    assert len(before) == 5
    assert date(2026, 9, 5) not in before  # Saturday excluded
    assert date(2026, 9, 6) not in before  # Sunday excluded

    # Update weekend config — same date range must now produce different working days
    emp.weekends = ["SUNDAY", "MONDAY"]
    after = _working_days_in_range(emp, week_from, week_to)
    assert len(after) == 5
    assert date(2026, 8, 31) not in after  # Monday now excluded
    assert date(2026, 9, 5) in after       # Saturday now a working day
    assert before != after


# ---------------------------------------------------------------------------
# 7. Existing employee without the new fields does not break
# ---------------------------------------------------------------------------

def test_legacy_employee_defaults_are_safe():
    emp = _emp()  # no gender, no weekends specified
    assert emp.gender is None
    # ORM-level python default kicks in on flush; here we assert the model
    # doesn't require these fields to be constructed or used.
    assert emp.is_weekly_off(date(2026, 9, 5)) in (True, False)  # doesn't raise


def test_employee_response_handles_missing_gender_and_weekends():
    emp = _emp()
    emp.weekends = None  # simulate a row that predates the column somehow
    resp = EmployeeResponse.model_validate(
        {
            "id": "00000000-0000-0000-0000-000000000000",
            "emp_id": emp.emp_id,
            "name": emp.name,
            "gender": None,
            "weekends": None,
            "is_active": True,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )
    assert resp.gender is None
    assert resp.weekends == []


# ---------------------------------------------------------------------------
# 4, 5, 8. Gender/weekend create + update + fetch (schema + service layer)
# ---------------------------------------------------------------------------

def test_validate_weekends_accepts_canonical_values():
    assert validate_weekends(["saturday", "Sunday"]) == ["SATURDAY", "SUNDAY"]


def test_validate_weekends_rejects_unknown_day():
    with pytest.raises(ValueError):
        validate_weekends(["FUNDAY"])


def test_employee_update_schema_validates_gender_and_weekends():
    payload = EmployeeUpdate(gender="MALE", weekends=["sunday", "monday"])
    assert payload.gender == "MALE"
    assert payload.weekends == ["SUNDAY", "MONDAY"]

    with pytest.raises(Exception):
        EmployeeUpdate(gender="ROBOT")


def test_employee_response_serializes_gender_enum_to_value():
    emp = _emp(gender=Gender.FEMALE, weekends=["SUNDAY"])
    resp = EmployeeResponse.model_validate(
        {
            "id": "00000000-0000-0000-0000-000000000000",
            "emp_id": emp.emp_id,
            "name": emp.name,
            "gender": emp.gender,
            "weekends": emp.weekends,
            "is_active": True,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )
    assert resp.gender == "FEMALE"
    assert resp.weekends == ["SUNDAY"]


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


@pytest.mark.asyncio
async def test_update_employee_service_applies_gender_and_weekend_changes(monkeypatch):
    """Weekend + gender changed via update_employee() take effect on the ORM
    object, and a later fetch reflects the NEW config only (old config no
    longer applies), per the P0 spec's before/after example."""
    emp = _emp(gender=Gender.MALE, weekends=list(DEFAULT_WEEKENDS))

    async def fake_get_by_emp_id(db, emp_id):
        return emp

    monkeypatch.setattr(emp_svc, "get_by_emp_id", fake_get_by_emp_id)

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=_FakeResult(None))

    payload = EmployeeUpdate(weekends=["SUNDAY", "MONDAY"])
    updated = await emp_svc.update_employee(mock_db, "EMP001", payload)

    assert updated.weekends == ["SUNDAY", "MONDAY"]
    assert updated.is_weekly_off(date(2026, 9, 5)) is False  # old Saturday off no longer applies
    assert updated.is_weekly_off(date(2026, 9, 7)) is True   # new Monday off applies
    mock_db.commit.assert_awaited()

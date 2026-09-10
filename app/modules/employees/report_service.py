"""Attendance report generation from the materialised attendance_records table.

Supports three periods, plus a custom date range:
  daily   — one row per employee for a single date
  weekly  — aggregated counts / avg_hours per employee across 7 days
  monthly — aggregated counts / avg_hours per employee across a calendar month
  custom  — aggregated counts / avg_hours per employee across an arbitrary
            inclusive [start_date, end_date] window (get_range_report)

Data is read straight from attendance_records (never recomputed from
track_sessions), so the output is permanently consistent regardless of
background dedup jobs running between two report downloads.
"""

from __future__ import annotations

from datetime import date, timedelta, datetime, timezone
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, and_, func
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.attendance import (
    Employee,
    AttendanceRecord,
    AttendanceStatus,
    ShiftSlot,
)
from app.core.db.models.leave import LeaveRequest
from app.modules.employees.schemas import (
    AttendanceReportEmployee,
    AttendanceReportResponse,
    AttendanceReportSummary,
)
from app.utils.time_utils import utc_now

# IST timezone (UTC+5:30) — timestamps are stored in UTC in the DB,
# but the frontend expects them in local time (IST).
IST = timezone(timedelta(hours=5, minutes=30))


def _utc_to_ist(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert a UTC datetime to IST for frontend display.
    If naive, assume UTC and convert."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def get_daily_report(
    db: AsyncSession,
    report_date: date,
    shift_slot_id: Optional[UUID] = None,
    emp_id: Optional[str] = None,
) -> AttendanceReportResponse:
    """One row per employee for a specific date."""
    employees = await _load_employees(db, shift_slot_id=shift_slot_id, emp_id=emp_id)
    emp_ids = [e.id for e in employees]

    # Load attendance records for that date
    records_q = select(AttendanceRecord).where(
        AttendanceRecord.employee_id.in_(emp_ids),
        AttendanceRecord.attendance_date == report_date,
    )
    records_rows = (await db.execute(records_q)).scalars().all()
    records_by_emp = {r.employee_id: r for r in records_rows}

    # Leave covering this date (overlap is prevented at apply-time, so at
    # most one per employee for a single date).
    leaves_q = select(LeaveRequest).where(
        LeaveRequest.employee_id.in_(emp_ids),
        LeaveRequest.date_from <= report_date,
        LeaveRequest.date_to >= report_date,
    )
    leave_by_emp = {
        l.employee_id: l for l in (await db.execute(leaves_q)).scalars().all()
    }

    # Build shift label map
    slot_map = await _load_slot_map(db, employees)

    rows: List[AttendanceReportEmployee] = []
    summary = _empty_summary(len(employees))

    for emp in employees:
        rec = records_by_emp.get(emp.id)
        leave = leave_by_emp.get(emp.id)
        shift_label = slot_map.get(emp.shift_slot_id)

        # Priority: weekly-off > leave > camera record > absent. A leave day
        # that happens to land on the employee's own weekly-off shows as
        # "weekend" (it was never charged against their quota either).
        if emp.is_weekly_off(report_date):
            row = AttendanceReportEmployee(
                emp_id=emp.emp_id,
                name=emp.name,
                shift_label=shift_label,
                status="weekend",
            )
            summary.weekend_offs += 1
        elif leave:
            row = AttendanceReportEmployee(
                emp_id=emp.emp_id,
                name=emp.name,
                shift_label=shift_label,
                status=AttendanceStatus.on_leave.value,
                leave_type=leave.leave_type.value,
            )
            summary.on_leave += 1
        elif rec:
            status = rec.status.value if rec.status else AttendanceStatus.present.value
            row = AttendanceReportEmployee(
                emp_id=emp.emp_id,
                name=emp.name,
                shift_label=shift_label,
                status=status,
                first_seen_at=_utc_to_ist(rec.first_seen_at),
                last_seen_at=_utc_to_ist(rec.last_seen_at),
                total_hours=rec.total_hours,
            )
            _increment_summary(summary, status)
        else:
            row = AttendanceReportEmployee(
                emp_id=emp.emp_id,
                name=emp.name,
                shift_label=shift_label,
                status=AttendanceStatus.absent.value,
            )
            summary.absent += 1

        rows.append(row)

    return AttendanceReportResponse(
        period="daily",
        date_from=report_date,
        date_to=report_date,
        employees=rows,
        summary=summary,
    )


async def get_weekly_report(
    db: AsyncSession,
    week_start: date,
    shift_slot_id: Optional[UUID] = None,
    emp_id: Optional[str] = None,
) -> AttendanceReportResponse:
    """Aggregate per employee across a 7-day window starting on `week_start`."""
    week_end = week_start + timedelta(days=6)
    return await _aggregate_report(db, week_start, week_end, "weekly", shift_slot_id, emp_id)


async def get_range_report(
    db: AsyncSession,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    shift_slot_id: Optional[UUID] = None,
    emp_id: Optional[str] = None,
) -> AttendanceReportResponse:
    """Custom inclusive date-range report — alternative to period=daily/weekly/monthly.

    attendance_date is a plain DATE column, so `>= start_date AND <= end_date`
    is exact at the SQL level with no midnight/timezone edge cases.

      - both start_date and end_date given -> that exact window
      - only start_date given               -> [start_date, today]
      - only end_date given                 -> [earliest recorded attendance_date, end_date]

    Caller (router) is expected to validate start_date <= end_date beforehand.
    """
    resolved_end = end_date if end_date is not None else utc_now().date()

    if start_date is not None:
        resolved_start = start_date
    else:
        employees = await _load_employees(db, shift_slot_id=shift_slot_id, emp_id=emp_id)
        earliest = await _earliest_attendance_date(db, [e.id for e in employees])
        resolved_start = earliest if earliest is not None else resolved_end

    return await _aggregate_report(
        db, resolved_start, resolved_end, "custom", shift_slot_id, emp_id
    )


async def get_monthly_report(
    db: AsyncSession,
    year: int,
    month: int,
    shift_slot_id: Optional[UUID] = None,
    emp_id: Optional[str] = None,
) -> AttendanceReportResponse:
    """Aggregate per employee for the full calendar month."""
    from calendar import monthrange
    _, last_day = monthrange(year, month)
    month_start = date(year, month, 1)
    month_end = date(year, month, last_day)
    return await _aggregate_report(db, month_start, month_end, "monthly", shift_slot_id, emp_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _aggregate_report(
    db: AsyncSession,
    date_from: date,
    date_to: date,
    period: str,
    shift_slot_id: Optional[UUID],
    emp_id_filter: Optional[str],
) -> AttendanceReportResponse:
    employees = await _load_employees(db, shift_slot_id=shift_slot_id, emp_id=emp_id_filter)
    emp_ids = [e.id for e in employees]

    # Load all records in the date range for these employees
    records_q = select(AttendanceRecord).where(
        AttendanceRecord.employee_id.in_(emp_ids),
        AttendanceRecord.attendance_date >= date_from,
        AttendanceRecord.attendance_date <= date_to,
    )
    all_records = (await db.execute(records_q)).scalars().all()

    # Group by employee
    by_emp: dict[UUID, list[AttendanceRecord]] = {e.id: [] for e in employees}
    for r in all_records:
        if r.employee_id in by_emp:
            by_emp[r.employee_id].append(r)

    # Leaves overlapping the report window, grouped by employee.
    leaves_q = select(LeaveRequest).where(
        LeaveRequest.employee_id.in_(emp_ids),
        LeaveRequest.date_from <= date_to,
        LeaveRequest.date_to >= date_from,
    )
    leaves_by_emp: dict[UUID, list[LeaveRequest]] = {e.id: [] for e in employees}
    for l in (await db.execute(leaves_q)).scalars().all():
        if l.employee_id in leaves_by_emp:
            leaves_by_emp[l.employee_id].append(l)

    slot_map = await _load_slot_map(db, employees)
    rows: List[AttendanceReportEmployee] = []
    summary = _empty_summary(len(employees))
    summary.absent = 0  # will compute per-employee

    for emp in employees:
        recs = by_emp[emp.id]
        present_recs = [r for r in recs if r.status in (AttendanceStatus.present, AttendanceStatus.late)]
        leaves = leaves_by_emp[emp.id]

        # Working days for THIS employee only — days in range that don't fall
        # on their own weekly-off configuration. Two employees in the same
        # report period can have different totals here.
        working_days = _working_days_in_range(emp, date_from, date_to)
        total_days = len(working_days)

        # Leave takes priority over a camera record on the same day (mirrors
        # the daily report's priority order).
        leave_days_set = {d for d in working_days if _date_covered_by_leaves(leaves, d)}
        present_on_working_days = {
            r.attendance_date for r in present_recs
            if r.attendance_date in working_days and r.attendance_date not in leave_days_set
        }
        leave_days = len(leave_days_set)
        absent_days = max(total_days - len(present_on_working_days) - leave_days, 0)
        late_days = sum(
            1 for r in recs
            if r.status == AttendanceStatus.late and r.attendance_date not in leave_days_set
        )
        hours_list = [
            r.total_hours for r in present_recs
            if r.total_hours is not None and r.attendance_date not in leave_days_set
        ]
        avg_hours = round(sum(hours_list) / len(hours_list), 2) if hours_list else None

        shift_label = slot_map.get(emp.shift_slot_id)

        # For summary: count employee present if they have any present_rec in range
        # (unfiltered by working-day, matching prior behavior — only leave-covered
        # dates are excluded, since leave takes priority over a camera record).
        present_recs_excl_leave = [r for r in present_recs if r.attendance_date not in leave_days_set]
        if present_recs_excl_leave:
            summary.present += 1
        else:
            summary.absent += 1
        if late_days:
            summary.late += 1
        if leave_days:
            summary.on_leave += 1

        rows.append(AttendanceReportEmployee(
            emp_id=emp.emp_id,
            name=emp.name,
            shift_label=shift_label,
            total_days=total_days,
            present_days=len(present_on_working_days),
            absent_days=absent_days,
            late_days=late_days,
            leave_days=leave_days,
            avg_hours_per_day=avg_hours,
        ))

    return AttendanceReportResponse(
        period=period,
        date_from=date_from,
        date_to=date_to,
        employees=rows,
        summary=summary,
    )


async def _earliest_attendance_date(db: AsyncSession, emp_ids: List[UUID]) -> Optional[date]:
    """MIN(attendance_date) across the given employees — DB-level, single indexed query.
    Used to bound an open-ended `end_date`-only range report."""
    if not emp_ids:
        return None
    result = await db.execute(
        select(func.min(AttendanceRecord.attendance_date)).where(
            AttendanceRecord.employee_id.in_(emp_ids)
        )
    )
    return result.scalar()


def _date_covered_by_leaves(leaves: list[LeaveRequest], d: date) -> bool:
    """True if `d` falls within any of the given (non-overlapping) leave ranges."""
    return any(l.date_from <= d <= l.date_to for l in leaves)


def _working_days_in_range(emp: Employee, date_from: date, date_to: date) -> set:
    """All dates in [date_from, date_to] that are NOT one of `emp`'s own
    weekly-off days. Each employee's weekend is independent — never a
    global Saturday/Sunday assumption."""
    days = set()
    d = date_from
    while d <= date_to:
        if not emp.is_weekly_off(d):
            days.add(d)
        d += timedelta(days=1)
    return days


async def _load_employees(
    db: AsyncSession,
    shift_slot_id: Optional[UUID] = None,
    emp_id: Optional[str] = None,
) -> List[Employee]:
    q = select(Employee).where(Employee.is_active.is_(True))
    if shift_slot_id:
        q = q.where(Employee.shift_slot_id == shift_slot_id)
    if emp_id:
        q = q.where(Employee.emp_id == emp_id)
    q = q.order_by(Employee.name)
    return (await db.execute(q)).scalars().all()


async def _load_slot_map(db: AsyncSession, employees: List[Employee]) -> dict:
    """Build {slot_id: label} map for the given employees."""
    slot_ids = {e.shift_slot_id for e in employees if e.shift_slot_id}
    if not slot_ids:
        return {}
    slots = (
        await db.execute(select(ShiftSlot).where(ShiftSlot.id.in_(slot_ids)))
    ).scalars().all()
    return {s.id: s.label for s in slots}


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

# Daily reports carry a single `status` per employee (no present/absent count),
# so present/absent sort falls back to a presence ranking instead of a count.
_STATUS_PRESENCE_RANK = {
    AttendanceStatus.present.value: 2,
    AttendanceStatus.late.value: 1,
    AttendanceStatus.absent.value: 0,
    AttendanceStatus.on_leave.value: -1,
    "weekend": -1,
}

SORTABLE_FIELDS = ("present", "absent", "check_in", "check_out")


def _sort_value(row: AttendanceReportEmployee, period: str, sort_by: str):
    if sort_by == "check_in":
        return row.first_seen_at
    if sort_by == "check_out":
        return row.last_seen_at

    if period == "daily":
        rank = _STATUS_PRESENCE_RANK.get(row.status, -1)
        return rank if sort_by == "present" else -rank

    return row.present_days if sort_by == "present" else row.absent_days


def sort_report_rows(
    rows: List[AttendanceReportEmployee],
    period: str,
    sort_by: Optional[str],
    sort_order: str = "asc",
) -> List[AttendanceReportEmployee]:
    """Sort report rows by present/absent/check_in/check_out.

    Rows with no value for the sort field (e.g. check_in on an absent
    employee) always sort to the end, regardless of asc/desc.
    """
    if not sort_by:
        return rows

    reverse = sort_order == "desc"
    with_value = [r for r in rows if _sort_value(r, period, sort_by) is not None]
    without_value = [r for r in rows if _sort_value(r, period, sort_by) is None]
    with_value.sort(key=lambda r: _sort_value(r, period, sort_by), reverse=reverse)
    return with_value + without_value


def _empty_summary(total: int) -> AttendanceReportSummary:
    return AttendanceReportSummary(total_employees=total, present=0, absent=0, late=0)


def _increment_summary(summary: AttendanceReportSummary, status: str) -> None:
    if status == AttendanceStatus.present.value:
        summary.present += 1
    elif status == AttendanceStatus.late.value:
        summary.present += 1  # late = was present
        summary.late += 1
    elif status == AttendanceStatus.absent.value:
        summary.absent += 1

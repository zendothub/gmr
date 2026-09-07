"""Attendance report generation from the materialised attendance_records table.

Supports three periods:
  daily   — one row per employee for a single date
  weekly  — aggregated counts / avg_hours per employee across 7 days
  monthly — aggregated counts / avg_hours per employee across a calendar month

Data is read straight from attendance_records (never recomputed from
track_sessions), so the output is permanently consistent regardless of
background dedup jobs running between two report downloads.
"""

from __future__ import annotations

from datetime import date, timedelta
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
from app.modules.employees.schemas import (
    AttendanceReportEmployee,
    AttendanceReportResponse,
    AttendanceReportSummary,
)


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

    # Build shift label map
    slot_map = await _load_slot_map(db, employees)

    rows: List[AttendanceReportEmployee] = []
    summary = _empty_summary(len(employees))

    for emp in employees:
        rec = records_by_emp.get(emp.id)
        shift_label = slot_map.get(emp.shift_slot_id)
        if rec:
            status = rec.status.value if rec.status else AttendanceStatus.present.value
            row = AttendanceReportEmployee(
                emp_id=emp.emp_id,
                name=emp.name,
                shift_label=shift_label,
                status=status,
                first_seen_at=rec.first_seen_at,
                last_seen_at=rec.last_seen_at,
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
    total_days = (date_to - date_from).days + 1

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

    slot_map = await _load_slot_map(db, employees)
    rows: List[AttendanceReportEmployee] = []
    summary = _empty_summary(len(employees))
    summary.absent = 0  # will compute per-employee

    for emp in employees:
        recs = by_emp[emp.id]
        present_recs = [r for r in recs if r.status in (AttendanceStatus.present, AttendanceStatus.late)]
        absent_days = total_days - len(present_recs)
        late_days = sum(1 for r in recs if r.status == AttendanceStatus.late)
        hours_list = [r.total_hours for r in present_recs if r.total_hours is not None]
        avg_hours = round(sum(hours_list) / len(hours_list), 2) if hours_list else None

        shift_label = slot_map.get(emp.shift_slot_id)

        # For summary: count employee present if they have any present_rec in range
        if present_recs:
            summary.present += 1
        else:
            summary.absent += 1
        if late_days:
            summary.late += 1

        rows.append(AttendanceReportEmployee(
            emp_id=emp.emp_id,
            name=emp.name,
            shift_label=shift_label,
            total_days=total_days,
            present_days=len(present_recs),
            absent_days=absent_days,
            late_days=late_days,
            avg_hours_per_day=avg_hours,
        ))

    return AttendanceReportResponse(
        period=period,
        date_from=date_from,
        date_to=date_to,
        employees=rows,
        summary=summary,
    )


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

"""Leave CRUD service — casual/sick leave, no approval workflow.

Admin applies on the employee's behalf (employees have no login) and it
reflects immediately in attendance reports (see report_service.py's leave
overlay). Casual + sick share one combined 12-day/calendar-year quota.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.db.models.attendance import Employee
from app.core.db.models.leave import LeaveRequest, LeaveType
from app.modules.leaves.schemas import LeaveCreate, LeaveUpdate, ANNUAL_LEAVE_QUOTA


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _get_employee_by_emp_id(db: AsyncSession, emp_id: str) -> Employee:
    emp = (
        await db.execute(select(Employee).where(Employee.emp_id == emp_id))
    ).scalar_one_or_none()
    if not emp:
        raise HTTPException(status_code=404, detail=f"Employee '{emp_id}' not found.")
    return emp


async def _get_leave_by_id(db: AsyncSession, leave_id: UUID) -> LeaveRequest:
    leave = (
        await db.execute(
            select(LeaveRequest)
            .options(selectinload(LeaveRequest.employee))
            .where(LeaveRequest.id == leave_id)
        )
    ).scalar_one_or_none()
    if not leave:
        raise HTTPException(status_code=404, detail="Leave request not found.")
    return leave


def _working_days_count(emp: Employee, date_from: date, date_to: date) -> int:
    """Days in [date_from, date_to] that are NOT one of emp's own weekly-off days."""
    count = 0
    d = date_from
    while d <= date_to:
        if not emp.is_weekly_off(d):
            count += 1
        d += timedelta(days=1)
    return count


async def _year_used_days(
    db: AsyncSession, employee_id: UUID, year: int, exclude_id: Optional[UUID] = None
) -> int:
    """Sum of days_count for this employee's leave requests starting in `year`."""
    q = select(func.coalesce(func.sum(LeaveRequest.days_count), 0)).where(
        LeaveRequest.employee_id == employee_id,
        func.extract("year", LeaveRequest.date_from) == year,
    )
    if exclude_id is not None:
        q = q.where(LeaveRequest.id != exclude_id)
    return (await db.execute(q)).scalar() or 0


async def _has_overlap(
    db: AsyncSession,
    employee_id: UUID,
    date_from: date,
    date_to: date,
    exclude_id: Optional[UUID] = None,
) -> bool:
    q = select(LeaveRequest.id).where(
        LeaveRequest.employee_id == employee_id,
        LeaveRequest.date_from <= date_to,
        LeaveRequest.date_to >= date_from,
    )
    if exclude_id is not None:
        q = q.where(LeaveRequest.id != exclude_id)
    return (await db.execute(q)).first() is not None


def _check_quota(used: int, requested: int, year: int) -> None:
    if used + requested > ANNUAL_LEAVE_QUOTA:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient leave balance: {max(ANNUAL_LEAVE_QUOTA - used, 0)} day(s) "
                f"remaining for {year}, requested {requested}."
            ),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_leave(db: AsyncSession, payload: LeaveCreate) -> LeaveRequest:
    emp = await _get_employee_by_emp_id(db, payload.emp_id)

    if await _has_overlap(db, emp.id, payload.date_from, payload.date_to):
        raise HTTPException(
            status_code=409, detail="Overlaps an existing leave request for this employee."
        )

    days_count = _working_days_count(emp, payload.date_from, payload.date_to)
    if days_count == 0:
        raise HTTPException(
            status_code=422,
            detail="Leave range contains no working days (all fall on the employee's weekly-off).",
        )

    used = await _year_used_days(db, emp.id, payload.date_from.year)
    _check_quota(used, days_count, payload.date_from.year)

    leave = LeaveRequest(
        employee_id=emp.id,
        leave_type=LeaveType(payload.leave_type),
        date_from=payload.date_from,
        date_to=payload.date_to,
        days_count=days_count,
        reason=payload.reason,
    )
    db.add(leave)
    await db.commit()
    await db.refresh(leave)
    leave.employee = emp  # already in the session — avoids a reload for the response
    return leave


async def update_leave(db: AsyncSession, leave_id: UUID, payload: LeaveUpdate) -> LeaveRequest:
    leave = await _get_leave_by_id(db, leave_id)
    emp = leave.employee

    new_date_from = payload.date_from if payload.date_from is not None else leave.date_from
    new_date_to = payload.date_to if payload.date_to is not None else leave.date_to
    if new_date_to < new_date_from:
        raise HTTPException(status_code=422, detail="date_to cannot be before date_from")
    if new_date_from.year != new_date_to.year:
        raise HTTPException(
            status_code=422, detail="leave request must fall within a single calendar year"
        )

    if new_date_from != leave.date_from or new_date_to != leave.date_to:
        if await _has_overlap(db, emp.id, new_date_from, new_date_to, exclude_id=leave.id):
            raise HTTPException(
                status_code=409, detail="Overlaps an existing leave request for this employee."
            )

        new_days_count = _working_days_count(emp, new_date_from, new_date_to)
        if new_days_count == 0:
            raise HTTPException(
                status_code=422,
                detail="Leave range contains no working days (all fall on the employee's weekly-off).",
            )

        used = await _year_used_days(db, emp.id, new_date_from.year, exclude_id=leave.id)
        _check_quota(used, new_days_count, new_date_from.year)

        leave.date_from = new_date_from
        leave.date_to = new_date_to
        leave.days_count = new_days_count

    if payload.leave_type is not None:
        leave.leave_type = LeaveType(payload.leave_type)
    if payload.reason is not None:
        leave.reason = payload.reason

    await db.commit()
    await db.refresh(leave)
    leave.employee = emp
    return leave


async def get_leave(db: AsyncSession, leave_id: UUID) -> LeaveRequest:
    return await _get_leave_by_id(db, leave_id)


async def list_leaves(
    db: AsyncSession,
    page: int = 1,
    size: int = 20,
    emp_id: Optional[str] = None,
    leave_type: Optional[str] = None,
    year: Optional[int] = None,
) -> dict:
    q = select(LeaveRequest).options(selectinload(LeaveRequest.employee)).join(
        Employee, Employee.id == LeaveRequest.employee_id
    )
    if emp_id:
        q = q.where(Employee.emp_id == emp_id)
    if leave_type:
        q = q.where(LeaveRequest.leave_type == LeaveType(leave_type))
    if year:
        q = q.where(func.extract("year", LeaveRequest.date_from) == year)

    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar() or 0

    q = q.order_by(LeaveRequest.date_from.desc()).offset((page - 1) * size).limit(size)
    items = (await db.execute(q)).scalars().all()

    return {"items": items, "total": total, "page": page, "size": size}


async def get_balance_remaining(db: AsyncSession, employee_id: UUID, year: int) -> int:
    used = await _year_used_days(db, employee_id, year)
    return max(ANNUAL_LEAVE_QUOTA - used, 0)

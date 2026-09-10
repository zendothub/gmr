"""Leave CRUD API routes — casual/sick leave, admin-applied on an employee's
behalf (no approval workflow, reflects in attendance immediately).

  POST   /api/v1/leaves       — apply for leave
  GET    /api/v1/leaves       — list (paginated, filterable)
  GET    /api/v1/leaves/{id}  — single leave request
  PUT    /api/v1/leaves/{id}  — edit type/dates/reason
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.core.db.models.leave import LeaveRequest
from app.modules.leaves import service as leave_svc
from app.modules.leaves.schemas import LeaveCreate, LeaveUpdate, LeaveResponse

router = APIRouter(prefix="/api/v1/leaves", tags=["Leaves"])


async def _build_leave_response(db: AsyncSession, leave: LeaveRequest) -> LeaveResponse:
    balance = await leave_svc.get_balance_remaining(db, leave.employee_id, leave.date_from.year)
    return LeaveResponse(
        id=leave.id,
        employee_id=leave.employee_id,
        emp_id=leave.employee.emp_id,
        employee_name=leave.employee.name,
        leave_type=leave.leave_type.value,
        date_from=leave.date_from,
        date_to=leave.date_to,
        days_count=leave.days_count,
        reason=leave.reason,
        balance_remaining=balance,
        created_at=leave.created_at,
        updated_at=leave.updated_at,
    )


@router.post("", response_model=LeaveResponse, status_code=201)
async def apply_leave(
    payload: LeaveCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Apply for leave on an employee's behalf.

    - **leave_type**: `CASUAL` or `SICK`
    - **date_from** / **date_to**: inclusive range, must fall within one calendar year
    - **reason**: optional

    Casual + sick share one combined 12-day/calendar-year quota per employee.
    Rejected (422) if the range has no working days or exceeds the remaining
    balance; rejected (409) if it overlaps an existing leave for that employee.
    """
    leave = await leave_svc.create_leave(db, payload)
    return await _build_leave_response(db, leave)


@router.get("", response_model=dict)
async def list_leaves(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    emp_id: Optional[str] = Query(None, description="Filter by employee ID"),
    leave_type: Optional[str] = Query(None, pattern="^(CASUAL|SICK)$"),
    year: Optional[int] = Query(None, description="Filter by calendar year (date_from's year)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List leave requests (paginated)."""
    result = await leave_svc.list_leaves(
        db, page=page, size=size, emp_id=emp_id, leave_type=leave_type, year=year,
    )
    return {
        "items": [await _build_leave_response(db, l) for l in result["items"]],
        "total": result["total"],
        "page": result["page"],
        "size": result["size"],
    }


@router.get("/{leave_id}", response_model=LeaveResponse)
async def get_leave(
    leave_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single leave request by ID."""
    leave = await leave_svc.get_leave(db, leave_id)
    return await _build_leave_response(db, leave)


@router.put("/{leave_id}", response_model=LeaveResponse)
async def update_leave(
    leave_id: UUID,
    payload: LeaveUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Edit leave type, dates, or reason. Re-validates overlap and quota if dates change."""
    leave = await leave_svc.update_leave(db, leave_id, payload)
    return await _build_leave_response(db, leave)

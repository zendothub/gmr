"""Employee API routes.

Registration (2 ways):
  POST /api/v1/employees/register           — Way 1: upload image
  PUT  /api/v1/employees/register/by-camera — Way 2: link camera-detected identity

Employee CRUD:
  GET    /api/v1/employees                  — list (paginated)
  GET    /api/v1/employees/{emp_id}         — single employee
  PUT    /api/v1/employees/{emp_id}         — update name/shift
  DELETE /api/v1/employees/{emp_id}         — soft-delete

Attendance report:
  GET    /api/v1/employees/attendance/report
"""

from datetime import date
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.employees import service as emp_svc
from app.modules.employees import report_service
from app.modules.employees.schemas import (
    EmployeeResponse,
    EmployeeUpdate,
    RegisterByImageResponse,
    RegisterByCameraBody,
    AttendanceReportResponse,
)

router = APIRouter(prefix="/api/v1/employees", tags=["Employees"])


# ---------------------------------------------------------------------------
# Registration — Way 1: image upload
# ---------------------------------------------------------------------------

@router.post("/register", response_model=RegisterByImageResponse, status_code=200)
async def register_by_image(
    emp_id: str = Form(..., description="Unique employee ID / badge number"),
    name: str = Form(..., description="Employee full name"),
    shift_slot_id: Optional[UUID] = Form(None, description="UUID of the assigned shift slot"),
    image: UploadFile = File(..., description="Clear face photo (JPEG or PNG)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Register an employee by uploading a face image.

    **Flow:**
    1. InsightFace detects the face and checks quality.
    2. Extracts a 512-dim ArcFace embedding.
    3. Searches the database for a matching person identity.
       - **Match found** → updates face crop, returns `already_registered=true`
         with the crop URL so the frontend can preview the stored photo.
       - **No match** → creates a new `PersonIdentity` + `Employee` row.

    **Errors:**
    - `400` — image cannot be decoded
    - `422` — no face detected or quality too low
    - `409` — `emp_id` already used by another employee
    - `503` — face recognition model unavailable
    """
    if image.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Only JPEG and PNG are accepted.",
        )

    image_bytes = await image.read()
    result = await emp_svc.register_by_image(
        db, image_bytes, emp_id=emp_id, name=name, shift_slot_id=shift_slot_id
    )
    return RegisterByImageResponse(
        employee=EmployeeResponse.model_validate(result["employee"]),
        already_registered=result["already_registered"],
        face_crop_url=result["face_crop_url"],
        message=result["message"],
    )


# ---------------------------------------------------------------------------
# Registration — Way 2: camera-detected person
# ---------------------------------------------------------------------------

@router.put("/register/by-camera", response_model=EmployeeResponse, status_code=200)
async def register_by_camera(
    payload: RegisterByCameraBody,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Register an employee by linking an existing camera-detected identity.

    The frontend selects a person from the **Debug → Active Tracks** or
    **Debug → Unique Persons** view and submits their `person_identity_id`
    together with the employee's `emp_id`, `name`, and optional `shift_slot_id`.

    **Errors:**
    - `404` — `person_identity_id` not found in the database
    - `409` — identity already linked to another employee, or `emp_id` already used
    """
    employee = await emp_svc.register_by_camera(db, payload)
    return EmployeeResponse.model_validate(employee)


# ---------------------------------------------------------------------------
# Attendance report
# ---------------------------------------------------------------------------

@router.get("/attendance/report", response_model=AttendanceReportResponse)
async def get_attendance_report(
    period: str = Query(..., pattern="^(daily|weekly|monthly)$", description="Report period"),
    # daily
    date: Optional[date] = Query(None, description="[daily] Target date (YYYY-MM-DD)"),
    # weekly
    week_start: Optional[date] = Query(None, description="[weekly] First day of the week (YYYY-MM-DD)"),
    # monthly
    year: Optional[int] = Query(None, description="[monthly] Year e.g. 2026"),
    month: Optional[int] = Query(None, ge=1, le=12, description="[monthly] Month 1-12"),
    # filters
    shift_slot_id: Optional[UUID] = Query(None, description="Filter by shift slot"),
    emp_id: Optional[str] = Query(None, description="Filter by a single employee ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fetch an attendance report from the **materialised** `attendance_records` table.

    Data is consistent across multiple downloads — it is never recomputed from
    raw track sessions and is immune to background dedup merges.

    **Period-specific parameters:**
    | period  | required params |
    |---------|----------------|
    | daily   | `date`          |
    | weekly  | `week_start`    |
    | monthly | `year`, `month` |

    **Optional filters:** `shift_slot_id`, `emp_id`
    """
    if period == "daily":
        if not date:
            raise HTTPException(status_code=422, detail="`date` is required for period=daily.")
        return await report_service.get_daily_report(
            db, report_date=date, shift_slot_id=shift_slot_id, emp_id=emp_id
        )

    elif period == "weekly":
        if not week_start:
            raise HTTPException(status_code=422, detail="`week_start` is required for period=weekly.")
        return await report_service.get_weekly_report(
            db, week_start=week_start, shift_slot_id=shift_slot_id, emp_id=emp_id
        )

    elif period == "monthly":
        if not year or not month:
            raise HTTPException(status_code=422, detail="`year` and `month` are required for period=monthly.")
        return await report_service.get_monthly_report(
            db, year=year, month=month, shift_slot_id=shift_slot_id, emp_id=emp_id
        )

    raise HTTPException(status_code=422, detail="Invalid period.")


# ---------------------------------------------------------------------------
# Employee CRUD
# ---------------------------------------------------------------------------

@router.get("", response_model=dict)
async def list_employees(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by name or emp_id"),
    shift_slot_id: Optional[UUID] = Query(None),
    active_only: bool = Query(True),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List registered employees (paginated)."""
    result = await emp_svc.list_employees(
        db, page=page, size=size, search=search,
        shift_slot_id=shift_slot_id, active_only=active_only,
    )
    return {
        "items": [EmployeeResponse.model_validate(e) for e in result["items"]],
        "total": result["total"],
        "page": result["page"],
        "size": result["size"],
    }


@router.get("/{emp_id}", response_model=EmployeeResponse)
async def get_employee(
    emp_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single employee by employee ID."""
    return EmployeeResponse.model_validate(await emp_svc.get_by_emp_id(db, emp_id))


@router.put("/{emp_id}", response_model=EmployeeResponse)
async def update_employee(
    emp_id: str,
    payload: EmployeeUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update employee name, shift slot, or active status."""
    return EmployeeResponse.model_validate(
        await emp_svc.update_employee(db, emp_id, payload)
    )


@router.delete("/{emp_id}", status_code=204)
async def deactivate_employee(
    emp_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Soft-delete an employee (sets `is_active=False`).
    All attendance history is preserved.
    """
    await emp_svc.deactivate_employee(db, emp_id)

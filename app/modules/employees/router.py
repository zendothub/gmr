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

import re
from datetime import date, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from loguru import logger
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
    GENDER_PATTERN,
    validate_weekends,
)

router = APIRouter(prefix="/api/v1/employees", tags=["Employees"])


# ---------------------------------------------------------------------------
# Internal helper — resolve face_crop_path → presigned MinIO URL
# ---------------------------------------------------------------------------

def _resolve_face_crop_url(path: Optional[str]) -> Optional[str]:
    """
    Given a raw MinIO object path (e.g. "crops/emp_face_20260908_...jpg"),
    return a presigned GET URL valid for 1 hour.  Returns None on any error
    or when path is None/empty.
    """
    if not path:
        return None
    try:
        from app.modules.storage.minio_client import get_public_client, BUCKET_PREFIX
        client = get_public_client()
        clean = path.lstrip("/")
        if clean.startswith(f"{BUCKET_PREFIX}/"):
            object_name = clean[len(BUCKET_PREFIX) + 1:]
        else:
            object_name = clean
        url = client.presigned_get_object(
            bucket_name=BUCKET_PREFIX,
            object_name=object_name,
            expires=timedelta(hours=1),
        )
        return url
    except Exception as exc:
        logger.warning(f"_resolve_face_crop_url failed for path={path!r}: {exc}")
        return None


def _build_employee_response(emp, face_crop_url: Optional[str] = None) -> EmployeeResponse:
    """Build an EmployeeResponse from an Employee ORM object, injecting the presigned URL."""
    resp = EmployeeResponse.model_validate(emp)
    resp.face_crop_url = face_crop_url if face_crop_url is not None else _resolve_face_crop_url(emp.face_crop_path)
    return resp


# ---------------------------------------------------------------------------
# Registration — Way 1: image upload
# ---------------------------------------------------------------------------

@router.post("/register", response_model=RegisterByImageResponse, status_code=200)
async def register_by_image(
    emp_id: str = Form(..., description="Unique employee ID / badge number"),
    name: str = Form(..., description="Employee full name"),
    # Plain (non-Optional) types + empty-value defaults here, not Optional[...] = Form(None):
    # Swagger UI doesn't render an input box for multipart form fields whose schema is
    # `anyOf [type, null]` (what Optional[...] produces under OpenAPI 3.1). Empty string /
    # empty list defaults sidestep that while keeping the field genuinely optional.
    gender: str = Form("", description="MALE, FEMALE, or OTHER — leave blank if unknown"),
    weekends: List[str] = Form(
        [], description="Employee-specific weekly-off days, e.g. SATURDAY, SUNDAY"
    ),
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

    gender_val: Optional[str] = gender.strip().upper() or None
    if gender_val and not re.match(GENDER_PATTERN, gender_val):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid gender '{gender_val}'. Must be MALE, FEMALE, or OTHER.",
        )

    try:
        weekends_val = validate_weekends(weekends) if weekends else None
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    image_bytes = await image.read()
    result = await emp_svc.register_by_image(
        db, image_bytes, emp_id=emp_id, name=name, shift_slot_id=shift_slot_id,
        gender=gender_val, weekends=weekends_val,
    )
    face_crop_url = result["face_crop_url"]
    # Resolve to a presigned URL if the service returned a raw MinIO path
    if face_crop_url and not face_crop_url.startswith("http"):
        face_crop_url = _resolve_face_crop_url(face_crop_url)
    emp_resp = _build_employee_response(result["employee"], face_crop_url=face_crop_url)
    return RegisterByImageResponse(
        employee=emp_resp,
        already_registered=result["already_registered"],
        face_crop_url=face_crop_url,
        message=result["message"],
    )


# ---------------------------------------------------------------------------
# Registration — Way 2: camera-detected person
# ---------------------------------------------------------------------------

@router.put("/register/by-camera", response_model=RegisterByImageResponse, status_code=200)
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

    Returns the same `RegisterByImageResponse` shape as the image-upload endpoint
    (`{ employee, already_registered, face_crop_url, message }`) so the frontend
    can display the face crop immediately after linking.

    **Errors:**
    - `404` — `person_identity_id` not found in the database
    - `409` — identity already linked to another employee, or `emp_id` already used
    """
    employee = await emp_svc.register_by_camera(db, payload)
    emp_resp = _build_employee_response(employee)
    return RegisterByImageResponse(
        employee=emp_resp,
        already_registered=False,
        face_crop_url=emp_resp.face_crop_url,
        message=(
            f"Employee '{employee.name}' registered and linked to the existing identity."
        ),
    )


# ---------------------------------------------------------------------------
# Attendance report
# ---------------------------------------------------------------------------

@router.get("/attendance/report", response_model=AttendanceReportResponse)
async def get_attendance_report(
    period: Optional[str] = Query(
        None, pattern="^(daily|weekly|monthly)$",
        description="Report period. Required unless start_date/end_date is used instead.",
    ),
    # daily
    date: Optional[date] = Query(None, description="[daily] Target date (YYYY-MM-DD)"),
    # weekly
    week_start: Optional[date] = Query(None, description="[weekly] First day of the week (YYYY-MM-DD)"),
    # monthly
    year: Optional[int] = Query(None, description="[monthly] Year e.g. 2026"),
    month: Optional[int] = Query(None, ge=1, le=12, description="[monthly] Month 1-12"),
    # custom date range — alternative to period
    start_date: Optional[date] = Query(
        None, description="Custom range start (YYYY-MM-DD), inclusive. Alternative to `period`."
    ),
    end_date: Optional[date] = Query(
        None, description="Custom range end (YYYY-MM-DD), inclusive. Alternative to `period`."
    ),
    # filters
    shift_slot_id: Optional[UUID] = Query(None, description="Filter by shift slot"),
    emp_id: Optional[str] = Query(None, description="Filter by a single employee ID"),
    # sorting
    sort_by: Optional[str] = Query(
        None, pattern="^(present|absent|check_in|check_out)$",
        description="Sort field: present, absent, check_in, check_out",
    ),
    sort_order: str = Query(
        "asc", pattern="^(asc|desc)$", description="Sort direction: asc or desc",
    ),
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

    **Custom date range** — pass `start_date` and/or `end_date` (YYYY-MM-DD) instead of
    `period` to fetch attendance for an arbitrary inclusive range
    (`start_date <= attendance_date <= end_date`):
    - both given      → exact `[start_date, end_date]` window
    - only start_date → `[start_date, today]`
    - only end_date   → `[earliest recorded attendance_date, end_date]`

    **Optional filters:** `shift_slot_id`, `emp_id` — apply to both modes.

    **Sorting:** `sort_by` (`present`, `absent`, `check_in`, `check_out`) + `sort_order`
    (`asc`/`desc`, default `asc`). `check_in`/`check_out` are only valid for
    `period=daily` — weekly/monthly/custom rows have no per-day timestamps.
    """
    if start_date is not None and end_date is not None and start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date cannot be greater than end_date")

    is_daily = period == "daily" and start_date is None and end_date is None
    if sort_by in ("check_in", "check_out") and not is_daily:
        raise HTTPException(
            status_code=422,
            detail="sort_by=check_in/check_out is only valid for period=daily.",
        )

    if start_date is not None or end_date is not None:
        response = await report_service.get_range_report(
            db, start_date=start_date, end_date=end_date,
            shift_slot_id=shift_slot_id, emp_id=emp_id,
        )

    elif period == "daily":
        if not date:
            raise HTTPException(status_code=422, detail="`date` is required for period=daily.")
        response = await report_service.get_daily_report(
            db, report_date=date, shift_slot_id=shift_slot_id, emp_id=emp_id
        )

    elif period == "weekly":
        if not week_start:
            raise HTTPException(status_code=422, detail="`week_start` is required for period=weekly.")
        response = await report_service.get_weekly_report(
            db, week_start=week_start, shift_slot_id=shift_slot_id, emp_id=emp_id
        )

    elif period == "monthly":
        if not year or not month:
            raise HTTPException(status_code=422, detail="`year` and `month` are required for period=monthly.")
        response = await report_service.get_monthly_report(
            db, year=year, month=month, shift_slot_id=shift_slot_id, emp_id=emp_id
        )

    elif not period:
        raise HTTPException(
            status_code=422,
            detail="`period` is required unless start_date and/or end_date is provided.",
        )
    else:
        raise HTTPException(status_code=422, detail="Invalid period.")

    if sort_by:
        response.employees = report_service.sort_report_rows(
            response.employees, response.period, sort_by, sort_order
        )
    return response


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
        "items": [_build_employee_response(e) for e in result["items"]],
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
    return _build_employee_response(await emp_svc.get_by_emp_id(db, emp_id))


@router.put("/{emp_id}", response_model=EmployeeResponse)
async def update_employee(
    emp_id: str,
    payload: EmployeeUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update employee name, shift slot, or active status."""
    return _build_employee_response(await emp_svc.update_employee(db, emp_id, payload))


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

"""Employee Check-In event feed API.

Provides a paginated, descending-by-time feed of employee check-in events.
Each item includes the employee name, check-in time, date, shift info,
status (on_time / late), and a **presigned MinIO URL** for the profile image.

Profile image access fix
========================
The ``face_crop_path`` stored in the database is a raw MinIO object key
(e.g. ``crops/emp_face_20260910_...jpg``).  Browsers cannot fetch this
directly — they need a **presigned GET URL** with temporary auth credentials
baked into the query string.  This router resolves every ``face_crop_path``
to a 1-hour presigned URL via the MinIO public client before returning the
response.  If the path is missing, empty, or the MinIO client is unavailable,
``face_crop_url`` is returned as ``null`` so the frontend can show a fallback
avatar instead of a broken image.
"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.employees import checkin_service
from app.modules.employees.checkin_schemas import CheckInResponse, CheckInListResponse

router = APIRouter(prefix="/api/v1/checkins", tags=["Employee Check-Ins"])


# ---------------------------------------------------------------------------
# Profile image helper — resolves raw MinIO path → presigned GET URL
# ---------------------------------------------------------------------------

def _resolve_face_crop_url(path: Optional[str]) -> Optional[str]:
    """Convert a raw MinIO object path to a presigned GET URL (1 hour expiry).

    Why this is needed:
      • ``face_crop_path`` stores the internal object key, e.g.
        ``crops/emp_face_20260910_123456_abcd1234.jpg``
      • MinIO is not publicly accessible — browsers need a presigned URL that
        embeds temporary S3 credentials in the query string.
      • Without this conversion the <img src="..."> tag would point at a raw
        path the browser cannot reach, producing a broken image.

    Returns ``None`` when:
      • ``path`` is None or empty (employee has no face crop yet)
      • The MinIO client is not configured
      • Presigned URL generation fails for any reason (network, bucket missing, …)
    """
    if not path:
        return None
    try:
        from app.modules.storage.minio_client import get_public_client, BUCKET_PREFIX
        client = get_public_client()

        # Strip bucket prefix if accidentally included in the stored path
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
        logger.warning(f"[CheckIn] Failed to generate presigned URL for path={path!r}: {exc}")
        return None


def _build_checkin_response(checkin) -> CheckInResponse:
    """Build a CheckInResponse from an ORM object, injecting the presigned URL."""
    resp = CheckInResponse.model_validate(checkin)
    resp.face_crop_url = _resolve_face_crop_url(checkin.face_crop_path)
    return resp


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("", response_model=CheckInListResponse)
async def list_checkins(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List employee check-in events, **most recent first** (descending by checked_in_at).

    Each item contains:
    - **employee_name** — full name at the time of check-in
    - **emp_code** — employee ID / badge number
    - **checked_in_at** — exact UTC timestamp of the first detection
    - **check_in_date** — calendar date (accounts for night-shift roll-over)
    - **face_crop_url** — presigned MinIO URL for the profile image
      (``null`` if unavailable — frontend should show a fallback avatar)
    - **status** — ``on_time`` or ``late`` relative to shift start
    - **shift_label** — human-readable shift name (e.g. "11AM-7PM")

    Pagination: use ``page`` and ``page_size``. Response includes ``total``
    and ``total_pages`` for the frontend to build paging controls.
    """
    items, total = await checkin_service.get_checkins(
        db,
        page=page,
        page_size=page_size,
    )

    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0

    return CheckInListResponse(
        items=[_build_checkin_response(ci) for ci in items],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )

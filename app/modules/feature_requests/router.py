"""Feature Requests API routes."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.feature_requests.schemas import (
    FeatureRequestCreate,
    FeatureRequestUpdate,
    FeatureRequestActiveToggle,
    FeatureRequestResponse,
    FeatureRequestListResponse,
)
from app.modules.feature_requests.service import FeatureRequestService

router = APIRouter(prefix="/api/feature-requests", tags=["Feature Requests"])


@router.post("", response_model=FeatureRequestResponse, status_code=201)
async def create_feature_request(
    payload: FeatureRequestCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Submit a new feature request.

    An email notification is sent to the developer team (configured via SMTP_* env vars).
    Priority defaults to 'low'; pass 'high' for urgent requests.
    """
    fr = await FeatureRequestService.create(
        db,
        title=payload.title,
        description=payload.description,
        priority=payload.priority,
    )
    return FeatureRequestResponse.model_validate(fr)


@router.get("", response_model=FeatureRequestListResponse)
async def list_feature_requests(
    status: Optional[str] = Query(
        None, pattern=r"^(queued|in_progress|live)$",
        description="Filter by status"
    ),
    priority: Optional[str] = Query(
        None, pattern=r"^(low|high)$",
        description="Filter by priority"
    ),
    is_active: Optional[bool] = Query(
        None,
        description="Filter by active flag (true / false)"
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all feature requests, ordered newest-first.

    Optionally filter by `status`, `priority`, and/or `is_active`.
    """
    items, total = await FeatureRequestService.list_requests(
        db,
        status_filter=status,
        priority_filter=priority,
        is_active_filter=is_active,
        page=page,
        page_size=page_size,
    )
    return FeatureRequestListResponse(
        items=[FeatureRequestResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/live", response_model=FeatureRequestListResponse)
async def list_live_feature_requests(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Shortcut: fetch only feature requests with 'live' status, newest-first."""
    items, total = await FeatureRequestService.get_live_requests(
        db, page=page, page_size=page_size
    )
    return FeatureRequestListResponse(
        items=[FeatureRequestResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{feature_id}", response_model=FeatureRequestResponse)
async def get_feature_request(
    feature_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single feature request by ID."""
    fr = await FeatureRequestService.get_by_id(db, UUID(feature_id))
    return FeatureRequestResponse.model_validate(fr)


@router.patch("/{feature_id}", response_model=FeatureRequestResponse)
async def update_feature_request(
    feature_id: str,
    payload: FeatureRequestUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Updates status, forecast_message, and/or priority on a feature request.

    **Auto-activation rule**: when `status` is set to `"live"`, `is_active` is
    automatically set to `true` — no extra call needed.
    """
    fr = await FeatureRequestService.update(
        db,
        UUID(feature_id),
        status=payload.status,
        forecast_message=payload.forecast_message,
        priority=payload.priority,
    )
    return FeatureRequestResponse.model_validate(fr)


@router.patch("/{feature_id}/active", response_model=FeatureRequestResponse)
async def toggle_feature_request_active(
    feature_id: str,
    payload: FeatureRequestActiveToggle,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Manually toggles is_active for a feature request.

    Use this to deactivate a live feature or re-activate a previously disabled one
    without changing its status.

    - `{ "is_active": false }` → deactivate
    - `{ "is_active": true }`  → activate
    """
    fr = await FeatureRequestService.set_active(db, UUID(feature_id), payload.is_active)
    return FeatureRequestResponse.model_validate(fr)


@router.delete("/{feature_id}", status_code=204)
async def delete_feature_request(
    feature_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Hard-delete a single feature request by ID.

    Works regardless of status (queued / in_progress / live).
    Returns 204 No Content on success.
    """
    await FeatureRequestService.delete_by_id(db, UUID(feature_id))


@router.delete("", status_code=200)
async def delete_all_feature_requests(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Hard-delete ALL feature requests regardless of status.

    ⚠️ This is a destructive, irreversible operation.
    Returns the count of deleted records.
    """
    deleted = await FeatureRequestService.delete_all(db)
    return {"deleted": deleted, "message": f"{deleted} feature request(s) permanently deleted"}

"""Feature Requests API routes."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db
from app.modules.feature_requests.schemas import (
    FeatureRequestCreate,
    FeatureRequestUpdate,
    FeatureRequestResponse,
    FeatureRequestListResponse,
)
from app.modules.feature_requests.service import FeatureRequestService

router = APIRouter(prefix="/api/feature-requests", tags=["Feature Requests"])


@router.post("", response_model=FeatureRequestResponse, status_code=201)
async def create_feature_request(
    payload: FeatureRequestCreate,
    db: AsyncSession = Depends(get_db),
):
    """Submit a new feature request from the admin dashboard.

    An email notification is sent to the developer team (configured via SMTP_* env vars).
    """
    fr = await FeatureRequestService.create(db, payload.title, payload.description)
    return FeatureRequestResponse.model_validate(fr)


@router.get("", response_model=FeatureRequestListResponse)
async def list_feature_requests(
    status: Optional[str] = Query(
        None, pattern=r"^(queued|in_progress|live)$",
        description="Filter by status"
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List all feature requests, ordered newest-first.

    Optionally filter by `status` (queued / in_progress / live).
    """
    items, total = await FeatureRequestService.list_requests(
        db, status_filter=status, page=page, page_size=page_size
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
):
    """Get a single feature request by ID."""
    fr = await FeatureRequestService.get_by_id(db, UUID(feature_id))
    return FeatureRequestResponse.model_validate(fr)


@router.patch("/{feature_id}", response_model=FeatureRequestResponse)
async def update_feature_request(
    feature_id: str,
    payload: FeatureRequestUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Developer updates status and/or forecast_message on a feature request."""
    fr = await FeatureRequestService.update(
        db,
        UUID(feature_id),
        status=payload.status,
        forecast_message=payload.forecast_message,
    )
    return FeatureRequestResponse.model_validate(fr)
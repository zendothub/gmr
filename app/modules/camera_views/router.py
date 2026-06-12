"""Camera View API routes."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.camera_views.schemas import CameraViewCreate, CameraViewUpdate, CameraViewResponse
from app.modules.camera_views.service import CameraViewService

router = APIRouter(tags=["Camera Views"])


@router.post("/api/cameras/{camera_id}/views", response_model=CameraViewResponse, status_code=201)
async def create_camera_view(
    camera_id: UUID,
    data: CameraViewCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new view/ROI for a camera."""
    view = await CameraViewService.create_view(db, camera_id, data)
    return CameraViewResponse.model_validate(view)


@router.get("/api/cameras/{camera_id}/views", response_model=List[CameraViewResponse])
async def list_camera_views(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all views for a camera."""
    views = await CameraViewService.get_views_for_camera(db, camera_id)
    return [CameraViewResponse.model_validate(v) for v in views]


@router.get("/api/camera-views/{view_id}", response_model=CameraViewResponse)
async def get_camera_view(
    view_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a camera view by ID."""
    view = await CameraViewService.get_view(db, view_id)
    return CameraViewResponse.model_validate(view)


@router.put("/api/camera-views/{view_id}", response_model=CameraViewResponse)
async def update_camera_view(
    view_id: UUID,
    data: CameraViewUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update a camera view."""
    view = await CameraViewService.update_view(db, view_id, data)
    return CameraViewResponse.model_validate(view)


@router.delete("/api/camera-views/{view_id}")
async def delete_camera_view(
    view_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a camera view."""
    return await CameraViewService.delete_view(db, view_id)


@router.post("/api/camera-views/{view_id}/set-default", response_model=CameraViewResponse)
async def set_default_view(
    view_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Set a camera view as the default."""
    view = await CameraViewService.set_default_view(db, view_id)
    return CameraViewResponse.model_validate(view)

"""Camera API routes."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.cameras.schemas import (
    RTSPTestRequest, RTSPTestResponse, CameraCreate, CameraUpdate,
    CameraResponse, CameraHealthResponse,
)
from app.modules.cameras.service import CameraService

router = APIRouter(prefix="/api/cameras", tags=["Cameras"])


@router.post("/test-rtsp", response_model=RTSPTestResponse)
async def test_rtsp(
    data: RTSPTestRequest,
    current_user: User = Depends(get_current_user),
):
    """Test RTSP stream connectivity before adding camera."""
    return CameraService.test_rtsp_stream(data.rtsp_url)


@router.post("", response_model=CameraResponse, status_code=201)
async def create_camera(
    data: CameraCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Add a new RTSP camera (tests stream before saving)."""
    camera = await CameraService.create_camera(db, data)
    return CameraResponse.model_validate(camera)


@router.get("", response_model=List[CameraResponse])
async def list_cameras(
    store_id: Optional[UUID] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all cameras."""
    cameras = await CameraService.get_cameras(db, store_id)
    return [CameraResponse.model_validate(c) for c in cameras]


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get camera details by ID."""
    camera = await CameraService.get_camera(db, camera_id)
    return CameraResponse.model_validate(camera)


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: UUID,
    data: CameraUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update camera settings."""
    camera = await CameraService.update_camera(db, camera_id, data)
    return CameraResponse.model_validate(camera)


@router.delete("/{camera_id}")
async def delete_camera(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a camera."""
    return await CameraService.delete_camera(db, camera_id)


@router.post("/{camera_id}/start", response_model=CameraResponse)
async def start_camera(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start camera streaming and AI processing."""
    camera = await CameraService.start_camera(db, camera_id)
    return CameraResponse.model_validate(camera)


@router.post("/{camera_id}/stop", response_model=CameraResponse)
async def stop_camera(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Stop camera streaming and AI processing."""
    camera = await CameraService.stop_camera(db, camera_id)
    return CameraResponse.model_validate(camera)


@router.get("/{camera_id}/health", response_model=CameraHealthResponse)
async def camera_health(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get camera health and streaming status."""
    return await CameraService.get_camera_health(db, camera_id)

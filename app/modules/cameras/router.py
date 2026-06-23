"""Camera API routes."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, Request
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
    """Test RTSP stream connectivity before adding a camera.

    Opens the RTSP URL with OpenCV, grabs one frame, and returns resolution/fps.
    Does NOT create a camera — use this to validate the URL before saving.
    """
    return CameraService.test_rtsp_stream(data.rtsp_url)


@router.post("", response_model=CameraResponse, status_code=201)
async def create_camera(
    request: Request,
    data: CameraCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Add a new RTSP camera.

    Saves the camera and returns it with MediaMTX feed URLs
    (webrtc_url / hls_url) built against the request's own host, so LAN
    clients get the correct IP rather than ``localhost``.
    """
    camera = await CameraService.create_camera(db, data)
    return CameraService.build_response(camera, public_host=request.url.hostname)


@router.get("", response_model=List[CameraResponse])
async def list_cameras(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all cameras with live MediaMTX feed URLs."""
    cameras = await CameraService.get_cameras(db)
    return [CameraService.build_response(c, public_host=request.url.hostname) for c in cameras]


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    request: Request,
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get camera details by ID (including feed URLs)."""
    camera = await CameraService.get_camera(db, camera_id)
    return CameraService.build_response(camera, public_host=request.url.hostname)


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    request: Request,
    camera_id: UUID,
    data: CameraUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update camera settings."""
    camera = await CameraService.update_camera(db, camera_id, data)
    return CameraService.build_response(camera, public_host=request.url.hostname)


@router.delete("/{camera_id}")
async def delete_camera(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a camera and all its zones."""
    return await CameraService.delete_camera(db, camera_id)


@router.post("/{camera_id}/start", response_model=CameraResponse)
async def start_camera(
    request: Request,
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start camera AI processing.

    After this call the backend begins pulling RTSP, running YOLO detection,
    tracking, and evaluating rules. The stream is also published to MediaMTX so
    the frontend can play the live feed via webrtc_url or hls_url.
    """
    camera = await CameraService.start_camera(db, camera_id)
    return CameraService.build_response(camera, public_host=request.url.hostname)


@router.post("/{camera_id}/stop", response_model=CameraResponse)
async def stop_camera(
    request: Request,
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Stop camera AI processing.

    RTSP connection is released and MediaMTX publishing stops (stream is reaped
    after STREAM_IDLE_TIMEOUT_SECONDS). No further detections or events are
    generated until the camera is started again.
    """
    camera = await CameraService.stop_camera(db, camera_id)
    return CameraService.build_response(camera, public_host=request.url.hostname)


@router.get("/{camera_id}/health", response_model=CameraHealthResponse)
async def camera_health(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Read-only health check for a camera (does NOT start or stop anything).

    Reports whether the camera worker is actively streaming (is_streaming),
    its current_fps, uptime_seconds, and any error_message from the worker.
    Use this to detect silent failures on a running camera.
    """
    return await CameraService.get_camera_health(db, camera_id)
"""Streaming API routes.

Zone-binding flow (per camera):
  1. POST /api/cameras/{id}/stream/start  -> returns WebRTC/HLS URLs (live preview)
  2. GET  /api/cameras/{id}/stream/snapshot -> JPEG still for drawing polygons
  3. POST /api/cameras/{id}/zones (zones module) -> save the polygon + zone type
  4. POST /api/cameras/{id}/stream/stop   -> detach viewer (auto-stops when idle)
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.streaming.schemas import StreamEndpointsResponse, StreamStatusResponse
from app.modules.streaming.service import StreamingService

router = APIRouter(prefix="/api/cameras", tags=["Streaming"])


@router.post("/{camera_id}/stream/start", response_model=StreamEndpointsResponse)
async def start_stream(
    request: Request,
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Start the live WebRTC/HLS preview.

    Returns URLs built against the requesting host so the frontend on another
    LAN laptop gets the correct IP (e.g. ``http://10.251.39.75:8889/...``)
    instead of ``localhost``.
    """
    return await StreamingService.start_stream(db, camera_id, public_host=request.url.hostname)


@router.post("/{camera_id}/stream/stop")
async def stop_stream(
    camera_id: UUID,
    force: bool = Query(False, description="Force-stop even if other viewers remain"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Detach a viewer; the stream auto-stops once idle (or force-stop now)."""
    return await StreamingService.stop_stream(db, camera_id, force=force)


@router.get("/{camera_id}/stream/status", response_model=StreamStatusResponse)
async def stream_status(
    request: Request,
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get publish status and playback URLs for a camera's preview.

    URLs are built against the requesting host so LAN clients see the correct IP.
    """
    return await StreamingService.get_status(db, camera_id, public_host=request.url.hostname)


@router.get(
    "/{camera_id}/stream/snapshot",
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def stream_snapshot(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return one JPEG frame from the camera for the polygon-drawing canvas.

    The `X-Frame-Width` / `X-Frame-Height` headers give the pixel dimensions so
    the frontend can map clicked polygon points back to source coordinates.
    """
    jpeg_bytes, width, height = await StreamingService.snapshot(db, camera_id)
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "X-Frame-Width": str(width),
            "X-Frame-Height": str(height),
            "Cache-Control": "no-store",
        },
    )
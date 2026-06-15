"""Streaming service - ties cameras to the StreamManager and snapshots."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException
from loguru import logger

from app.core.db.models.camera import Camera
from app.modules.streaming.manager import StreamManager
from app.modules.streaming.mediamtx import MediaMTXManager
from app.modules.streaming.snapshot import grab_snapshot_jpeg
from app.modules.streaming.schemas import (
    StreamEndpointsResponse, StreamStatusResponse,
)


class StreamingService:

    @staticmethod
    async def _get_camera(db: AsyncSession, camera_id: UUID) -> Camera:
        result = await db.execute(select(Camera).where(Camera.id == camera_id))
        camera = result.scalar_one_or_none()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        return camera

    @staticmethod
    async def start_stream(
        db: AsyncSession, camera_id: UUID, public_host: str | None = None,
    ) -> StreamEndpointsResponse:
        """Start (or attach a viewer to) the live preview for a camera."""
        camera = await StreamingService._get_camera(db, camera_id)
        manager = StreamManager.get_instance()
        # run the (blocking) ffmpeg spawn off the event loop
        import anyio
        endpoints = await anyio.to_thread.run_sync(
            manager.add_viewer, camera.id, camera.rtsp_url, None, public_host,
        )
        return StreamEndpointsResponse(
            camera_id=camera.id,
            path=endpoints.path,
            webrtc_url=endpoints.webrtc_url,
            hls_url=endpoints.hls_url,
            rtsp_url=endpoints.rtsp_url,
        )

    @staticmethod
    async def stop_stream(db: AsyncSession, camera_id: UUID, force: bool = False) -> dict:
        """Detach a viewer (or force-stop) the preview for a camera."""
        await StreamingService._get_camera(db, camera_id)
        manager = StreamManager.get_instance()
        if force:
            stopped = manager.stop_stream(camera_id)
            return {"message": "Stream stopped" if stopped else "Stream was not running"}
        manager.remove_viewer(camera_id)
        return {"message": "Viewer removed; stream will stop when idle"}

    @staticmethod
    async def get_status(
        db: AsyncSession, camera_id: UUID, public_host: str | None = None,
    ) -> StreamStatusResponse:
        """Get live publish status; reflects endpoints even when not yet started."""
        camera = await StreamingService._get_camera(db, camera_id)
        manager = StreamManager.get_instance()
        status = manager.get_status(camera_id, public_host=public_host)
        if status:
            ep = status["endpoints"]
            return StreamStatusResponse(
                camera_id=camera.id,
                is_publishing=status["is_publishing"],
                viewers=status["viewers"],
                uptime_seconds=status["uptime_seconds"],
                last_error=status["last_error"],
                webrtc_url=ep["webrtc_url"],
                hls_url=ep["hls_url"],
                rtsp_url=ep["rtsp_url"],
            )
        # Not running yet - still surface the URLs it WILL have.
        endpoints = MediaMTXManager().endpoints(camera.id, public_host=public_host)
        return StreamStatusResponse(
            camera_id=camera.id,
            is_publishing=False,
            viewers=0,
            uptime_seconds=0.0,
            webrtc_url=endpoints.webrtc_url,
            hls_url=endpoints.hls_url,
            rtsp_url=endpoints.rtsp_url,
        )

    @staticmethod
    async def snapshot(db: AsyncSession, camera_id: UUID):
        """Grab one JPEG frame for the zone-drawing canvas. Returns (bytes, w, h)."""
        camera = await StreamingService._get_camera(db, camera_id)
        import anyio
        result = await anyio.to_thread.run_sync(grab_snapshot_jpeg, camera.rtsp_url)
        if result is None:
            raise HTTPException(
                status_code=502,
                detail="Could not grab a frame from the camera stream.",
            )
        return result

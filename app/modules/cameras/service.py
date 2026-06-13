"""Camera service - RTSP test, CRUD, start/stop, health."""

from typing import List, Optional
from uuid import UUID

import cv2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status
from loguru import logger

from app.core.db.models.camera import Camera, CameraStatus
from app.modules.cameras.schemas import (
    CameraCreate, CameraUpdate, CameraResponse, RTSPTestResponse, CameraHealthResponse
)


class CameraService:

    @staticmethod
    def test_rtsp_stream(rtsp_url: str, timeout: int = 10) -> RTSPTestResponse:
        """
        Test RTSP stream connectivity using OpenCV.

        Args:
            rtsp_url: RTSP URL to test
            timeout: Connection timeout in seconds
        """
        cap = None
        try:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)

            if not cap.isOpened():
                return RTSPTestResponse(
                    success=False,
                    message="Failed to open RTSP stream. Check URL and camera status.",
                )

            ret, frame = cap.read()
            if not ret or frame is None:
                return RTSPTestResponse(
                    success=False,
                    message="Connected but failed to read frame from stream.",
                )

            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 0

            return RTSPTestResponse(
                success=True,
                message="RTSP stream is accessible and returning frames.",
                resolution=f"{w}x{h}",
                fps=round(fps, 2),
            )

        except Exception as e:
            logger.error(f"RTSP test failed for {rtsp_url}: {e}")
            return RTSPTestResponse(
                success=False,
                message=f"RTSP test error: {str(e)}",
            )
        finally:
            if cap is not None:
                cap.release()

    @staticmethod
    async def create_camera(db: AsyncSession, data: CameraCreate) -> Camera:
        """Create a new camera after RTSP validation."""
        # Test RTSP first
        test_result = CameraService.test_rtsp_stream(data.rtsp_url)
        if not test_result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"RTSP stream test failed: {test_result.message}",
            )

        camera = Camera(
            name=data.name,
            rtsp_url=data.rtsp_url,
            store_id=data.store_id,
            role=data.role,
            fps_target=data.fps_target,
            resolution=test_result.resolution or data.resolution,
            detection_model=data.detection_model,
            reid_enabled=data.reid_enabled,
            demographic_enabled=data.demographic_enabled,
            frame_rotation=data.frame_rotation,
            location_description=data.location_description,
            status=CameraStatus.INACTIVE,
            is_active=True,
        )
        db.add(camera)
        await db.flush()
        await db.refresh(camera)
        logger.info(f"Camera created: {camera.name} (id={camera.id})")
        return camera

    @staticmethod
    async def get_cameras(db: AsyncSession, store_id: Optional[UUID] = None) -> List[Camera]:
        """List all cameras, optionally filtered by store."""
        query = select(Camera)
        if store_id:
            query = query.where(Camera.store_id == store_id)
        query = query.order_by(Camera.created_at.desc())
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_camera(db: AsyncSession, camera_id: UUID) -> Camera:
        """Get camera by ID."""
        result = await db.execute(select(Camera).where(Camera.id == camera_id))
        camera = result.scalar_one_or_none()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        return camera

    @staticmethod
    async def update_camera(db: AsyncSession, camera_id: UUID, data: CameraUpdate) -> Camera:
        """Update camera settings."""
        camera = await CameraService.get_camera(db, camera_id)
        update_data = data.model_dump(exclude_unset=True)

        # If RTSP URL changed, test it first
        if "rtsp_url" in update_data and update_data["rtsp_url"] != camera.rtsp_url:
            test_result = CameraService.test_rtsp_stream(update_data["rtsp_url"])
            if not test_result.success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"New RTSP stream test failed: {test_result.message}",
                )

        for key, value in update_data.items():
            setattr(camera, key, value)

        await db.flush()
        await db.refresh(camera)
        logger.info(f"Camera updated: {camera.name} (id={camera.id})")
        return camera

    @staticmethod
    async def delete_camera(db: AsyncSession, camera_id: UUID) -> dict:
        """Delete a camera."""
        camera = await CameraService.get_camera(db, camera_id)
        await db.delete(camera)
        logger.info(f"Camera deleted: {camera.name} (id={camera_id})")
        return {"message": f"Camera '{camera.name}' deleted successfully"}

    @staticmethod
    async def start_camera(db: AsyncSession, camera_id: UUID) -> Camera:
        """Mark camera as active and start worker."""
        camera = await CameraService.get_camera(db, camera_id)
        camera.status = CameraStatus.ACTIVE
        await db.flush()
        await db.refresh(camera)

        # Start camera worker
        try:
            from app.modules.ai_runtime.worker_supervisor import WorkerSupervisor
            supervisor = WorkerSupervisor.get_instance()
            if supervisor:
                await supervisor.start_camera(camera_id)
        except Exception as e:
            logger.warning(f"Could not start camera worker: {e}")

        logger.info(f"Camera started: {camera.name}")
        return camera

    @staticmethod
    async def stop_camera(db: AsyncSession, camera_id: UUID) -> Camera:
        """Mark camera as inactive and stop worker."""
        camera = await CameraService.get_camera(db, camera_id)
        camera.status = CameraStatus.INACTIVE
        await db.flush()
        await db.refresh(camera)

        # Stop camera worker
        try:
            from app.modules.ai_runtime.worker_supervisor import WorkerSupervisor
            supervisor = WorkerSupervisor.get_instance()
            if supervisor:
                await supervisor.stop_camera(camera_id)
        except Exception as e:
            logger.warning(f"Could not stop camera worker: {e}")

        logger.info(f"Camera stopped: {camera.name}")
        return camera

    @staticmethod
    async def get_camera_health(db: AsyncSession, camera_id: UUID) -> CameraHealthResponse:
        """Get camera health status."""
        camera = await CameraService.get_camera(db, camera_id)

        is_streaming = False
        current_fps = None
        uptime_seconds = None
        error_message = None

        try:
            from app.modules.ai_runtime.worker_supervisor import WorkerSupervisor
            supervisor = WorkerSupervisor.get_instance()
            if supervisor:
                worker_status = supervisor.get_worker_status(camera_id)
                if worker_status:
                    is_streaming = worker_status.get("is_running", False)
                    current_fps = worker_status.get("current_fps")
                    uptime_seconds = worker_status.get("uptime_seconds")
                    error_message = worker_status.get("error_message")
        except Exception as e:
            error_message = str(e)

        return CameraHealthResponse(
            camera_id=camera.id,
            name=camera.name,
            status=camera.status,
            is_streaming=is_streaming,
            current_fps=current_fps,
            uptime_seconds=uptime_seconds,
            error_message=error_message,
        )

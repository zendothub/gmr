"""Camera service - RTSP test, CRUD, start/stop, health."""

from typing import List, Optional
from uuid import UUID

import cv2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status
from loguru import logger

from app.core.db.models.camera import Camera, CameraStatus
from app.core.db.models.store import Store
from app.modules.cameras.schemas import (
    CameraCreate, CameraUpdate, CameraResponse, RTSPTestResponse, CameraHealthResponse
)
from app.modules.streaming.mediamtx import MediaMTXManager, camera_path
from app.modules.streaming.manager import StreamManager
from app.modules.rule_engine.config_loader import load_camera_config
import anyio


class CameraService:

    @staticmethod
    def build_response(camera: Camera, public_host: Optional[str] = None) -> CameraResponse:
        """Serialize a camera + attach its MediaMTX feed URLs (WebRTC/HLS).

        Feed URLs are derived from the camera's stream_path so the frontend can
        play the MediaMTX-published stream without an extra round-trip.

        ``public_host`` is the host the API request arrived on (e.g. the LAN IP
        ``192.168.1.158``); the WebRTC/HLS feed is built on that same host so it
        works for remote browsers instead of resolving to ``localhost``.
        """
        resp = CameraResponse.model_validate(camera)
        endpoints = MediaMTXManager().endpoints(camera.id, public_host=public_host)
        resp.stream_path = camera.stream_path or endpoints.path
        resp.webrtc_url = endpoints.webrtc_url
        resp.hls_url = endpoints.hls_url
        # Populate store-derived fields if a store is linked
        if camera.store:
            resp.store_name = camera.store.name
            resp.store_zone_gate = camera.store.zone_gate
        # Populate store-zone (position) fields if linked
        if camera.store_zone:
            resp.zone_id = camera.zone_id
            resp.zone_name = camera.store_zone.name
        return resp


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
    async def create_camera_v2(db: AsyncSession, data: "CameraCreateV2") -> Camera:
        """Create a new camera linked to a store (V2).

        Same logic as create_camera but store_id is required.
        The store must exist — validated by the caller (router).
        """
        from app.modules.cameras.schemas import CameraCreateV2

        resolution = None
        test_result = None
        if not data.skip_rtsp_test:
            test_result = CameraService.test_rtsp_stream(data.rtsp_url)
            if not test_result.success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"RTSP stream test failed: {test_result.message}",
                )
            resolution = test_result.resolution

        camera_kwargs: dict = {
            "name": data.name,
            "rtsp_url": data.rtsp_url,
            "store_id": data.store_id,
            "status": CameraStatus.INACTIVE,
            "is_active": True,
        }
        if data.zone_id:
            camera_kwargs["zone_id"] = data.zone_id
        if resolution:
            camera_kwargs["resolution"] = resolution
        camera = Camera(**camera_kwargs)

        db.add(camera)
        await db.flush()
        camera.stream_path = camera_path(camera.id)
        await db.flush()
        await db.refresh(camera)

        logger.info(f"Camera created (v2): {camera.name} (id={camera.id}, store_id={camera.store_id})")

        # Do NOT auto-start the StreamManager publisher here — the router
        # calls start_camera() immediately after, which handles stream setup:
        #   - burnin_enabled=True  → StreamBroadcaster in CameraWorker pushes to MediaMTX
        #   - burnin_enabled=False → StreamManager.add_viewer() pulls RTSP → MediaMTX
        # Starting both simultaneously causes dual-ffmpeg conflicts (exit code 224)
        # because most cameras/NVRs only allow one RTSP connection.

        return camera

    @staticmethod
    async def create_camera(db: AsyncSession, data: CameraCreate) -> Camera:
        """Create a new camera (name + rtsp_url + area). RTSP probe is optional."""
        resolution = None
        test_result = None
        if not data.skip_rtsp_test:
            test_result = CameraService.test_rtsp_stream(data.rtsp_url)
            if not test_result.success:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"RTSP stream test failed: {test_result.message}",
                )
            resolution = test_result.resolution

        # Only name + rtsp_url + area + zone + store are sent by the frontend.
        # All AI-config fields (fps_target, detection_model, reid_enabled, ...)
        # use the model's column defaults — never exposed to the client.
        camera_kwargs: dict = {
            "name": data.name,
            "rtsp_url": data.rtsp_url,
            "area_id": data.area_id,
            "status": CameraStatus.INACTIVE,
            "is_active": True,
        }
        if hasattr(data, "zone_id") and data.zone_id:
            camera_kwargs["zone_id"] = data.zone_id
        # If store_id is provided, validate and link store
        if hasattr(data, "store_id") and data.store_id:
            store_result = await db.execute(select(Store).where(Store.id == data.store_id))
            store = store_result.scalar_one_or_none()
            if not store:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Store not found: {data.store_id}",
                )
            camera_kwargs["store_id"] = data.store_id
        # Auto-detected resolution from RTSP probe (overrides model default).
        if resolution:
            camera_kwargs["resolution"] = resolution
        camera = Camera(**camera_kwargs)

        db.add(camera)
        await db.flush()
        # Persist the deterministic MediaMTX feed path now that we have the id.
        camera.stream_path = camera_path(camera.id)
        await db.flush()
        await db.refresh(camera)

        logger.info(f"Camera created: {camera.name} (id={camera.id})")

        # Auto-start the stream publisher (ffmpeg → MediaMTX) so the WebRTC URL
        # returned in the response works immediately — the frontend can play the
        # live feed right after "Add Camera" without an extra start call.
        try:
            manager = StreamManager.get_instance()
            import anyio
            await anyio.to_thread.run_sync(
                manager.add_viewer, camera.id, camera.rtsp_url
            )
            logger.info(f"Stream publisher auto-started for new camera {camera.id}")
        except Exception as e:
            logger.warning(f"Could not auto-start stream publisher for new camera: {e}")

        return camera

    @staticmethod
    async def get_cameras(db: AsyncSession) -> List[Camera]:
        """List all cameras."""
        query = select(Camera).order_by(Camera.created_at.desc())
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
    async def _teardown_resources(camera_id: UUID) -> None:
        """Stop the camera worker and stream publisher (ffmpeg + watchdog).

        Best-effort: failures are logged as warnings but never raised so
        callers (stop_camera, delete_camera) can continue with their own
        cleanup (status update / DB delete).
        """
        # Stop camera worker (AI processing).
        try:
            from app.modules.ai_runtime.worker_supervisor import WorkerSupervisor
            supervisor = WorkerSupervisor.get_instance()
            if supervisor:
                await supervisor.stop_camera(camera_id)
        except Exception as e:
            logger.warning(f"Could not stop camera worker: {e}")

        # Stop MediaMTX stream publisher (force-stop releases WHEP path +
        # terminates ffmpeg subprocess + allows watchdog to exit).
        try:
            manager = StreamManager.get_instance()
            manager.stop_stream(camera_id)
            logger.info(f"Stream publisher stopped for camera {camera_id}")
        except Exception as e:
            logger.warning(f"Could not stop stream publisher: {e}")

    @staticmethod
    async def delete_camera(db: AsyncSession, camera_id: UUID) -> dict:
        """Delete a camera.

        Stops the camera worker + stream publisher (ffmpeg + watchdog) first,
        then removes the database row.

        Also cleans up referencing rows in person_face_embeddings and
        person_embeddings — those FKs lack ON DELETE CASCADE, so PostgreSQL
        would otherwise block the deletion with a foreign-key violation.
        """
        from app.core.db.models.person import PersonFaceEmbedding, PersonEmbedding
        from sqlalchemy import update

        camera = await CameraService.get_camera(db, camera_id)

        # Tear down worker and stream publisher so no orphaned subprocesses
        # or watchdog threads linger after the DB row is gone.
        await CameraService._teardown_resources(camera_id)

        # Nullify or remove embedding rows that still reference this camera.
        # person_face_embeddings.camera_id and person_embeddings.camera_id have
        # ForeignKey("cameras.id") without ON DELETE, so we must unlink them first.
        await db.execute(
            update(PersonFaceEmbedding)
            .where(PersonFaceEmbedding.camera_id == camera_id)
            .values(camera_id=None)
        )
        await db.execute(
            update(PersonEmbedding)
            .where(PersonEmbedding.camera_id == camera_id)
            .values(camera_id=None)
        )

        await db.delete(camera)
        await db.flush()
        logger.info(f"Camera deleted: {camera.name} (id={camera_id})")
        return {"message": f"Camera '{camera.name}' deleted successfully"}

    @staticmethod
    async def start_camera(db: AsyncSession, camera_id: UUID) -> Camera:
        """Mark camera as active and start worker."""
        camera = await CameraService.get_camera(db, camera_id)
        logger.info(
            ">>> start_camera: BEGIN"
            f" | camera_id={camera.id}"
            f" | name='{camera.name}'"
            f" | current_status={camera.status}"
            f" | burnin_enabled={camera.burnin_enabled}"
        )

        camera.status = CameraStatus.ACTIVE
        await db.flush()
        await db.refresh(camera)
        logger.info(f">>> start_camera: status set to ACTIVE for {camera.id}")

        # Start camera worker (YOLO detection + ByteTrack tracking + rule evaluation)
        logger.info(f">>> start_camera: launching camera worker for {camera.id}...")
        try:
            from app.modules.ai_runtime.worker_supervisor import WorkerSupervisor
            supervisor = WorkerSupervisor.get_instance()
            if supervisor:
                await supervisor.start_camera(camera_id)
                logger.info(
                    f">>> start_camera: worker supervisor launched for {camera.id}"
                )
            else:
                logger.error(
                    f">>> start_camera: WorkerSupervisor instance is None for {camera.id}"
                )
        except Exception as e:
            logger.error(
                f">>> start_camera: failed to start camera worker for {camera.id}: {e}",
                exc_info=True,
            )

        # Start MediaMTX stream publisher so WHEP/HLS endpoints resolve.
        # Skip if burnin_enabled — the CameraWorker's StreamBroadcaster handles it.
        if not camera.burnin_enabled:
            logger.info(
                f">>> start_camera: starting stream publisher (non-burnin) for {camera.id}..."
            )
            try:
                manager = StreamManager.get_instance()
                await anyio.to_thread.run_sync(
                    manager.add_viewer, camera.id, camera.rtsp_url
                )
                logger.info(
                    f">>> start_camera: stream publisher started for {camera.id}"
                )
            except Exception as e:
                logger.error(
                    f">>> start_camera: stream publisher failed for {camera.id}: {e}",
                    exc_info=True,
                )
        else:
            # Wait for the CameraWorker's StreamBroadcaster to push the first
            # annotated frame to MediaMTX before returning stream URLs — otherwise
            # the frontend gets a 404 on WHEP/HLS for the first few seconds.
            logger.info(
                f">>> start_camera: waiting for StreamBroadcaster (burn-in mode) for {camera_id}..."
            )
            try:
                from app.modules.ai_runtime.worker_supervisor import WorkerSupervisor
                supervisor = WorkerSupervisor.get_instance()
                if supervisor:
                    worker = supervisor.workers.get(str(camera_id))
                    if worker and worker.stream_broadcaster:
                        ready = await anyio.to_thread.run_sync(
                            worker.stream_broadcaster.wait_until_ready, 15.0
                        )
                        if ready:
                            logger.info(
                                f">>> start_camera: StreamBroadcaster ready for {camera_id}"
                            )
                        else:
                            logger.warning(
                                f">>> start_camera: StreamBroadcaster NOT ready after timeout for {camera_id}"
                            )
                    else:
                        logger.warning(
                            f">>> start_camera: no worker/stream_broadcaster found for {camera_id} yet"
                        )
            except Exception as e:
                logger.error(
                    f">>> start_camera: StreamBroadcaster wait failed for {camera_id}: {e}",
                    exc_info=True,
                )

        logger.info(
            f">>> start_camera: COMPLETE"
            f" | camera_id={camera.id}"
            f" | name='{camera.name}'"
            f" | status={camera.status}"
        )
        return camera

    @staticmethod
    async def stop_camera(db: AsyncSession, camera_id: UUID) -> Camera:
        """Mark camera as inactive and stop worker."""
        camera = await CameraService.get_camera(db, camera_id)
        camera.status = CameraStatus.INACTIVE
        await db.flush()
        await db.refresh(camera)

        await CameraService._teardown_resources(camera_id)

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

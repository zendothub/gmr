"""Application lifecycle management - startup and shutdown events."""

import asyncio
import time
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI

from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    settings = get_settings()

    # --- STARTUP ---
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Detect compute device once (CUDA → MPS → CPU).
    # Also probes the FFmpeg build for h264_nvenc / h264_videotoolbox support.
    # Result is cached in memory AND written to runtime/device_info.json.
    # All AI components (YOLO, OSNet, InsightFace, FFmpeg) read from this single source.
    try:
        from app.utils.device import detect_and_save_device, get_ffmpeg_video_codec_args
        detect_and_save_device()
        # Pre-warm the FFmpeg encoder probe so the result is cached before
        # any camera worker starts.
        codec_args = get_ffmpeg_video_codec_args(settings.FFMPEG_BINARY)
        logger.info(f"FFmpeg video encoder selected: {codec_args[1]}")
    except Exception as e:
        logger.warning(f"Device detection failed: {e} — models will fall back to CPU")

    # Initialize MinIO bucket
    try:
        from app.modules.storage.minio_client import get_client
        get_client()  # _ensure_bucket() is called implicitly
        logger.info(f"MinIO bucket '{settings.MINIO_BUCKET_PREFIX}' ready")
    except Exception as e:
        logger.error(f"Failed to initialize MinIO: {e}")
        raise

    # Background job scheduler runs in a SEPARATE process (retail-ai-worker.service).
    # Do NOT start APScheduler here — it blocks the event loop during heavy
    # DB/MinIO operations (dedup, sweep) and freezes the API for 1-2 minutes.
    logger.info("Background jobs handled by retail-ai-worker.service (separate process)")

    # Wait for MediaMTX to be ready before restoring camera workers.
    # On boot, the docker-compose service may have started but MediaMTX
    # may need a few more seconds to initialize and listen on port 9997.
    # Poll the API until it responds (max 30 seconds).
    try:
        import httpx
        mediamtx_api_url = f"http://{settings.MEDIAMTX_HOST}:{settings.MEDIAMTX_API_PORT}/v1/paths"
        logger.info(f"Waiting for MediaMTX to be ready at {mediamtx_api_url}...")
        start_wait = time.monotonic()
        async with httpx.AsyncClient() as client:
            while time.monotonic() - start_wait < 30.0:
                try:
                    resp = await client.get(mediamtx_api_url, timeout=2.0)
                    if resp.status_code == 200:
                        logger.info(f"MediaMTX is ready (took {time.monotonic() - start_wait:.1f}s)")
                        break
                except Exception:
                    pass
                await asyncio.sleep(1.0)
            else:
                logger.warning("MediaMTX did not become ready within 30s — proceeding anyway")
    except ImportError:
        logger.warning("httpx not installed — skipping MediaMTX readiness check")
    except Exception as e:
        logger.warning(f"MediaMTX readiness check failed: {e} — proceeding anyway")

    # Auto-restart camera workers + stream publishers for all cameras
    # that were active before the server was restarted.  Without this step
    # every camera shows "404 path not found" after restart because:
    #   - Workers were stopped on shutdown
    #   - Stream publishers (ffmpeg → MediaMTX) were killed on shutdown
    #   - The DB still says status=active but nothing is actually running
    try:
        from app.modules.ai_runtime.worker_supervisor import WorkerSupervisor
        from app.modules.streaming.manager import StreamManager
        from app.core.db.session import AsyncSessionLocal
        from app.core.db.models.camera import Camera, CameraStatus
        from sqlalchemy import select

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Camera).where(Camera.status == CameraStatus.ACTIVE)
            )
            active_cameras = list(result.scalars().all())

        if active_cameras:
            logger.info(
                f"Restoring {len(active_cameras)} active camera(s) after server restart..."
            )
            supervisor = WorkerSupervisor.get_instance()
            stream_mgr = StreamManager.get_instance()
            import anyio

            for camera in active_cameras:
                try:
                    # Restart camera worker (YOLO + tracking + rules).
                    # The worker's StreamBroadcaster handles stream publishing
                    # when burnin_enabled=True — do NOT call add_viewer as well,
                    # or two ffmpeg processes will fight over the same RTSP URL.
                    if supervisor:
                        await supervisor.start_camera(camera.id)
                        logger.info(
                            f"Restored camera worker: {camera.name} ({camera.id})"
                        )

                    # Only start a separate stream publisher if burn-in is off.
                    # When burnin_enabled=True, the CameraWorker's StreamBroadcaster
                    # already pushes annotated frames → MediaMTX.
                    if not camera.burnin_enabled:
                        await anyio.to_thread.run_sync(
                            stream_mgr.add_viewer, camera.id, camera.rtsp_url
                        )
                        logger.info(
                            f"Restored stream publisher: {camera.name} ({camera.id})"
                        )
                    else:
                        logger.info(
                            f"Stream publishing handled by CameraWorker (burn-in mode) for {camera.name}"
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to restore camera {camera.name} ({camera.id}): {e}"
                    )
        else:
            logger.info("No active cameras to restore after restart")
    except Exception as e:
        logger.warning(f"Could not restore active cameras on startup: {e}")

    # Full-res continuous recording (filesystem only). Failures here must never
    # block AI workers / streaming.
    try:
        from app.modules.recording.service import RecordingSupervisor

        await RecordingSupervisor.get_instance().start()
    except Exception as e:
        logger.warning(f"Camera recording supervisor failed to start: {e}")

    logger.info("Application startup complete")

    yield

    # --- SHUTDOWN ---
    logger.info("Shutting down application...")

    # Background job scheduler is in a separate process (retail-ai-worker.service).
    # It is not stopped here — it manages its own lifecycle via systemd.

    try:
        from app.modules.recording.service import RecordingSupervisor

        await RecordingSupervisor.get_instance().stop()
    except Exception as e:
        logger.warning(f"Error stopping camera recording: {e}")

    # Stop AI runtime workers if running
    try:
        from app.modules.ai_runtime.worker_supervisor import WorkerSupervisor
        supervisor = WorkerSupervisor.get_instance()
        if supervisor:
            await supervisor.stop_all()
            logger.info("All camera workers stopped")
    except Exception as e:
        logger.warning(f"Error stopping workers during shutdown: {e}")

    # Shut down the shared inference thread pool
    try:
        from app.modules.ai_runtime.inference_pool import shutdown_inference_executor
        shutdown_inference_executor()
    except Exception as e:
        logger.warning(f"Error shutting down inference executor: {e}")

    # Stop all live stream publishers (ffmpeg -> MediaMTX)
    try:
        from app.modules.streaming.manager import StreamManager
        StreamManager.get_instance().shutdown()
    except Exception as e:
        logger.warning(f"Error shutting down stream publishers: {e}")

    logger.info("Application shutdown complete")


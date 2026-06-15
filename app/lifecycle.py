"""Application lifecycle management - startup and shutdown events."""

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

    # Initialize MinIO bucket
    try:
        from app.modules.storage.minio_client import get_client
        get_client()  # _ensure_bucket() is called implicitly
        logger.info(f"MinIO bucket '{settings.MINIO_BUCKET_PREFIX}' ready")
    except Exception as e:
        logger.error(f"Failed to initialize MinIO: {e}")
        raise

    # Start background job scheduler
    try:
        from app.modules.jobs.scheduler import start_scheduler
        start_scheduler()
    except Exception as e:
        logger.warning(f"Could not start background job scheduler: {e}")

    logger.info("Application startup complete")

    yield

    # --- SHUTDOWN ---
    logger.info("Shutting down application...")

    # Stop background job scheduler
    try:
        from app.modules.jobs.scheduler import stop_scheduler
        stop_scheduler()
    except Exception as e:
        logger.warning(f"Error stopping job scheduler during shutdown: {e}")

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


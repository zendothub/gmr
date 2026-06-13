"""Worker supervisor - manages lifecycle of all camera workers.

Singleton that the API layer (cameras module, runtime module) uses to
start/stop per-camera AI workers and to query their health.
"""

import asyncio
import uuid
from typing import Dict, Optional

from loguru import logger

from app.config import get_settings
from app.core.db.session import AsyncSessionLocal
from app.modules.ai_runtime.camera_worker import CameraWorker
from app.modules.rule_engine.config_loader import (
    load_camera_config,
    load_runtime_config,
    load_active_cameras,
)


class WorkerSupervisor:
    """Singleton supervisor for camera workers."""

    _instance: Optional["WorkerSupervisor"] = None

    def __init__(self):
        self.settings = get_settings()
        self.workers: Dict[str, CameraWorker] = {}  # camera_id (str) -> worker
        self._lock = asyncio.Lock()
        self._monitor_task: Optional[asyncio.Task] = None
        self._crash_retries: Dict[str, int] = {}  # camera_id (str) -> retry_count

    @classmethod
    def get_instance(cls) -> "WorkerSupervisor":
        """Get (or lazily create) the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    async def start_camera(self, camera_id: uuid.UUID) -> dict:
        """Start a camera worker (loads camera + runtime config from DB)."""
        async with self._lock:
            key = str(camera_id)

            if key in self.workers and self.workers[key].is_running:
                logger.info(f"Camera worker already running: {camera_id}")
                return self.workers[key].get_status()

            if len(self.workers) >= self.settings.MAX_WORKERS:
                raise RuntimeError(
                    f"Maximum number of workers ({self.settings.MAX_WORKERS}) reached"
                )

            async with AsyncSessionLocal() as db:
                camera_config = await load_camera_config(db, camera_id)
                if not camera_config:
                    raise ValueError(f"Camera not found: {camera_id}")
                runtime_config = await load_runtime_config(db, camera_id)

            worker = CameraWorker(camera_config, runtime_config)
            await worker.start()
            self.workers[key] = worker
            
            # Reset crash retry count when started explicitly via API
            self._crash_retries[key] = 0
            
            # Start background monitoring task if not already running
            if self._monitor_task is None or self._monitor_task.done():
                self._monitor_task = asyncio.create_task(self._monitor_workers_loop())
                
            logger.info(f"Worker supervisor: camera {camera_id} started")
            return worker.get_status()

    async def stop_camera(self, camera_id: uuid.UUID) -> bool:
        """Stop a camera worker."""
        async with self._lock:
            key = str(camera_id)
            worker = self.workers.pop(key, None)
            self._crash_retries.pop(key, None)

        if not worker:
            logger.info(f"No running worker for camera {camera_id}")
            return False

        await worker.stop()
        logger.info(f"Worker supervisor: camera {camera_id} stopped")
        return True

    async def start_all_active(self) -> dict:
        """Start workers for all cameras with status=active in the database."""
        started, failed = [], []
        async with AsyncSessionLocal() as db:
            cameras = await load_active_cameras(db)

        for cam in cameras:
            if cam.get("status") != "active":
                continue
            try:
                await self.start_camera(cam["id"])
                started.append(str(cam["id"]))
            except Exception as e:
                logger.error(f"Failed to start camera {cam['id']}: {e}")
                failed.append({"camera_id": str(cam["id"]), "error": str(e)})

        return {"started": started, "failed": failed}

    async def stop_all(self) -> int:
        """Stop all running workers. Returns number of stopped workers."""
        async with self._lock:
            workers = list(self.workers.values())
            self.workers.clear()
            self._crash_retries.clear()
            if self._monitor_task:
                self._monitor_task.cancel()
                self._monitor_task = None

        for worker in workers:
            try:
                await worker.stop()
            except Exception as e:
                logger.error(f"Error stopping worker {worker.camera_id}: {e}")

        logger.info(f"Worker supervisor: stopped {len(workers)} workers")
        return len(workers)

    # ------------------------------------------------------------------
    # Config reload (no per-frame DB queries: reload on demand only)
    # ------------------------------------------------------------------

    async def reload_config(self) -> dict:
        """Reload rules/zones/views from PostgreSQL into all running workers."""
        reloaded, failed = [], []
        for key, worker in list(self.workers.items()):
            try:
                async with AsyncSessionLocal() as db:
                    runtime_config = await load_runtime_config(db, worker.camera_id)
                worker.apply_runtime_config(runtime_config)
                reloaded.append(key)
            except Exception as e:
                logger.error(f"Config reload failed for camera {key}: {e}")
                failed.append({"camera_id": key, "error": str(e)})

        logger.info(f"Runtime config reloaded for {len(reloaded)} workers")
        return {"reloaded": reloaded, "failed": failed}

    # ------------------------------------------------------------------
    # Monitoring & Crash Recovery
    # ------------------------------------------------------------------

    async def _monitor_workers_loop(self):
        """Periodically check if workers crashed and restart them if needed."""
        while True:
            try:
                await asyncio.sleep(5.0)  # check every 5 seconds
                async with self._lock:
                    for key, worker in list(self.workers.items()):
                        crashed = False
                        if worker.is_running:
                            if worker._task is None or worker._task.done():
                                crashed = True

                        if crashed:
                            logger.error(f"Camera worker for camera {key} has crashed!")
                            retries = self._crash_retries.get(key, 0)
                            max_retries = self.settings.WORKER_MAX_CRASH_RETRIES

                            if retries < max_retries:
                                self._crash_retries[key] = retries + 1
                                logger.info(f"Attempting restart {retries + 1}/{max_retries} for camera {key} in 10s...")
                                asyncio.create_task(self._restart_worker_after_delay(key))
                            else:
                                logger.critical(f"Camera worker for camera {key} exceeded max crash retries ({max_retries}). Marking as error.")
                                worker.is_running = False
                                worker.error_message = "Crashed and exceeded maximum auto-restart attempts."
                                asyncio.create_task(self._mark_camera_error(key))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker monitor loop: {e}")

    async def _restart_worker_after_delay(self, key: str):
        """Wait 10 seconds and restart the camera worker."""
        await asyncio.sleep(10.0)
        async with self._lock:
            # Verify if worker is still managed and requires a restart
            worker = self.workers.get(key)
            if not worker or not worker.is_running or (worker._task and not worker._task.done()):
                return

            logger.info(f"Restarting camera worker {key} now...")
            try:
                # Stop cleanly first
                await worker.stop()

                # Reload configuration parameters
                async with AsyncSessionLocal() as db:
                    camera_config = await load_camera_config(db, uuid.UUID(key))
                    runtime_config = await load_runtime_config(db, uuid.UUID(key))

                # Replace with new worker instance
                new_worker = CameraWorker(camera_config, runtime_config)
                await new_worker.start()
                self.workers[key] = new_worker
                logger.info(f"Camera worker {key} restarted successfully.")
            except Exception as e:
                logger.error(f"Failed to restart camera worker {key}: {e}")

    async def _mark_camera_error(self, key: str):
        """Update camera status to ERROR in PostgreSQL."""
        try:
            from app.core.db.models.camera import Camera, CameraStatus
            from sqlalchemy import update
            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(Camera)
                    .where(Camera.id == uuid.UUID(key))
                    .values(status=CameraStatus.ERROR)
                )
                await db.commit()
            logger.info(f"Camera {key} status set to ERROR in database.")
        except Exception as e:
            logger.error(f"Failed to set camera {key} status to ERROR: {e}")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_worker_status(self, camera_id: uuid.UUID) -> Optional[dict]:
        """Get status of a single camera worker."""
        worker = self.workers.get(str(camera_id))
        return worker.get_status() if worker else None

    def get_all_status(self) -> dict:
        """Get status of all workers."""
        return {
            "total_workers": len(self.workers),
            "max_workers": self.settings.MAX_WORKERS,
            "workers": [w.get_status() for w in self.workers.values()],
        }
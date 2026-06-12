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
            logger.info(f"Worker supervisor: camera {camera_id} started")
            return worker.get_status()

    async def stop_camera(self, camera_id: uuid.UUID) -> bool:
        """Stop a camera worker."""
        async with self._lock:
            key = str(camera_id)
            worker = self.workers.pop(key, None)

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
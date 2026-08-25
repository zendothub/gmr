"""Recording supervisor — starts/stops per-camera recorders (no DB writes)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, Optional

from loguru import logger
from sqlalchemy import select

from app.config import Settings, get_settings
from app.core.db.models.camera import Camera, CameraStatus
from app.core.db.session import AsyncSessionLocal
from app.modules.recording.paths import resolve_recording_root
from app.modules.recording.recorder import CameraRecorder


class RecordingSupervisor:
    """Singleton: keep one full-res recorder per ACTIVE camera."""

    _instance: Optional["RecordingSupervisor"] = None

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.root: Optional[Path] = None
        self.recorders: Dict[str, CameraRecorder] = {}
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "RecordingSupervisor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    async def start(self) -> None:
        if not self.settings.ENABLE_CAMERA_RECORDING:
            logger.info("Camera recording disabled (ENABLE_CAMERA_RECORDING=false)")
            return
        if self._task and not self._task.done():
            return

        try:
            self.root = resolve_recording_root(
                explicit_root=self.settings.RECORDING_ROOT,
                hdd_mount=self.settings.RECORDING_HDD_MOUNT,
                fallback_root=self.settings.RECORDING_FALLBACK_ROOT,
                root_folder_name=self.settings.RECORDING_ROOT_FOLDER_NAME,
            )
        except Exception as e:
            logger.error(f"Recording root resolve failed — recording off: {e}")
            return

        self._stop.clear()
        self._task = asyncio.create_task(self._supervise_loop(), name="recording-supervisor")
        logger.info(
            f"RecordingSupervisor started root={self.root} "
            f"chunk_hours={self.settings.RECORDING_CHUNK_HOURS}"
        )

    async def stop(self) -> None:
        self._stop.set()
        async with self._lock:
            cams = list(self.recorders.keys())
            for key in cams:
                rec = self.recorders.pop(key, None)
                if rec:
                    await rec.stop()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=20.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        logger.info("RecordingSupervisor stopped")

    def status(self) -> dict:
        return {
            "enabled": bool(self.settings.ENABLE_CAMERA_RECORDING),
            "root": str(self.root) if self.root else None,
            "chunk_hours": self.settings.RECORDING_CHUNK_HOURS,
            "recorders": [r.status() for r in self.recorders.values()],
        }

    async def _supervise_loop(self) -> None:
        refresh = max(10, int(self.settings.RECORDING_CAMERA_REFRESH_SECONDS))
        while not self._stop.is_set():
            try:
                await self._sync_cameras()
            except Exception as e:
                logger.error(f"Recording camera sync failed: {e}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=refresh)
            except asyncio.TimeoutError:
                pass

    async def _sync_cameras(self) -> None:
        if self.root is None:
            return
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Camera).where(Camera.status == CameraStatus.ACTIVE)
            )
            cameras = list(result.scalars().all())

        wanted: Dict[str, Camera] = {str(c.id): c for c in cameras if c.rtsp_url}

        async with self._lock:
            # Stop removed / inactive
            for key in list(self.recorders.keys()):
                if key not in wanted:
                    rec = self.recorders.pop(key)
                    await rec.stop()
                    logger.info(f"Recording stopped (camera inactive): {key}")

            # Start new
            for key, cam in wanted.items():
                existing = self.recorders.get(key)
                if existing and existing.is_running:
                    # RTSP / name change → restart
                    if existing.rtsp_url != cam.rtsp_url or existing.camera_name != cam.name:
                        await existing.stop()
                        self.recorders.pop(key, None)
                    else:
                        continue
                rec = CameraRecorder(
                    camera_id=key,
                    camera_name=cam.name or key,
                    rtsp_url=cam.rtsp_url,
                    settings=self.settings,
                    root=self.root,
                )
                rec.start()
                self.recorders[key] = rec

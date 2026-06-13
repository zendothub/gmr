"""StreamManager - process-wide registry of active stream publishers.

Responsibilities:
- Start/stop one publisher per camera (idempotent, thread-safe).
- Ref-count viewers so a stream is only republished while someone is watching.
- Auto-stop streams that have been idle (no viewers) past STREAM_IDLE_TIMEOUT.

Optimization: a single publisher (ffmpeg copy/remux) serves all viewers of a
camera via MediaMTX fan-out, so adding viewers costs ~nothing on the backend.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from loguru import logger

from app.config import get_settings
from app.modules.streaming.base import StreamPublisher
from app.modules.streaming.ffmpeg_publisher import FFmpegPublisher
from app.modules.streaming.mediamtx import MediaMTXManager, StreamEndpoints


@dataclass
class _StreamHandle:
    publisher: StreamPublisher
    endpoints: StreamEndpoints
    viewers: int = 0
    started_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)


class StreamManager:
    """Singleton managing all live republished streams."""

    _instance: Optional["StreamManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self.settings = get_settings()
        self.mtx = MediaMTXManager()
        self._streams: Dict[str, _StreamHandle] = {}
        self._lock = threading.Lock()
        self._reaper: Optional[threading.Thread] = None
        self._stop_reaper = threading.Event()

    # -- singleton ---------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "StreamManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._start_reaper()
        return cls._instance

    # -- publisher factory (swap transport here if needed) ----------------

    def _make_publisher(self, camera_id: uuid.UUID, rtsp_url: str) -> StreamPublisher:
        target = self.mtx.ingest_target(camera_id)
        return FFmpegPublisher(source_url=rtsp_url, target=target)

    # -- public API --------------------------------------------------------

    def start_stream(self, camera_id: uuid.UUID, rtsp_url: str) -> StreamEndpoints:
        """Ensure a publisher is running for this camera; returns playback URLs."""
        key = str(camera_id)
        with self._lock:
            handle = self._streams.get(key)
            if handle is None:
                publisher = self._make_publisher(camera_id, rtsp_url)
                publisher.start()
                handle = _StreamHandle(
                    publisher=publisher,
                    endpoints=self.mtx.endpoints(camera_id),
                )
                self._streams[key] = handle
                logger.info(f"Stream started for camera {key}")
            elif not handle.publisher.is_alive():
                handle.publisher.start()
            handle.last_active_at = time.time()
            return handle.endpoints

    def add_viewer(self, camera_id: uuid.UUID, rtsp_url: str) -> StreamEndpoints:
        """Register a viewer (starts the stream if needed)."""
        endpoints = self.start_stream(camera_id, rtsp_url)
        with self._lock:
            handle = self._streams.get(str(camera_id))
            if handle:
                handle.viewers += 1
                handle.last_active_at = time.time()
        return endpoints

    def remove_viewer(self, camera_id: uuid.UUID) -> None:
        """De-register a viewer; stream is reaped later if it stays idle."""
        with self._lock:
            handle = self._streams.get(str(camera_id))
            if handle:
                handle.viewers = max(0, handle.viewers - 1)
                handle.last_active_at = time.time()

    def stop_stream(self, camera_id: uuid.UUID) -> bool:
        """Force-stop a camera's publisher."""
        key = str(camera_id)
        with self._lock:
            handle = self._streams.pop(key, None)
        if handle:
            handle.publisher.stop()
            logger.info(f"Stream stopped for camera {key}")
            return True
        return False

    def get_status(self, camera_id: uuid.UUID) -> Optional[dict]:
        with self._lock:
            handle = self._streams.get(str(camera_id))
            if not handle:
                return None
            return {
                "camera_id": str(camera_id),
                "is_publishing": handle.publisher.is_alive(),
                "viewers": handle.viewers,
                "uptime_seconds": round(time.time() - handle.started_at, 1),
                "last_error": handle.publisher.last_error,
                "endpoints": {
                    "webrtc_url": handle.endpoints.webrtc_url,
                    "hls_url": handle.endpoints.hls_url,
                    "rtsp_url": handle.endpoints.rtsp_url,
                },
            }

    def shutdown(self) -> None:
        """Stop all publishers (called on app shutdown)."""
        self._stop_reaper.set()
        with self._lock:
            handles = list(self._streams.values())
            self._streams.clear()
        for h in handles:
            h.publisher.stop()
        logger.info("StreamManager shut down; all publishers stopped")

    # -- idle reaper -------------------------------------------------------

    def _start_reaper(self) -> None:
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True)
        self._reaper.start()

    def _reap_loop(self) -> None:
        timeout = self.settings.STREAM_IDLE_TIMEOUT_SECONDS
        while not self._stop_reaper.wait(10.0):
            now = time.time()
            to_stop = []
            with self._lock:
                for key, handle in list(self._streams.items()):
                    if handle.viewers <= 0 and (now - handle.last_active_at) > timeout:
                        to_stop.append((key, handle))
                        del self._streams[key]
            for key, handle in to_stop:
                handle.publisher.stop()
                logger.info(f"Stream reaped (idle) for camera {key}")

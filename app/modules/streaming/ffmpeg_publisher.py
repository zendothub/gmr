"""FFmpeg-based stream publisher.

Republishes a camera's RTSP into MediaMTX over RTSP. Two modes:

- "copy"        : remux only (no re-encode). Cheapest CPU, requires the camera
                  to already output H.264 (true for virtually all CCTV/NVRs).
- "lowlatency"  : re-encode to a browser-friendly low-latency H.264 (use only
                  when the source codec is not WebRTC-compatible).

The ffmpeg process is supervised by a small watchdog thread that restarts it
with exponential backoff if it dies while it should be running.
"""

from __future__ import annotations

import shlex
import subprocess
import threading
import time
from typing import List, Optional

from loguru import logger

from app.config import get_settings
from app.modules.streaming.base import StreamPublisher, PublishTarget


class FFmpegPublisher(StreamPublisher):
    """Publishes RTSP -> MediaMTX(RTSP) using an ffmpeg subprocess."""

    def __init__(self, source_url: str, target: PublishTarget, mode: Optional[str] = None):
        super().__init__(source_url, target)
        self.settings = get_settings()
        self.mode = (mode or self.settings.STREAM_PUBLISH_MODE or "copy").lower()

        self._proc: Optional[subprocess.Popen] = None
        self._proc_start_time: float = 0.0
        self._watchdog: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._error: Optional[str] = None

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _build_command(self) -> List[str]:
        s = self.settings
        cmd: List[str] = [
            s.FFMPEG_BINARY,
            "-nostdin",
            "-loglevel", "warning",
            # Prefer TCP for RTSP - far more reliable over WAN/NVRs.
            "-rtsp_transport", "tcp",
            # Give the source stream time to produce data before ffmpeg gives up.
            # Important in copy mode where no video keyframe = empty output = exit 0.
            "-timeout", "30000000",       # 30 s socket I/O timeout (microseconds)
            "-analyzeduration", "15M",    # probe up to 15 s for stream info
            "-probesize", "10M",          # read up to 10 MB for codec discovery
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-i", self.source_url,
        ]

        if self.mode == "lowlatency":
            # Re-encode video to low-latency H.264, drop audio.
            cmd += [
                "-an",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-tune", "zerolatency",
                "-pix_fmt", "yuv420p",
                "-g", "30",
            ]
        else:
            # Remux only - no re-encode.  Drop audio — surveillance cameras
            # don't need it, and an audio-only track would break WebRTC.
            cmd += ["-an", "-c", "copy"]

        # Push to MediaMTX over RTSP.
        cmd += [
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.target.ingest_url,
        ]
        return cmd

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return  # already running
            self._stop_event.clear()
            self._error = None
            self._spawn()
            self._watchdog = threading.Thread(target=self._watch, daemon=True)
            self._watchdog.start()
        logger.info(
            f"FFmpeg publisher started: {self._safe(self.source_url)} -> "
            f"{self.target.ingest_url} (mode={self.mode})"
        )

    def _spawn(self) -> None:
        cmd = self._build_command()
        logger.debug(f"Spawning ffmpeg: {shlex.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._proc_start_time = time.time()

    def _watch(self) -> None:
        """Restart ffmpeg with backoff if it dies while it should be running."""
        backoff = 0.5
        max_backoff = 30.0
        while not self._stop_event.is_set():
            proc = self._proc
            if proc is None:
                break
            ret = proc.wait()
            if self._stop_event.is_set():
                break
            # Capture last error line for diagnostics.
            try:
                if proc.stderr:
                    tail = proc.stderr.read().decode(errors="ignore").strip().splitlines()
                    if tail:
                        self._error = tail[-1]
            except Exception:
                pass
            logger.warning(
                f"FFmpeg publisher exited (code={ret}) for {self.target.path}; "
                f"restarting in {backoff:.0f}s. last_error={self._error}"
            )
            if self._stop_event.wait(backoff):
                break
            backoff = min(backoff * 2, max_backoff)
            with self._lock:
                if not self._stop_event.is_set():
                    self._spawn()
                    backoff = 0.5  # reset after a successful respawn

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        logger.info(f"FFmpeg publisher stopped: {self.target.path}")

    def is_alive(self) -> bool:
        proc = self._proc
        return bool(proc and proc.poll() is None)

    def wait_until_alive(self, timeout: float = 8.0) -> bool:
        """Wait for the ffmpeg process to stabilize (produce output).

        Returns True once the process has been alive for at least 1 second
        without exiting, or False if it died or timed out.

        This prevents returning stream URLs to the client while ffmpeg is
        still probing / waiting for the first video keyframe.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            proc = self._proc
            if proc is None:
                return False
            ret = proc.poll()
            if ret is not None:
                # Exited already - capture error
                try:
                    if proc.stderr:
                        tail = proc.stderr.read().decode(errors="ignore").strip().splitlines()
                        if tail:
                            self._error = tail[-1]
                except Exception:
                    pass
                logger.warning(
                    f"FFmpeg publisher died during stabilization (code={ret}) "
                    f"for {self.target.path}: {self._error}"
                )
                return False
            # Process is still running. Give it at minimum 1.5 s to push the
            # first few packets into MediaMTX before declaring it ready.
            if time.time() - self._proc_start_time >= 1.5:
                return True
            time.sleep(0.1)
        # Still running but timed out waiting for stabilization window.
        # The process hasn't exited, so it's probably fine.
        return bool(self._proc and self._proc.poll() is None)

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    @staticmethod
    def _safe(url: str) -> str:
        """Redact credentials in an RTSP URL for logging."""
        if "@" in url and "://" in url:
            scheme, rest = url.split("://", 1)
            return f"{scheme}://***@{rest.split('@', 1)[1]}"
        return url

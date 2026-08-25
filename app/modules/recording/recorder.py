"""Per-camera full-resolution RTSP recorder (FFmpeg, no DB)."""

from __future__ import annotations

import asyncio
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import Settings
from app.modules.recording.paths import camera_output_path, chunk_window, ensure_dir


class CameraRecorder:
    """Records one camera RTSP stream into wall-clock chunks on disk."""

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        rtsp_url: str,
        settings: Settings,
        root: Path,
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.settings = settings
        self.root = root
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._current_file: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(), name=f"rec-{self.camera_id[:8]}")
        logger.info(
            f"CameraRecorder started: {self.camera_name} ({self.camera_id}) → {self.root}"
        )

    async def stop(self) -> None:
        self._stop.set()
        self._kill_proc()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=15.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        logger.info(f"CameraRecorder stopped: {self.camera_name}")

    def status(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "running": self.is_running,
            "current_file": self._current_file,
            "root": str(self.root),
        }

    def _kill_proc(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
        except Exception as e:
            logger.warning(f"Recorder kill failed camera={self.camera_name}: {e}")
        finally:
            self._proc = None

    async def _run_loop(self) -> None:
        delay = float(self.settings.RECORDING_RESTART_DELAY_SECONDS)
        chunk_h = float(self.settings.RECORDING_CHUNK_HOURS)
        while not self._stop.is_set():
            try:
                now = datetime.now().astimezone()
                start, end = chunk_window(now, chunk_h)
                remaining = (end - now).total_seconds()
                if remaining < 2.0:
                    # Wait for next slot boundary
                    await asyncio.sleep(min(2.0, delay))
                    continue

                out_path = camera_output_path(self.root, self.camera_name, start, end)
                ensure_dir(out_path.parent)

                # Skip finished complete files for this slot (restart mid-chunk → append new part)
                if out_path.exists() and out_path.stat().st_size > 0:
                    # If file already covers this slot and we restarted late, write a
                    # unique suffix so we never overwrite a good take.
                    stem = out_path.stem
                    suffix = now.strftime("%H%M%S")
                    out_path = out_path.with_name(f"{stem}_cont_{suffix}.mp4")

                await self._record_chunk(out_path, remaining, start, end)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Recorder loop error camera={self.camera_name}: {e}")
                await asyncio.sleep(delay)

    async def _record_chunk(
        self,
        out_path: Path,
        duration_sec: float,
        start: datetime,
        end: datetime,
    ) -> None:
        # Must end with .mp4 so FFmpeg can pick the muxer (.mp4.partial fails).
        partial = out_path.with_name(out_path.stem + ".partial.mp4")
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass

        cmd = self._build_ffmpeg_cmd(partial, duration_sec)
        self._current_file = str(out_path)
        logger.info(
            f"Recording chunk camera={self.camera_name} "
            f"slot={start.isoformat()}→{end.isoformat()} "
            f"dur={duration_sec:.0f}s file={out_path.name}"
        )

        loop = asyncio.get_running_loop()
        log_path = Path("logs") / "camera_recording.log"
        ensure_dir(log_path.parent)

        def _run() -> int:
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write(
                    f"\n--- {datetime.now().isoformat()} {self.camera_name} "
                    f"{' '.join(cmd)}\n"
                )
                logf.flush()
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=logf,
                    start_new_session=True,
                )
                return self._proc.wait()

        try:
            rc = await loop.run_in_executor(None, _run)
        finally:
            self._proc = None

        if self._stop.is_set():
            # Interrupted on shutdown — keep partial if any bytes written
            if partial.exists() and partial.stat().st_size > 0:
                try:
                    partial.replace(out_path.with_name(out_path.stem + "_interrupted.mp4"))
                except OSError:
                    pass
            elif partial.exists():
                partial.unlink(missing_ok=True)
            return

        if rc == 0 and partial.exists() and partial.stat().st_size > 0:
            partial.replace(out_path)
            logger.info(
                f"Recording saved camera={self.camera_name} file={out_path} "
                f"size_mb={out_path.stat().st_size / 1e6:.1f}"
            )
        else:
            logger.error(
                f"Recording chunk failed camera={self.camera_name} rc={rc} "
                f"partial={partial.exists()}"
            )
            if partial.exists() and partial.stat().st_size == 0:
                partial.unlink(missing_ok=True)
            await asyncio.sleep(float(self.settings.RECORDING_RESTART_DELAY_SECONDS))

    def _build_ffmpeg_cmd(self, out_path: Path, duration_sec: float) -> list:
        """Full-resolution stream copy (native camera res, no re-encode)."""
        ff = self.settings.FFMPEG_BINARY or "ffmpeg"
        # Cap duration slightly under slot end so next chunk can start cleanly
        dur = max(1.0, float(duration_sec) - 0.5)
        # Note: do not pass -rw_timeout as a global flag — this host's FFmpeg
        # rejects it before -i ("Option rw_timeout not found"). RTSP TCP is enough.
        return [
            ff,
            "-nostdin",
            "-hide_banner",
            "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-fflags", "+genpts",
            "-i", self.rtsp_url,
            "-t", f"{dur:.1f}",
            "-map", "0:v:0",
            "-c:v", "copy",
            "-an",
            "-f", "mp4",
            "-movflags", "+faststart",
            "-y",
            str(out_path),
        ]

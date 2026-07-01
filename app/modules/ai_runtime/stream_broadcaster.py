"""Stream broadcaster with YOLO bounding box burn-in.

Takes frames from the shared LatestFrameBuffer (same single RTSP capture thread
used by the AI loop), draws cached bounding boxes + zone overlays + person count,
and pipes the annotated frames to an FFmpeg subprocess that encodes and pushes
into MediaMTX.

This means the browser sees the annotated stream natively via WebRTC/HLS, with
no frontend overlay required.

Architecture:
  LatestFrameBuffer (background capture thread, full camera FPS)
       │
       ├── _run_loop (AI, 5-10fps) → updates latest_tracks[] in-place
       │
       └── StreamBroadcaster (daemon thread, STREAM_BURNIN_FPS)
              ├─ get_latest() raw frame
              ├─ resize to FFmpeg dimensions if needed
              ├─ draw zone overlays (semi-transparent blue fill + green border)
              ├─ draw cached bboxes + "Persons: N" counter
              └─ pipe to ffmpeg stdin → libx264 → rtsp://mediamtx:8554/cam_<uuid>
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from app.config import get_settings
from app.modules.ai_runtime.frame_buffer import LatestFrameBuffer
from app.modules.streaming.mediamtx import camera_path
from app.utils.geometry import polygon_from_json


class StreamBroadcaster:
    """Pipes annotated frames (zones + boxes + person count) into MediaMTX via FFmpeg."""

    def __init__(
        self,
        frame_buffer: LatestFrameBuffer,
        camera_id: uuid.UUID,
        width: int,
        height: int,
        fps: int = 15,
        max_restart_attempts: int = 10,
    ):
        self.settings = get_settings()
        self.frame_buffer = frame_buffer
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.max_restart_attempts = max_restart_attempts

        # Shared mutable state — populated by CameraWorker._process_frame()
        # each time YOLO tracking runs (mutated IN-PLACE via .clear() + .append()).
        # Format: [{"x1": int, "y1": int, "x2": int, "y2": int}, ...]
        self.latest_tracks: List[dict] = []

        # Shared zone definitions — set by CameraWorker._start_broadcaster().
        # Format: [{"name": str, "polygon": <json>}, ...]
        self.latest_zones: List[dict] = []

        self._proc: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self._first_frame_sent = threading.Event()
        self._consecutive_failures = 0
        self._fatal_error = False
        self._successful_writes = 0  # Track consecutive successful writes

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._error = None
        self._spawn_ffmpeg()
        self._thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"StreamBroadcaster started: camera={self.camera_id} "
            f"{self.width}x{self.height} @ {self.fps}fps"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._kill_ffmpeg()
        logger.info(f"StreamBroadcaster stopped: camera={self.camera_id}")

    def is_alive(self) -> bool:
        proc = self._proc
        return bool(proc and proc.poll() is None)

    def has_fatal_error(self) -> bool:
        """Check if the broadcaster has encountered a fatal error and stopped trying."""
        return self._fatal_error

    def get_error(self) -> Optional[str]:
        """Get the last error message, if any."""
        return self._error

    def wait_until_ready(self, timeout: float = 15.0) -> bool:
        """Wait for at least one annotated frame to be pushed to FFmpeg.

        Returns True once a frame has been written to ffmpeg stdin, or False
        if ffmpeg died or the timeout was reached with no stream data.
        """
        if not self._first_frame_sent.wait(timeout):
            logger.warning(
                f"StreamBroadcaster for camera {self.camera_id} "
                f"did not send first frame within {timeout}s"
            )
            return False
        return self.is_alive()

    # ------------------------------------------------------------------
    # FFmpeg subprocess
    # ------------------------------------------------------------------

    def _ingest_url(self) -> str:
        """MediaMTX RTSP ingest URL for this camera's path."""
        path = camera_path(self.camera_id)
        return (
            f"rtsp://{self.settings.MEDIAMTX_HOST}:"
            f"{self.settings.MEDIAMTX_RTSP_PORT}/{path}"
        )

    def _build_command(self) -> List[str]:
        s = self.settings
        return [
            s.FFMPEG_BINARY,
            "-nostdin",
            "-loglevel", "warning",
            # Raw BGR frames piped in via stdin
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            # Encode to low-latency H.264 (no audio)
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-g", str(self.fps * 2),  # keyframe interval ~2 seconds
            # Push to MediaMTX
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self._ingest_url(),
        ]

    def _spawn_ffmpeg(self) -> None:
        cmd = self._build_command()
        logger.debug(f"StreamBroadcaster spawning ffmpeg: {' '.join(cmd)}")
        if self.settings.STREAM_PIPELINE_LOG:
            import os
            os.makedirs("logs", exist_ok=True)
            self._log_file = open("logs/stream_pipeline.log", "a", encoding="utf-8")
            stderr_dest = self._log_file
        else:
            self._log_file = None
            stderr_dest = subprocess.DEVNULL
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_dest,
        )

    def _kill_ffmpeg(self) -> None:
        proc = self._proc
        self._proc = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ------------------------------------------------------------------
    # Broadcast loop
    # ------------------------------------------------------------------

    def _broadcast_loop(self) -> None:
        """Continuously read raw frames, draw overlays, pipe to FFmpeg."""
        interval = 1.0 / self.fps
        last_frame_ts = 0.0
        backoff = 0.5

        while not self._stop_event.is_set():
            loop_start = time.time()

            # Get latest frame from shared buffer (same capture thread as AI)
            frame, frame_ts = self.frame_buffer.get_latest()

            if frame is None or frame_ts <= last_frame_ts:
                # No new frame yet — wait a bit
                self._stop_event.wait(interval / 4)
                continue

            # ----------------------------------------------------------------
            # Detect resolution mismatch on the very first real frame.
            # If the RTSP stream resolution differs from what FFmpeg was told,
            # restart FFmpeg with the correct dimensions to avoid the tiling
            # mosaic effect.
            # ----------------------------------------------------------------
            actual_h, actual_w = frame.shape[:2]
            if actual_w != self.width or actual_h != self.height:
                logger.info(
                    f"StreamBroadcaster resolution mismatch for camera {self.camera_id}: "
                    f"expected {self.width}x{self.height}, got {actual_w}x{actual_h}. "
                    f"Restarting FFmpeg with correct dimensions."
                )
                self._kill_ffmpeg()
                self.width = actual_w
                self.height = actual_h
                self._spawn_ffmpeg()
                last_frame_ts = 0.0
                continue

            # Check if FFmpeg died; respawn with backoff and retry limit
            if self._proc is None or self._proc.poll() is not None:
                self._consecutive_failures += 1
                self._successful_writes = 0  # Reset on failure
                
                if self._proc is not None:
                    exit_code = self._proc.returncode
                    logger.warning(
                        f"StreamBroadcaster ffmpeg died (code={exit_code}) "
                        f"for camera {self.camera_id} (attempt {self._consecutive_failures}/{self.max_restart_attempts})"
                    )
                    
                    # Exit code 224 typically means connection refused to MediaMTX
                    if exit_code == 224:
                        logger.error(
                            f"StreamBroadcaster cannot connect to MediaMTX at {self._ingest_url()} "
                            f"for camera {self.camera_id}. Check MediaMTX service and network connectivity."
                        )
                
                # Check if we've exceeded max restart attempts
                if self._consecutive_failures >= self.max_restart_attempts:
                    self._fatal_error = True
                    self._error = f"FFmpeg failed {self._consecutive_failures} times. Giving up."
                    logger.error(
                        f"StreamBroadcaster for camera {self.camera_id} exceeded max restart attempts "
                        f"({self.max_restart_attempts}). Stopping broadcast. Error: {self._error}"
                    )
                    break
                
                # Wait with exponential backoff before retrying
                wait_time = min(backoff * (2 ** (self._consecutive_failures - 1)), 30.0)
                logger.info(f"StreamBroadcaster waiting {wait_time:.1f}s before retry...")
                self._stop_event.wait(wait_time)
                
                if self._stop_event.is_set():
                    break
                
                self._spawn_ffmpeg()
                # Reset frame ts to force a fresh frame after respawn
                last_frame_ts = 0.0
                continue

            # FFmpeg is alive - try to write frame
            last_frame_ts = frame_ts

            try:
                annotated = self._draw_overlays(frame)
                self._proc.stdin.write(annotated.tobytes())
                self._proc.stdin.flush()  # Ensure data is sent
                
                # Increment successful writes counter
                self._successful_writes += 1
                
                # Only reset failure counter after 3+ consecutive successful writes
                # This prevents premature "recovered" messages when FFmpeg dies immediately
                if self._consecutive_failures > 0 and self._successful_writes >= 3:
                    logger.info(
                        f"StreamBroadcaster for camera {self.camera_id} recovered after "
                        f"{self._consecutive_failures} failures ({self._successful_writes} successful writes)"
                    )
                    self._consecutive_failures = 0
                    backoff = 0.5
                
                if not self._first_frame_sent.is_set():
                    self._first_frame_sent.set()
                    logger.info(
                        f"StreamBroadcaster first frame sent for camera {self.camera_id} "
                        f"({actual_w}x{actual_h})"
                    )
            except (BrokenPipeError, OSError) as e:
                # Reset successful writes counter on pipe error
                self._successful_writes = 0
                # Don't log as warning - this is expected when FFmpeg dies
                # The next loop iteration will detect the dead process and handle it
                pass
            except Exception as e:
                self._successful_writes = 0
                logger.error(f"StreamBroadcaster error for camera {self.camera_id}: {e}")

            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_sec = max(0.001, interval - elapsed)
            self._stop_event.wait(sleep_sec)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Draw zone fills, bounding boxes, and person count on the frame.

        Returns a copy with overlays drawn.  Does NOT mutate the original frame
        (which may be concurrently read by the AI processing loop).

        Overlay layers (bottom → top):
          1. Semi-transparent blue zone fill + green border + name label
          2. Green bounding boxes per tracked person
          3. Semi-transparent "Persons: N" counter in the top-left corner
        """
        display = frame.copy()
        height, width = display.shape[:2]

        # ----------------------------------------------------------------
        # 1. Draw zones — semi-transparent blue fill + solid green outline
        # ----------------------------------------------------------------
        zones = self.latest_zones  # snapshot reference (list is mutated in-place by AI loop)
        if zones:
            # We need one addWeighted call per zone to get per-zone transparency.
            # Reuse a single overlay buffer to avoid repeated allocations.
            overlay = display.copy()
            for zone in zones:
                try:
                    zone_name = zone.get("name", "Zone")
                    poly = zone.get("polygon")
                    poly_points = polygon_from_json(poly)
                    if not poly_points:
                        continue

                    pts = np.array(
                        [(int(x * width), int(y * height)) for x, y in poly_points],
                        dtype=np.int32,
                    )

                    # Semi-transparent blue fill (BGR: 255, 100, 0 ≈ vivid blue)
                    cv2.fillPoly(overlay, [pts], (255, 100, 0))
                    cv2.addWeighted(overlay, 0.20, display, 0.80, 0, display)
                    # Reset overlay to the updated display for the next zone
                    overlay = display.copy()

                    # Solid green outline, 2 px
                    cv2.polylines(display, [pts], True, (0, 255, 0), 2)

                    # Zone name label near the first vertex
                    first_pt = pts[0]
                    label_x = max(4, first_pt[0])
                    label_y = max(16, first_pt[1] - 6)
                    cv2.putText(
                        display,
                        zone_name,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                except Exception:
                    continue

        # ----------------------------------------------------------------
        # 2. Draw bounding boxes for each tracked person
        # ----------------------------------------------------------------
        tracks = self.latest_tracks  # snapshot reference
        for track in tracks:
            try:
                x1 = int(track["x1"])
                y1 = int(track["y1"])
                x2 = int(track["x2"])
                y2 = int(track["y2"])
                # Green bounding box, 2px thick
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except (KeyError, ValueError, TypeError):
                continue

        # ----------------------------------------------------------------
        # 3. Draw person count in top-left corner
        # ----------------------------------------------------------------
        count = len(tracks)
        text = f"Persons: {count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Semi-transparent black background for readability
        bg_overlay = display.copy()
        cv2.rectangle(
            bg_overlay,
            (8, 8),
            (8 + text_w + 12, 8 + text_h + baseline + 8),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(bg_overlay, 0.5, display, 0.5, 0, display)

        # White text
        cv2.putText(
            display,
            text,
            (14, 8 + text_h + baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        return display

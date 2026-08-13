"""Stream broadcaster with YOLO bounding box burn-in (dual quality).

Takes frames from the shared LatestFrameBuffer (same single RTSP capture thread
used by the AI loop), draws cached bounding boxes + zone overlays + person count,
and pipes the annotated frames to FFmpeg subprocesses that encode and push into
MediaMTX.

Dual quality (bandwidth-aware):
  - LD  ``cam_<uuid>``     — 640×360 @ 15fps  (dashboard / multi-cam grid)
  - HD  ``cam_<uuid>_hd``  — 1280×720 @ 24fps (fullscreen)

Architecture:
  LatestFrameBuffer (background capture thread, full camera FPS)
       │
       ├── _run_loop (AI, 5-10fps) → updates latest_tracks[] in-place
       │
       └── StreamBroadcaster (daemon thread, max(LD_FPS, HD_FPS))
              ├─ get_latest() raw frame
              ├─ draw zone overlays + bboxes + person count (source res)
              ├─ resize → LD ffmpeg  → rtsp://mediamtx/.../cam_<uuid>
              └─ resize → HD ffmpeg  → rtsp://mediamtx/.../cam_<uuid>_hd
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from app.config import get_settings
from app.modules.ai_runtime.frame_buffer import LatestFrameBuffer
from app.modules.streaming.mediamtx import camera_path
from app.utils.geometry import polygon_from_json


@dataclass
class _QualityPipe:
    """One FFmpeg encode path (LD or HD)."""
    name: str  # "ld" | "hd"
    width: int
    height: int
    fps: int
    bitrate: str
    path: str
    proc: Optional[subprocess.Popen] = None
    log_file: object = None
    last_write_ts: float = 0.0
    consecutive_failures: int = 0
    successful_writes: int = 0
    first_frame_sent: threading.Event = field(default_factory=threading.Event)


class StreamBroadcaster:
    """Pipes annotated frames (zones + boxes + person count) into MediaMTX via FFmpeg.

    Publishes LD always; HD when ``STREAM_PUBLISH_HD`` is enabled.
    """

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
        # Source-frame dimensions (updated if RTSP resolution changes)
        self.width = width
        self.height = height
        # Legacy single-fps field kept for callers / logs; loop runs at max quality FPS
        self.fps = fps
        self.max_restart_attempts = max_restart_attempts

        # Shared mutable state — populated by CameraWorker._process_frame()
        # each time YOLO tracking runs (mutated IN-PLACE via .clear() + .append()).
        # Format: [{"x1": int, "y1": int, "x2": int, "y2": int}, ...]
        self.latest_tracks: List[dict] = []

        # Shared zone definitions — set by CameraWorker._start_broadcaster().
        # Format: [{"name": str, "polygon": <json>}, ...]
        self.latest_zones: List[dict] = []

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self._first_frame_sent = threading.Event()
        self._fatal_error = False

        self._pipes: List[_QualityPipe] = []
        self._build_pipes()

    # ------------------------------------------------------------------
    # Quality pipe setup
    # ------------------------------------------------------------------

    def _build_pipes(self) -> None:
        s = self.settings
        ld_w = max(2, int(s.STREAM_LD_WIDTH) // 2 * 2)
        ld_h = max(2, int(s.STREAM_LD_HEIGHT) // 2 * 2)
        ld_fps = max(1, int(s.STREAM_LD_FPS or s.STREAM_BURNIN_FPS or 15))
        self._pipes = [
            _QualityPipe(
                name="ld",
                width=ld_w,
                height=ld_h,
                fps=ld_fps,
                bitrate=s.STREAM_LD_BITRATE or s.STREAM_BITRATE,
                path=camera_path(self.camera_id, quality="ld"),
            )
        ]
        if s.STREAM_PUBLISH_HD:
            hd_w, hd_h = self._compute_hd_size(self.width, self.height)
            hd_fps = max(1, int(s.STREAM_HD_FPS or 24))
            self._pipes.append(
                _QualityPipe(
                    name="hd",
                    width=hd_w,
                    height=hd_h,
                    fps=hd_fps,
                    bitrate=s.STREAM_HD_BITRATE or s.STREAM_BITRATE,
                    path=camera_path(self.camera_id, quality="hd"),
                )
            )
        self.fps = max(p.fps for p in self._pipes)

    def _compute_hd_size(self, src_w: int, src_h: int) -> tuple[int, int]:
        """Scale source to STREAM_HD_HEIGHT, capped by STREAM_HD_MAX_WIDTH (even dims)."""
        s = self.settings
        target_h = max(2, int(s.STREAM_HD_HEIGHT))
        max_w = max(2, int(s.STREAM_HD_MAX_WIDTH))
        if src_h <= 0 or src_w <= 0:
            src_w, src_h = 1920, 1080
        scale = target_h / float(src_h)
        out_w = int(round(src_w * scale))
        out_h = target_h
        if out_w > max_w:
            scale = max_w / float(src_w)
            out_w = max_w
            out_h = int(round(src_h * scale))
        # H.264 needs even dimensions
        out_w = max(2, out_w // 2 * 2)
        out_h = max(2, out_h // 2 * 2)
        return out_w, out_h

    def _refresh_hd_size_from_source(self) -> None:
        for pipe in self._pipes:
            if pipe.name == "hd":
                nw, nh = self._compute_hd_size(self.width, self.height)
                if nw != pipe.width or nh != pipe.height:
                    logger.info(
                        f"StreamBroadcaster HD size update camera={self.camera_id}: "
                        f"{pipe.width}x{pipe.height} → {nw}x{nh}"
                    )
                    pipe.width, pipe.height = nw, nh
                    self._kill_pipe(pipe)
                    self._spawn_pipe(pipe)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._error = None
        self._fatal_error = False
        for pipe in self._pipes:
            self._spawn_pipe(pipe)
        self._thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self._thread.start()
        desc = ", ".join(f"{p.name}={p.width}x{p.height}@{p.fps}fps/{p.bitrate}" for p in self._pipes)
        logger.info(f"StreamBroadcaster started: camera={self.camera_id} [{desc}]")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        for pipe in self._pipes:
            self._kill_pipe(pipe)
        logger.info(f"StreamBroadcaster stopped: camera={self.camera_id}")

    def is_alive(self) -> bool:
        """True if at least the LD pipe is running."""
        for pipe in self._pipes:
            if pipe.name == "ld":
                return bool(pipe.proc and pipe.proc.poll() is None)
        return any(p.proc and p.proc.poll() is None for p in self._pipes)

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
    # FFmpeg subprocess (per quality)
    # ------------------------------------------------------------------

    def _ingest_url(self, path: str) -> str:
        return (
            f"rtsp://{self.settings.MEDIAMTX_HOST}:"
            f"{self.settings.MEDIAMTX_RTSP_PORT}/{path}"
        )

    def _build_command(self, pipe: _QualityPipe) -> List[str]:
        s = self.settings
        from app.utils.device import get_ffmpeg_video_codec_args
        video_codec_args = get_ffmpeg_video_codec_args(s.FFMPEG_BINARY, bitrate=pipe.bitrate)
        return [
            s.FFMPEG_BINARY,
            "-nostdin",
            "-loglevel", "warning",
            # Raw BGR frames piped in via stdin
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{pipe.width}x{pipe.height}",
            "-r", str(pipe.fps),
            "-i", "-",
            # Encode to H.264 using the best available hardware (no audio).
            "-an",
            *video_codec_args,
            "-g", str(pipe.fps * 2),  # keyframe interval ~2 seconds
            # Push to MediaMTX
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self._ingest_url(pipe.path),
        ]

    def _spawn_pipe(self, pipe: _QualityPipe) -> None:
        cmd = self._build_command(pipe)
        logger.debug(f"StreamBroadcaster spawning ffmpeg ({pipe.name}): {' '.join(cmd)}")
        if self.settings.STREAM_PIPELINE_LOG:
            import os
            os.makedirs("logs", exist_ok=True)
            pipe.log_file = open("logs/stream_pipeline.log", "a", encoding="utf-8")
            stderr_dest = pipe.log_file
        else:
            pipe.log_file = None
            stderr_dest = subprocess.DEVNULL
        pipe.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr_dest,
        )
        pipe.successful_writes = 0

    def _kill_pipe(self, pipe: _QualityPipe) -> None:
        proc = pipe.proc
        pipe.proc = None
        if proc and proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        if pipe.log_file:
            try:
                pipe.log_file.close()
            except Exception:
                pass
            pipe.log_file = None

    def _ensure_pipe_alive(self, pipe: _QualityPipe) -> bool:
        """Respawn pipe if dead. Returns False if fatal (gave up)."""
        if pipe.proc is not None and pipe.proc.poll() is None:
            return True

        pipe.consecutive_failures += 1
        pipe.successful_writes = 0
        if pipe.proc is not None:
            exit_code = pipe.proc.returncode
            logger.warning(
                f"StreamBroadcaster ffmpeg ({pipe.name}) died (code={exit_code}) "
                f"for camera {self.camera_id} "
                f"(attempt {pipe.consecutive_failures}/{self.max_restart_attempts})"
            )
            if exit_code == 224:
                logger.error(
                    f"StreamBroadcaster cannot connect to MediaMTX at "
                    f"{self._ingest_url(pipe.path)} for camera {self.camera_id}."
                )

        if pipe.consecutive_failures >= self.max_restart_attempts:
            # LD fatal stops the broadcaster; HD-only failure is non-fatal
            if pipe.name == "ld":
                self._fatal_error = True
                self._error = (
                    f"FFmpeg LD failed {pipe.consecutive_failures} times. Giving up."
                )
                logger.error(
                    f"StreamBroadcaster for camera {self.camera_id} exceeded max "
                    f"restart attempts on LD. Stopping. Error: {self._error}"
                )
            else:
                logger.error(
                    f"StreamBroadcaster HD pipe gave up for camera {self.camera_id}; "
                    f"LD continues."
                )
            return False

        wait_time = min(0.5 * (2 ** (pipe.consecutive_failures - 1)), 30.0)
        logger.info(
            f"StreamBroadcaster ({pipe.name}) waiting {wait_time:.1f}s before retry..."
        )
        self._stop_event.wait(wait_time)
        if self._stop_event.is_set():
            return False
        self._spawn_pipe(pipe)
        return True

    # ------------------------------------------------------------------
    # Broadcast loop
    # ------------------------------------------------------------------

    def _broadcast_loop(self) -> None:
        """Continuously read raw frames, draw overlays, pipe to LD/HD FFmpeg."""
        loop_fps = max(p.fps for p in self._pipes)
        interval = 1.0 / loop_fps
        last_frame_ts = 0.0

        while not self._stop_event.is_set():
            if self._fatal_error:
                break

            loop_start = time.time()

            frame, frame_ts = self.frame_buffer.get_latest()
            if frame is None or frame_ts <= last_frame_ts:
                self._stop_event.wait(interval / 4)
                continue

            actual_h, actual_w = frame.shape[:2]
            if actual_w != self.width or actual_h != self.height:
                logger.info(
                    f"StreamBroadcaster source resolution change camera={self.camera_id}: "
                    f"{self.width}x{self.height} → {actual_w}x{actual_h}"
                )
                self.width = actual_w
                self.height = actual_h
                self._refresh_hd_size_from_source()

            last_frame_ts = frame_ts
            now = time.time()

            try:
                annotated = self._draw_overlays(frame)
            except Exception as e:
                logger.error(f"StreamBroadcaster draw error for camera {self.camera_id}: {e}")
                self._stop_event.wait(interval)
                continue

            any_written = False
            for pipe in self._pipes:
                # Per-pipe FPS throttle
                min_interval = 1.0 / pipe.fps
                if pipe.last_write_ts and (now - pipe.last_write_ts) < (min_interval * 0.85):
                    continue

                # HD may have given up — skip without killing LD
                if pipe.consecutive_failures >= self.max_restart_attempts:
                    continue

                if not self._ensure_pipe_alive(pipe):
                    if pipe.name == "ld" and self._fatal_error:
                        break
                    continue

                try:
                    out = self._resize_for_pipe(annotated, pipe)
                    pipe.proc.stdin.write(out.tobytes())
                    pipe.proc.stdin.flush()
                    pipe.last_write_ts = now
                    pipe.successful_writes += 1
                    any_written = True

                    if pipe.consecutive_failures > 0 and pipe.successful_writes >= 3:
                        logger.info(
                            f"StreamBroadcaster ({pipe.name}) for camera {self.camera_id} "
                            f"recovered after {pipe.consecutive_failures} failures"
                        )
                        pipe.consecutive_failures = 0

                    if not pipe.first_frame_sent.is_set():
                        pipe.first_frame_sent.set()
                        logger.info(
                            f"StreamBroadcaster first {pipe.name} frame sent for "
                            f"camera {self.camera_id} ({pipe.width}x{pipe.height}@{pipe.fps}fps)"
                        )
                except (BrokenPipeError, OSError):
                    pipe.successful_writes = 0
                except Exception as e:
                    pipe.successful_writes = 0
                    logger.error(
                        f"StreamBroadcaster ({pipe.name}) write error "
                        f"for camera {self.camera_id}: {e}"
                    )

            if any_written and not self._first_frame_sent.is_set():
                self._first_frame_sent.set()

            if self._fatal_error:
                break

            elapsed = time.time() - loop_start
            sleep_sec = max(0.001, interval - elapsed)
            self._stop_event.wait(sleep_sec)

    @staticmethod
    def _resize_for_pipe(frame: np.ndarray, pipe: _QualityPipe) -> np.ndarray:
        h, w = frame.shape[:2]
        if w == pipe.width and h == pipe.height:
            return frame
        return cv2.resize(frame, (pipe.width, pipe.height), interpolation=cv2.INTER_AREA)

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
        # 2. Draw bounding boxes for each tracked person (+ Optional Face Box)
        # ----------------------------------------------------------------
        tracks = self.latest_tracks  # snapshot reference
        for track in tracks:
            try:
                x1 = int(track["x1"])
                y1 = int(track["y1"])
                x2 = int(track["x2"])
                y2 = int(track["y2"])
                confidence = float(track.get("confidence", 0.0))

                # Green bounding box, 2px thick
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw confidence and bbox size at top-left corner of box
                bw = x2 - x1
                bh = y2 - y1
                track_id = track.get("track_id")
                if track_id is not None:
                    label = f"T:{track_id} C:{confidence:.2f} {bw}x{bh}"
                else:
                    label = f"Anon C:{confidence:.2f} {bw}x{bh}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.40
                thickness = 1
                (lbl_w, lbl_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Semi-transparent black background behind label
                label_y_offset = max(1, y1 - lbl_h - 4)
                bg_overlay = display.copy()
                cv2.rectangle(
                    bg_overlay,
                    (x1, label_y_offset),
                    (x1 + lbl_w + 4, y1),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(bg_overlay, 0.5, display, 0.5, 0, display)

                # White text on the label
                cv2.putText(
                    display,
                    label,
                    (x1 + 2, y1 - 3),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

                # Draw Yellow/Orange Face bounding box if available
                face_bbox = track.get("face_bbox")
                face_score = track.get("face_score", 0.0)
                if face_bbox is not None and face_score > 0.0:
                    fx1 = int(face_bbox["x1"])
                    fy1 = int(face_bbox["y1"])
                    fx2 = int(face_bbox["x2"])
                    fy2 = int(face_bbox["y2"])

                    # Draw fine border for face
                    cv2.rectangle(display, (fx1, fy1), (fx2, fy2), (0, 165, 255), 1)

                    # Mini label under face bbox viz, e.g. "F:0.89"
                    flabel = f"F:{face_score:.2f}"
                    ffont = cv2.FONT_HERSHEY_SIMPLEX
                    ffont_scale = 0.32
                    fthickness = 1
                    (flbl_w, flbl_h), fbaseline = cv2.getTextSize(flabel, ffont, ffont_scale, fthickness)

                    # Highlight background behind face tag
                    f_overlay = display.copy()
                    cv2.rectangle(
                        f_overlay,
                        (fx1, fy2),
                        (fx1 + flbl_w + 2, fy2 + flbl_h + 4),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.addWeighted(f_overlay, 0.5, display, 0.5, 0, display)

                    # White text for face score
                    cv2.putText(
                        display,
                        flabel,
                        (fx1 + 1, fy2 + flbl_h + 2),
                        ffont,
                        ffont_scale,
                        (255, 255, 255),
                        fthickness,
                        cv2.LINE_AA,
                    )
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

"""Latest-frame buffer strategy for RTSP streams.

A dedicated capture thread continuously reads frames from the RTSP stream and
overwrites the latest frame. The processing loop always works on the most
recent frame, so processing latency never causes a growing backlog of
stale frames (which is what happens with a naive cv2 read loop).
"""

import os
# Set OpenCV FFMPEG timeout option (10,000,000 microseconds = 10 seconds)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;10000000"
# Silence FFmpeg standard error output logs in OpenCV
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class LatestFrameBuffer:
    """Thread-safe holder for the most recent frame of an RTSP stream."""

    def __init__(
        self,
        rtsp_url: str,
        reconnect_delay: float = 5.0,
        max_reconnect_delay: float = 60.0,
        frame_rotation: Optional[int] = None,
    ):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.frame_rotation = frame_rotation  # None, 90, 180, 270 (degrees)

        self._frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.is_connected: bool = False
        self.last_error: Optional[str] = None
        self.frames_captured: int = 0
        self.reconnect_count: int = 0

    def start(self):
        """Start the background capture thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Frame buffer capture thread started for {self.rtsp_url}")

    def stop(self):
        """Stop the capture thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.is_connected = False
        logger.info(f"Frame buffer capture thread stopped for {self.rtsp_url}")

    def reset(self):
        """Reset the frame buffer: stop and restart the capture thread."""
        logger.warning(f"Resetting frame buffer for stream {self.rtsp_url}")
        self.stop()
        self._frame = None
        self._frame_ts = 0.0
        self.start()

    def get_latest(self) -> Tuple[Optional[np.ndarray], float]:
        """Get the most recent frame and its capture timestamp."""
        with self._lock:
            if self._frame is None:
                return None, 0.0
            return self._frame.copy(), self._frame_ts

    def _capture_loop(self):
        """Continuously read frames; always keep only the latest one."""
        current_delay = self.reconnect_delay

        while not self._stop_event.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                # Keep the internal OpenCV buffer minimal
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not cap.isOpened():
                    raise ConnectionError(f"Cannot open RTSP stream: {self.rtsp_url}")

                self.is_connected = True
                self.last_error = None
                current_delay = self.reconnect_delay  # reset backoff on success
                logger.info(f"RTSP stream connected: {self.rtsp_url}")

                consecutive_failures = 0
                while not self._stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        consecutive_failures += 1
                        if consecutive_failures > 30:
                            raise ConnectionError("Too many consecutive read failures")
                        time.sleep(0.05)
                        continue

                    # Apply rotation if specified in camera configuration
                    if self.frame_rotation:
                        try:
                            if self.frame_rotation == 90:
                                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                            elif self.frame_rotation == 180:
                                frame = cv2.rotate(frame, cv2.ROTATE_180)
                            elif self.frame_rotation in (270, -90):
                                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        except Exception as e:
                            logger.error(f"Failed to rotate frame (rotation={self.frame_rotation}): {e}")

                    consecutive_failures = 0
                    with self._lock:
                        self._frame = frame
                        self._frame_ts = time.time()
                        self.frames_captured += 1

            except Exception as e:
                self.is_connected = False
                self.last_error = str(e)
                self.reconnect_count += 1
                logger.warning(f"RTSP capture error ({self.rtsp_url}): {e}")
            finally:
                if cap is not None:
                    cap.release()

            if not self._stop_event.is_set():
                logger.info(f"Reconnecting to RTSP stream in {current_delay:.0f}s...")
                self._stop_event.wait(current_delay)
                # Exponential backoff to be gentle on flaky NVRs/cameras
                current_delay = min(current_delay * 2, self.max_reconnect_delay)

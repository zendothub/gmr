"""Single-frame snapshot grabber for the zone-drawing canvas.

The zone-binding UI needs one representative still frame (with known pixel
dimensions) so the operator can click polygon points. We grab it directly from
the camera RTSP with OpenCV - this is independent of the live WebRTC preview.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
from loguru import logger

from app.config import get_settings


def grab_snapshot_jpeg(rtsp_url: str) -> Optional[Tuple[bytes, int, int]]:
    """Grab one frame from an RTSP stream and return (jpeg_bytes, width, height).

    Returns None if the stream cannot be opened / read.
    """
    settings = get_settings()
    cap = None
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, settings.SNAPSHOT_TIMEOUT_SECONDS * 1000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.warning(f"Snapshot: cannot open RTSP stream {rtsp_url}")
            return None

        # Skip a few frames to let auto-exposure / decoder settle.
        frame = None
        for _ in range(5):
            ret, f = cap.read()
            if ret and f is not None:
                frame = f
        if frame is None:
            logger.warning(f"Snapshot: failed to read frame from {rtsp_url}")
            return None

        h, w = frame.shape[:2]
        ok, buf = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.SNAPSHOT_JPEG_QUALITY],
        )
        if not ok:
            return None
        return buf.tobytes(), w, h
    except Exception as e:
        logger.error(f"Snapshot error for {rtsp_url}: {e}")
        return None
    finally:
        if cap is not None:
            cap.release()

"""Abstract streaming publisher contract.

A StreamPublisher takes a single camera's source (RTSP) and republishes it to a
streaming server endpoint (a MediaMTX path). Concrete implementations (ffmpeg,
GStreamer, ...) only need to implement start/stop/is_alive, so the rest of the
platform stays decoupled from the transport tool.
"""

from __future__ import annotations

import abc
import time as _time
from dataclasses import dataclass


@dataclass
class PublishTarget:
    """Where a publisher should push the republished stream."""
    # MediaMTX path name, e.g. "cam_<uuid>"
    path: str
    # Full RTSP ingest URL on the streaming server, e.g. rtsp://host:8554/cam_<uuid>
    ingest_url: str


class StreamPublisher(abc.ABC):
    """Republishes one source stream into the streaming server."""

    def __init__(self, source_url: str, target: PublishTarget):
        self.source_url = source_url
        self.target = target

    @abc.abstractmethod
    def start(self) -> None:
        """Start republishing (non-blocking)."""
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop republishing and release resources."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_alive(self) -> bool:
        """Return True while the publisher is actively running."""
        raise NotImplementedError

    def wait_until_alive(self, timeout: float = 8.0) -> bool:
        """Wait for the publisher to stabilize (produce output).

        Default implementation polls is_alive() in a sleep loop. Concrete
        publishers may override with transport-specific logic.

        Returns True if the publisher is confirmed alive, False otherwise.
        """
        deadline = _time.time() + timeout
        while _time.time() < deadline:
            if self.is_alive():
                return True
            _time.sleep(0.1)
        return self.is_alive()

    @property
    def last_error(self) -> str | None:
        return None

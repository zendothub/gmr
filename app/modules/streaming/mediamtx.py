"""MediaMTX integration helpers.

MediaMTX (https://github.com/bluenviron/mediamtx) is a zero-dependency media
server. We push each camera's RTSP into a MediaMTX "path" and let the browser
pull it back as WebRTC (WHEP) for low latency, with HLS as a fallback.

Dual quality:
  - LD path  ``cam_<uuid>``     — dashboard / multi-cam grid (default)
  - HD path  ``cam_<uuid>_hd``  — fullscreen / single-cam detail

This module only builds path names / URLs - it has no runtime side effects, so
it is trivially unit-testable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal, Optional

from app.config import get_settings
from app.modules.streaming.base import PublishTarget

StreamQuality = Literal["ld", "hd"]


@dataclass
class StreamEndpoints:
    """All the URLs the frontend may use to consume one camera's stream."""
    path: str
    webrtc_url: str   # WHEP endpoint LD (preferred default, lowest bandwidth)
    hls_url: str      # HLS fallback LD
    rtsp_url: str     # server-side RTSP LD (debugging / VLC)
    # HD variant (fullscreen). None when STREAM_PUBLISH_HD is disabled.
    path_hd: Optional[str] = None
    webrtc_url_hd: Optional[str] = None
    hls_url_hd: Optional[str] = None
    rtsp_url_hd: Optional[str] = None


def camera_path(camera_id: uuid.UUID | str, quality: StreamQuality = "ld") -> str:
    """Deterministic MediaMTX path name for a camera (stable across restarts).

    ``quality="ld"`` → ``cam_<uuid>`` (backward-compatible default)
    ``quality="hd"`` → ``cam_<uuid>_hd``
    """
    base = f"cam_{str(camera_id).replace('-', '')}"
    if quality == "hd":
        return f"{base}_hd"
    return base


class MediaMTXManager:
    """Builds MediaMTX ingest targets and public playback URLs."""

    def __init__(self):
        self.settings = get_settings()

    # -- internal host:port ------------------------------------------------

    @property
    def _host(self) -> str:
        return self.settings.MEDIAMTX_HOST

    def _public_base(self, port: int, scheme: str = "http", public_host: str | None = None) -> str:
        """Public base URL used by the browser.

        Resolution order:
        1. ``MEDIAMTX_PUBLIC_URL`` static override (full base, ignores port).
        2. ``public_host`` (e.g. derived from the incoming request) - so the feed
           is served on the same IP/host the browser used to reach the API.
        3. ``MEDIAMTX_HOST`` + the relevant port.
        """
        base = (self.settings.MEDIAMTX_PUBLIC_URL or "").rstrip("/")
        if base:
            # If the user put e.g. "feed-retaileye.bluecloudsoftech.com", add public scheme
            if "://" not in base:
                return f"{scheme}://{base}"
            return base

        if public_host:
            return f"{scheme}://{public_host}:{port}"
        return f"{scheme}://{self._host}:{port}"


    # -- ingest (backend -> MediaMTX) -------------------------------------

    def ingest_target(
        self, camera_id: uuid.UUID | str, quality: StreamQuality = "ld",
    ) -> PublishTarget:
        """RTSP ingest URL the publisher pushes into."""
        path = camera_path(camera_id, quality=quality)
        ingest_url = (
            f"rtsp://{self._host}:{self.settings.MEDIAMTX_RTSP_PORT}/{path}"
        )
        return PublishTarget(path=path, ingest_url=ingest_url)

    # -- egress (MediaMTX -> browser) -------------------------------------

    def _urls_for_path(self, path: str, public_host: str | None = None) -> tuple[str, str, str]:
        webrtc = f"{self._public_base(self.settings.MEDIAMTX_WEBRTC_PORT, public_host=public_host)}/{path}/whep"
        hls = f"{self._public_base(self.settings.MEDIAMTX_HLS_PORT, public_host=public_host)}/{path}/index.m3u8"
        rtsp = f"rtsp://{self._host}:{self.settings.MEDIAMTX_RTSP_PORT}/{path}"
        return webrtc, hls, rtsp

    def endpoints(
        self, camera_id: uuid.UUID | str, public_host: str | None = None
    ) -> StreamEndpoints:
        """Public URLs for consuming the stream from a browser.

        ``public_host`` (typically the host the API request came in on) makes the
        WebRTC/HLS feed resolve on the same IP the browser already reached, so it
        works over LAN without any hard-coded ``localhost``.

        LD URLs are always returned as ``webrtc_url`` / ``hls_url`` (dashboard default).
        HD URLs are returned when ``STREAM_PUBLISH_HD`` is enabled.
        """
        path_ld = camera_path(camera_id, quality="ld")
        webrtc, hls, rtsp = self._urls_for_path(path_ld, public_host=public_host)

        path_hd = None
        webrtc_hd = hls_hd = rtsp_hd = None
        if self.settings.STREAM_PUBLISH_HD:
            path_hd = camera_path(camera_id, quality="hd")
            webrtc_hd, hls_hd, rtsp_hd = self._urls_for_path(path_hd, public_host=public_host)

        return StreamEndpoints(
            path=path_ld,
            webrtc_url=webrtc,
            hls_url=hls,
            rtsp_url=rtsp,
            path_hd=path_hd,
            webrtc_url_hd=webrtc_hd,
            hls_url_hd=hls_hd,
            rtsp_url_hd=rtsp_hd,
        )

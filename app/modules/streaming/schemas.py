"""Streaming Pydantic schemas."""

from typing import Optional
from uuid import UUID
from pydantic import BaseModel


class StreamEndpointsResponse(BaseModel):
    """Playback URLs returned to the zone-binding UI."""
    camera_id: UUID
    path: str
    webrtc_url: str   # preferred (WHEP) - lowest latency
    hls_url: str      # fallback
    rtsp_url: str     # server-side (debug / VLC)


class StreamStatusResponse(BaseModel):
    camera_id: UUID
    is_publishing: bool
    viewers: int
    uptime_seconds: float
    last_error: Optional[str] = None
    webrtc_url: Optional[str] = None
    hls_url: Optional[str] = None
    rtsp_url: Optional[str] = None


class SnapshotInfoResponse(BaseModel):
    """Metadata for a grabbed snapshot frame (image served separately)."""
    camera_id: UUID
    width: int
    height: int
    image_url: str

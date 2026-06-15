"""Camera Pydantic schemas."""

from typing import Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class RTSPTestRequest(BaseModel):
    rtsp_url: str = Field(..., description="RTSP stream URL to test")


class RTSPTestResponse(BaseModel):
    success: bool
    message: str
    resolution: Optional[str] = None
    fps: Optional[float] = None


class CameraCreate(BaseModel):
    """Minimal camera add form for Apollo Pharmacy.

    Only name, rtsp_url, and area (dropdown) are sent by the frontend.
    AI-config fields (fps_target, resolution, detection_model, reid_enabled,
    demographic_enabled, frame_rotation, location_description) are never
    exposed — the backend applies model defaults internally.

    A camera has NO role/type — roles belong at the ZONE level because one
    camera may cover multiple zones (entry, exit, billing, pickup, ...).
    """
    name: str = Field(..., min_length=1, max_length=255)
    rtsp_url: str
    area_id: Optional[UUID] = Field(None, description="Area chosen from dropdown (Entry, Exit, Billing, Medicine Pickup, ...)")
    skip_rtsp_test: bool = Field(default=False, description="Skip the RTSP connectivity probe (use when camera is offline)")



class CameraUpdate(BaseModel):
    """Editable camera fields — AI config is internal-only."""
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    area_id: Optional[UUID] = None
    is_active: Optional[bool] = None


class CameraResponse(BaseModel):
    """Public camera response — only essential fields.

    AI-config fields (fps_target, resolution, detection_model, reid_enabled,
    demographic_enabled, frame_rotation, location_description) are internal-only
    and NOT exposed to the frontend.
    """
    id: UUID
    name: str
    rtsp_url: str
    area_id: Optional[UUID] = None
    status: str
    is_active: bool
    # MediaMTX path the backend republishes the feed into.
    stream_path: Optional[str] = None
    # Browser-playable feed URLs (served by MediaMTX, derived from stream_path).
    webrtc_url: Optional[str] = None
    hls_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True




class CameraHealthResponse(BaseModel):
    camera_id: UUID
    name: str
    status: str
    is_streaming: bool
    current_fps: Optional[float] = None
    uptime_seconds: Optional[float] = None
    last_frame_at: Optional[datetime] = None
    error_message: Optional[str] = None

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
    # Initial add form: name + rtsp_url + area (dropdown). Nothing else.
    # All AI-config fields use model defaults and run internally.
    name: str = Field(..., min_length=1, max_length=255)
    rtsp_url: str
    area_id: Optional[UUID] = Field(None, description="Area chosen from dropdown")
    # Internal flag: skip the (slow) RTSP probe when the stream is offline at add time.
    skip_rtsp_test: bool = Field(default=False, description="Skip RTSP connectivity probe")



class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    area_id: Optional[UUID] = None


class CameraResponse(BaseModel):
    id: UUID
    name: str
    rtsp_url: str
    area_id: Optional[UUID] = None
    status: str
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

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
    store_id: UUID
    role: str = "general"
    fps_target: int = Field(default=5, ge=1, le=30)
    resolution: Optional[str] = "1920x1080"
    detection_model: str = "yolov8n"
    reid_enabled: bool = True
    demographic_enabled: bool = False
    frame_rotation: Optional[int] = Field(default=None, ge=0, le=270)  # None, 90, 180, 270
    location_description: Optional[str] = None
    area_id: Optional[UUID] = Field(None, description="Area chosen from dropdown")
    # Internal flag: skip the (slow) RTSP probe when the stream is offline at add time.
    skip_rtsp_test: bool = Field(default=False, description="Skip RTSP connectivity probe")



class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    role: Optional[str] = None
    fps_target: Optional[int] = Field(default=None, ge=1, le=30)
    resolution: Optional[str] = None
    detection_model: Optional[str] = None
    reid_enabled: Optional[bool] = None
    demographic_enabled: Optional[bool] = None
    frame_rotation: Optional[int] = Field(default=None, ge=0, le=270)
    location_description: Optional[str] = None
    is_active: Optional[bool] = None
    area_id: Optional[UUID] = None


class CameraResponse(BaseModel):
    id: UUID
    name: str
    rtsp_url: str
    area_id: Optional[UUID] = None
    status: str
    fps_target: int
    resolution: Optional[str]
    detection_model: str
    reid_enabled: bool
    demographic_enabled: bool
    frame_rotation: Optional[int]
    location_description: Optional[str]
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

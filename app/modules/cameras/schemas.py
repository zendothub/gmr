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


class CameraResponse(BaseModel):
    id: UUID
    name: str
    rtsp_url: str
    store_id: UUID
    role: str
    status: str
    fps_target: int
    resolution: Optional[str]
    detection_model: str
    reid_enabled: bool
    demographic_enabled: bool
    frame_rotation: Optional[int]
    location_description: Optional[str]
    is_active: bool
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

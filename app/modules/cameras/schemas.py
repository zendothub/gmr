"""Camera Pydantic schemas."""

from typing import Optional, List, Dict, Any
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


# ---------------------------------------------------------------------------
# V1 schemas — kept for backward compatibility (area_id based)
# ---------------------------------------------------------------------------

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
    zone_id: Optional[UUID] = Field(None, description="Zone / position within the store (Entry, Checkout, Aisle 3, …)")
    skip_rtsp_test: bool = Field(default=False, description="Skip the RTSP connectivity probe (use when camera is offline)")


class CameraUpdate(BaseModel):
    """Editable camera fields — AI config is internal-only."""
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    area_id: Optional[UUID] = None
    store_id: Optional[UUID] = None
    is_active: Optional[bool] = None
    burnin_enabled: Optional[bool] = None


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
    store_id: Optional[UUID] = None
    store_name: Optional[str] = None
    store_zone_gate: Optional[str] = None
    zone_id: Optional[UUID] = None
    zone_name: Optional[str] = None
    status: str
    is_active: bool
    # MediaMTX path the backend republishes into.
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


# ---------------------------------------------------------------------------
# V2 schemas — store-linked camera creation (replaces area_id with store_id)
# ---------------------------------------------------------------------------

class CameraCreateV2(BaseModel):
    """V2 camera creation — links to a Store instead of an Area.

    When a store is selected from the dropdown:
    - store_id is saved on the camera
    - The store's zone_gate (physical location) is auto-populated in the response
    - All cameras of a store can be queried via GET /api/v2/cameras?store_id=...

    zone_id optionally links the camera to a specific store zone / position
    (e.g. "Entry", "Checkout", "Aisle 3") chosen from the Zone/Position dropdown.

    The eye icon on the camera row opens the camera's live stream where
    the operator can draw polygon detection zones
    (POST /api/v2/cameras/{camera_id}/zones).
    """
    name: str = Field(..., min_length=1, max_length=255)
    rtsp_url: str
    store_id: UUID = Field(..., description="Store chosen from dropdown — camera is linked to this store")
    zone_id: Optional[UUID] = Field(None, description="Zone / position within the store (Entry, Checkout, Aisle 3, …)")
    skip_rtsp_test: bool = Field(default=False, description="Skip the RTSP connectivity probe (use when camera is offline)")


class CameraUpdateV2(BaseModel):
    """V2 camera update — store-linked fields only."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    rtsp_url: Optional[str] = None
    store_id: Optional[UUID] = Field(None, description="Re-link camera to a different store")
    zone_id: Optional[UUID] = Field(None, description="Zone / position within the store (Entry, Checkout, Aisle 3, …)")
    is_active: Optional[bool] = None
    burnin_enabled: Optional[bool] = None
    skip_rtsp_test: bool = Field(default=True, description="Skip the RTSP connectivity probe on update (use when camera is offline)")


class CameraByStoreResponse(BaseModel):
    """Grouped cameras response: store → list of cameras."""
    store_id: UUID
    store_name: str
    store_zone_gate: Optional[str] = None
    cameras: list[CameraResponse]

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# V2 Detection-zone schemas (polygon editor — eye icon)
#
# These are the zones drawn on the camera's live feed.
# Each polygon is assigned a detection event type that drives what AI
# analytics are measured inside that polygon area.
#
# NOTE: "Store zones" (physical gate/location labels like "Gate B4") live in
# the stores module (/api/stores/zones) and are a completely different concept.
# ---------------------------------------------------------------------------

# Friendly event-type labels shown in the polygon-editor zone dropdown.
DETECTION_EVENT_TYPES = [
    "footfall",
    "dwell_time",
    "queue_length",
    "entry_exit",
    "heatmap",
    "purchase_intent",
]

DETECTION_EVENT_LABELS: Dict[str, str] = {
    "footfall":       "Footfall",
    "dwell_time":     "Dwell Time",
    "queue_length":   "Queue Length",
    "entry_exit":     "Entry/Exit",
    "heatmap":        "Heatmap",
    "purchase_intent": "Purchase Intent",
}


# ---------------------------------------------------------------------------
# V2 Live-feed card schema  (GET /api/v2/cameras/feeds)
# ---------------------------------------------------------------------------

class CameraFeedZoneSummary(BaseModel):
    """Minimal zone info shown on a Live Feeds card."""
    id: UUID
    name: str
    zone_type: str
    zone_type_label: str
    is_active: bool

    class Config:
        from_attributes = True


class CameraFeedResponse(BaseModel):
    """Camera data shaped for the Live Feeds grid card.

    Includes stream URLs, store context, status badge label,
    and a summary of detection zones drawn on this camera.
    """
    id: UUID
    name: str

    # Store context
    store_id: Optional[UUID] = None
    store_name: Optional[str] = None
    store_zone_gate: Optional[str] = None         # e.g. "Gate B4"

    # Location description (optional free-text)
    location_description: Optional[str] = None

    # Status
    status: str                                    # active | inactive | error | maintenance
    status_display: str                            # LIVE | OFFLINE | RECONNECTING | MAINTENANCE

    # Stream
    stream_path: Optional[str] = None
    webrtc_url: Optional[str] = None
    hls_url: Optional[str] = None

    # Detection zones on this camera
    zones: List[CameraFeedZoneSummary] = []
    zone_count: int = 0

    class Config:
        from_attributes = True


# Mapping from DB status → display badge string
_STATUS_DISPLAY_MAP: Dict[str, str] = {
    "active":      "LIVE",
    "inactive":    "OFFLINE",
    "error":       "RECONNECTING",
    "maintenance": "MAINTENANCE",
}


def _extract_protocol(rtsp_url: str) -> str:
    """Derive a short protocol label from the camera URL scheme."""
    url = (rtsp_url or "").lower()
    if url.startswith("rtsp"):
        return "RTSP"
    if url.startswith("rtmp"):
        return "RTMP"
    if url.startswith("https"):
        return "HTTPS"
    if url.startswith("http"):
        return "HTTP"
    return url.split("://")[0].upper() if "://" in url else "RTSP"


class DetectionZoneCreate(BaseModel):
    """Create a detection polygon zone on a camera's live feed.

    Called from the polygon editor (eye icon) after the operator draws a
    polygon on the camera stream.  `zone_type` defaults to "footfall" so
    the zone is immediately functional even before the operator picks an
    event type from the dropdown.
    """
    name: str = Field(..., min_length=1, max_length=255, description='Zone label, e.g. "Zone 1", "Counter Area"')
    zone_type: str = Field(
        default="footfall",
        description=(
            "Detection event type for this polygon. "
            "One of: footfall, dwell_time, queue_length, entry_exit, heatmap, purchase_intent. "
            "Defaults to 'footfall'."
        ),
    )
    shape: str = Field(default="polygon", description="polygon | line")
    polygon: Optional[Dict[str, Any]] = Field(
        None,
        description='Polygon points drawn on the camera frame. '
                    'Format: {"points": [[x1,y1],[x2,y2],...]} where coords are pixel offsets.',
    )
    is_active: bool = True


class DetectionZoneUpdate(BaseModel):
    """Update a detection polygon zone — modify polygon points or event type."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    zone_type: Optional[str] = Field(
        None,
        description="Change the detection event type: footfall | dwell_time | queue_length | entry_exit | heatmap | purchase_intent",
    )
    shape: Optional[str] = None
    polygon: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated polygon points drawn on the camera frame.",
    )
    is_active: Optional[bool] = None


class DetectionZoneResponse(BaseModel):
    """Detection polygon zone response — used inside the polygon editor panel."""
    id: UUID
    camera_id: Optional[UUID]
    name: str
    zone_type: str
    zone_type_label: Optional[str] = None  # Human-readable label (e.g. "Footfall")
    shape: str
    polygon: Optional[Dict[str, Any]]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CameraPolygonEditorResponse(BaseModel):
    """Response for the polygon editor panel (triggered by eye icon on camera row).

    Provides everything the frontend needs to render the polygon editor:
    - Camera identity + store context
    - Live stream URLs (WebRTC/HLS from MediaMTX)
    - Existing detection zones (with polygon points + event types)
    - Available event types for the dropdown
    """
    # Camera info
    id: UUID
    name: str
    store_id: Optional[UUID] = None
    store_name: Optional[str] = None
    store_zone_gate: Optional[str] = None
    status: str
    # Stream feed URLs — browser plays this from MediaMTX
    stream_path: Optional[str] = None
    webrtc_url: Optional[str] = None
    hls_url: Optional[str] = None
    # Existing detection zones (drawn polygons)
    zones: List[DetectionZoneResponse] = []
    # Available event types for the zone-type dropdown
    available_event_types: List[Dict[str, str]] = []

    class Config:
        from_attributes = True

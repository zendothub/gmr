"""Camera V2 API routes — store-linked camera management.

V2 replaces the Area concept with direct Store linkage.
When a camera is created via V2:
  - store_id is required (chosen from a dropdown)
  - The store's zone_gate (physical location) is auto-populated in the response
  - The area_id field is gone — area module is preserved but not used in V2

Eye icon → Polygon Editor
  Clicking the eye icon on a camera row calls:
    GET /api/v2/cameras/{camera_id}/polygon-editor
  which returns the live stream URLs + existing detection zones.
  The operator draws polygons on the live feed, then calls:
    POST /api/v2/cameras/{camera_id}/zones   → create a new detection zone
    PUT  /api/v2/cameras/{camera_id}/zones/{zone_id}  → update polygon / event type
    DELETE /api/v2/cameras/{camera_id}/zones/{zone_id} → remove a zone

Zone types in this context  (camera-feed polygon types):
  footfall | dwell_time | queue_length | entry_exit | heatmap | purchase_intent
  Default: footfall — the zone is immediately functional before the operator
  picks an event from the dropdown.

Store-wise analytics:
  GET /api/v2/cameras?store_id=<uuid>   → cameras of a specific store
  GET /api/v2/cameras/by-store          → all stores with their cameras grouped

Two distinct "zone" concepts in this system:
  1. Store zones  — physical gate/location labels (e.g. "Gate B4")
                    managed via /api/stores/zones
  2. Detection zones — polygons drawn on a camera's live feed
                    managed via /api/v2/cameras/{id}/zones  ← this file
"""

from typing import List, Optional, Dict
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.core.db.models.camera import Camera, CameraStatus, Zone
from app.core.db.models.store import Store
from app.modules.jobs.tasks import _probe_rtsp as _rtsp_probe
from app.modules.cameras.schemas import (
    CameraCreateV2,
    CameraUpdateV2,
    CameraResponse,
    CameraByStoreResponse,
    CameraFeedResponse,
    CameraFeedZoneSummary,
    _STATUS_DISPLAY_MAP,
    DetectionZoneCreate,
    DetectionZoneUpdate,
    DetectionZoneResponse,
    CameraPolygonEditorResponse,
    DETECTION_EVENT_TYPES,
    DETECTION_EVENT_LABELS,
)
from app.modules.cameras.service import CameraService
from app.modules.streaming.mediamtx import MediaMTXManager

v2_router = APIRouter(prefix="/api/v2/cameras", tags=["Cameras V2"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_detection_zone_response(zone: Zone) -> DetectionZoneResponse:
    """Serialize a Zone ORM object → DetectionZoneResponse with human label."""
    return DetectionZoneResponse(
        id=zone.id,
        camera_id=zone.camera_id,
        name=zone.name,
        zone_type=zone.zone_type,
        zone_type_label=DETECTION_EVENT_LABELS.get(zone.zone_type, zone.zone_type),
        shape=zone.shape,
        polygon=zone.polygon,
        is_active=zone.is_active,
        created_at=zone.created_at,
        updated_at=zone.updated_at,
    )


def _available_event_types() -> List[Dict[str, str]]:
    """Return the list of detection event type choices for the dropdown."""
    return [
        {"value": k, "label": v}
        for k, v in DETECTION_EVENT_LABELS.items()
    ]


async def _get_camera_or_404(db: AsyncSession, camera_id: UUID) -> Camera:
    """Fetch camera by ID with store + store_zone relationships eagerly loaded.

    In async SQLAlchemy, lazy-loading a relationship (camera.store) raises
    MissingGreenlet when store_id is set.  We use selectinload so that
    ``camera.store`` and ``camera.store_zone`` are always available without
    extra round-trips.
    """
    result = await db.execute(
        select(Camera)
        .options(selectinload(Camera.store), selectinload(Camera.store_zone))
        .where(Camera.id == camera_id)
    )
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


async def _get_zone_or_404(db: AsyncSession, camera_id: UUID, zone_id: UUID) -> Zone:
    result = await db.execute(
        select(Zone).where(Zone.id == zone_id, Zone.camera_id == camera_id)
    )
    zone = result.scalar_one_or_none()
    if not zone:
        raise HTTPException(status_code=404, detail="Detection zone not found for this camera")
    return zone


# ─────────────────────────────────────────────────────────────────────────────
# Camera CRUD (V2)
# ─────────────────────────────────────────────────────────────────────────────

@v2_router.post("", response_model=CameraResponse, status_code=201)
async def create_camera_v2(
    request: Request,
    data: CameraCreateV2,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new camera linked to a store (V2).

    The frontend shows a **Store** dropdown — selecting a store auto-populates
    the store's `zone_gate` (physical location, e.g. "Gate B4").  The camera
    is linked to that store via `store_id`.

    After creation, click the **eye icon** on the camera row to open the live
    stream and draw detection polygon zones.
    """
    # Validate store exists
    store_result = await db.execute(select(Store).where(Store.id == data.store_id))
    store = store_result.scalar_one_or_none()
    if not store:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Store not found: {data.store_id}",
        )

    camera = await CameraService.create_camera_v2(db, data)
    # Re-fetch with store eagerly loaded so build_response can access camera.store
    camera = await _get_camera_or_404(db, camera.id)
    return CameraService.build_response(camera, public_host=request.url.hostname)


@v2_router.get("", response_model=List[CameraResponse])
async def list_cameras_v2(
    request: Request,
    store_id: Optional[UUID] = Query(
        None,
        description="Filter by store ID — returns only cameras linked to that store.",
    ),
    name: Optional[str] = Query(
        None,
        description=(
            "Case-insensitive partial name search. "
            "e.g. `name=entry` matches 'Entry Gate Cam', 'Main Entry'."
        ),
    ),
    status: Optional[str] = Query(
        None,
        description=(
            "Filter by camera status. "
            "Allowed values: `active` | `inactive` | `maintenance` | `error`."
        ),
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List cameras (V2) — supports combined filtering.

    All query params are optional and composable:

    | Param | Effect |
    |---|---|
    | `store_id` | Only cameras linked to that store |
    | `name` | Case-insensitive partial match on camera name |
    | `status` | Exact match on camera status (active / inactive / maintenance / error) |

    Examples:
    - `GET /api/v2/cameras` → all cameras
    - `GET /api/v2/cameras?store_id=xxx` → cameras of store xxx
    - `GET /api/v2/cameras?status=active` → all active cameras
    - `GET /api/v2/cameras?store_id=xxx&status=inactive` → inactive cameras of store xxx
    - `GET /api/v2/cameras?name=aisle&status=active` → active cameras matching "aisle"
    """
    # Validate status value if provided
    # NOTE: can't use fastapi.status here — the local `status` variable shadows the module.
    valid_statuses = {"active", "inactive", "maintenance", "error"}
    if status and status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{status}'. Must be one of: {', '.join(sorted(valid_statuses))}",
        )

    query = (
        select(Camera)
        .options(selectinload(Camera.store), selectinload(Camera.store_zone))
        .order_by(Camera.created_at.desc())
    )

    if store_id:
        query = query.where(Camera.store_id == store_id)

    if name:
        # ilike = case-insensitive LIKE (PostgreSQL)
        query = query.where(Camera.name.ilike(f"%{name}%"))

    if status:
        query = query.where(Camera.status == status)

    result = await db.execute(query)
    cameras = list(result.scalars().all())
    return [CameraService.build_response(c, public_host=request.url.hostname) for c in cameras]


@v2_router.get("/by-store", response_model=List[CameraByStoreResponse])
async def cameras_by_store(
    request: Request,
    store_id: Optional[UUID] = Query(
        None,
        description="Optional store filter. If omitted, returns all stores with their cameras.",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get cameras grouped by store.

    Returns a list where each item is a store with its list of cameras.
    Use this for **store-wise analytics** — fetch all cameras for a store
    then query analytics per camera via `/api/analytics/*?store_id=...`.
    """
    store_query = select(Store).order_by(Store.name)
    if store_id:
        store_query = store_query.where(Store.id == store_id)
    store_result = await db.execute(store_query)
    stores = list(store_result.scalars().all())

    grouped: List[CameraByStoreResponse] = []
    for store in stores:
        cam_result = await db.execute(
            select(Camera)
            # Eagerly load store + store_zone so build_response can access both
            .options(selectinload(Camera.store), selectinload(Camera.store_zone))
            .where(Camera.store_id == store.id)
            .order_by(Camera.created_at.desc())
        )
        store_cameras = list(cam_result.scalars().all())
        if store_cameras or store_id:  # always include if specifically filtered
            grouped.append(
                CameraByStoreResponse(
                    store_id=store.id,
                    store_name=store.name,
                    store_zone_gate=store.zone_gate,
                    cameras=[
                        CameraService.build_response(c, public_host=request.url.hostname)
                        for c in store_cameras
                    ],
                )
            )

    return grouped


@v2_router.get("/feeds", response_model=List[CameraFeedResponse])
async def list_camera_feeds(
    request: Request,
    store_id: Optional[UUID] = Query(
        None,
        description=(
            "Filter by store. Omit for all stores, pass a UUID for a specific store. "
            "Matches the 'All Stores' / store dropdown on the Live Feeds page."
        ),
    ),
    status: Optional[str] = Query(
        None,
        description=(
            "Filter by camera status. "
            "Allowed values: `active` | `inactive` | `maintenance` | `error`. "
            "Omit for all statuses."
        ),
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Live Feeds grid data — cameras with stream URLs, store context, and zone summaries.

    Designed for the **Live Feeds** page — returns exactly the data each feed card needs:

    | Field | Description |
    |---|---|
    | `webrtc_url` / `hls_url` | Browser-playable stream URL from MediaMTX |
    | `status_display` | Badge label: LIVE / OFFLINE / RECONNECTING / MAINTENANCE |
    | `store_name` | Store this camera belongs to |
    | `store_zone_gate` | Physical location, e.g. "Gate B4" |
    | `zones` | Detection zones drawn on this camera |
    | `zone_count` | Number of zones |

    **Query params (both optional):**
    - `store_id` — All Stores dropdown: pass a store UUID or omit for all
    - `status` — Status filter dropdown: `active` / `inactive` / `maintenance` / `error`
    """
    valid_statuses = {"active", "inactive", "maintenance", "error"}
    if status and status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{status}'. Allowed: {', '.join(sorted(valid_statuses))}",
        )

    query = (
        select(Camera)
        # Eagerly load store (camera.store.name / .zone_gate) and zones list
        .options(
            selectinload(Camera.store),
            selectinload(Camera.zones),
        )
        .order_by(Camera.created_at.desc())
    )

    if store_id:
        query = query.where(Camera.store_id == store_id)

    if status:
        query = query.where(Camera.status == status)

    result = await db.execute(query)
    cameras = list(result.scalars().all())

    feeds: List[CameraFeedResponse] = []
    for camera in cameras:
        endpoints = MediaMTXManager().endpoints(camera.id, public_host=request.url.hostname)
        stream_path = camera.stream_path or endpoints.path

        zone_summaries = [
            CameraFeedZoneSummary(
                id=z.id,
                name=z.name,
                zone_type=z.zone_type,
                zone_type_label=DETECTION_EVENT_LABELS.get(z.zone_type, z.zone_type),
                is_active=z.is_active,
            )
            for z in camera.zones
        ]

        feeds.append(
            CameraFeedResponse(
                id=camera.id,
                name=camera.name,
                store_id=camera.store_id,
                store_name=camera.store.name if camera.store else None,
                store_zone_gate=camera.store.zone_gate if camera.store else None,
                location_description=camera.location_description,
                status=camera.status,
                status_display=_STATUS_DISPLAY_MAP.get(camera.status, camera.status.upper()),
                stream_path=stream_path,
                webrtc_url=endpoints.webrtc_url,
                hls_url=endpoints.hls_url,
                zones=zone_summaries,
                zone_count=len(zone_summaries),
            )
        )

    return feeds


@v2_router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera_v2(
    request: Request,
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get camera details by ID (V2 — includes store info)."""
    # Use _get_camera_or_404 (not CameraService.get_camera) so that camera.store
    # is eagerly loaded and build_response doesn't trigger a lazy SELECT.
    camera = await _get_camera_or_404(db, camera_id)
    return CameraService.build_response(camera, public_host=request.url.hostname)


@v2_router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera_v2(
    request: Request,
    camera_id: UUID,
    data: CameraUpdateV2,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update a camera's name, RTSP URL, or store link (V2).

    Selecting a different store from the dropdown re-links the camera and
    auto-populates the new store's `zone_gate` in the response.
    """
    camera = await _get_camera_or_404(db, camera_id)

    # If re-linking to a different store, validate it exists
    if data.store_id and data.store_id != camera.store_id:
        store_result = await db.execute(select(Store).where(Store.id == data.store_id))
        store = store_result.scalar_one_or_none()
        if not store:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Store not found: {data.store_id}",
            )

    update_data = data.model_dump(exclude_unset=True)

    # If RTSP URL changed, test the new URL
    if "rtsp_url" in update_data and update_data["rtsp_url"] != camera.rtsp_url:
        test_result = CameraService.test_rtsp_stream(update_data["rtsp_url"])
        if not test_result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"New RTSP stream test failed: {test_result.message}",
            )

    for key, value in update_data.items():
        setattr(camera, key, value)

    await db.flush()
    # Re-fetch with store eagerly loaded: db.refresh() expires the store
    # relationship, so build_response would face a lazy-load in async context.
    camera = await _get_camera_or_404(db, camera_id)
    return CameraService.build_response(camera, public_host=request.url.hostname)


@v2_router.delete("/{camera_id}", status_code=200)
async def delete_camera_v2(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a camera (V2).

    Stops the camera worker + stream publisher (ffmpeg/watchdog) before
    removing the database row.  All detection zones are cascade-deleted.
    """
    return await CameraService.delete_camera(db, camera_id)


@v2_router.post("/{camera_id}/check-status", response_model=CameraResponse)
async def check_camera_status(
    request: Request,
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Probe a single camera's RTSP stream on-demand and return its updated model.

    Immediately tests whether the camera's RTSP URL is reachable, updates
    `camera.status` in the database, and returns the refreshed camera object.

    Status rules (same as the background job):
    - Camera in **MAINTENANCE** → status is **not changed** (operator-set).
    - RTSP reachable → `active`
    - RTSP unreachable → `inactive`

    Use this when you need a guaranteed fresh status instead of waiting for
    the next 2-minute background probe cycle.
    """
    camera = await _get_camera_or_404(db, camera_id)

    if camera.status == CameraStatus.MAINTENANCE:
        # Do not override operator-set maintenance status
        return CameraService.build_response(camera, public_host=request.url.hostname)

    # Run the blocking cv2 RTSP probe in a thread so we don't stall the event loop
    rtsp_url = camera.rtsp_url
    reachable: bool = await anyio.to_thread.run_sync(
        lambda: _rtsp_probe(rtsp_url)
    )

    new_status = CameraStatus.ACTIVE if reachable else CameraStatus.INACTIVE
    if camera.status != new_status:
        camera.status = new_status
        await db.flush()

    # Re-fetch to get the freshest state with store loaded
    camera = await _get_camera_or_404(db, camera_id)
    return CameraService.build_response(camera, public_host=request.url.hostname)


# ─────────────────────────────────────────────────────────────────────────────
# Polygon Editor  (eye icon)
# ─────────────────────────────────────────────────────────────────────────────

@v2_router.get("/{camera_id}/polygon-editor", response_model=CameraPolygonEditorResponse)
async def get_polygon_editor(
    request: Request,
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Open the polygon editor for a camera (triggered by eye icon on camera row).

    Returns:
    - Camera identity + store context (name, zone_gate)
    - Live stream URLs (WebRTC/HLS) so the frontend can render the feed
    - All existing detection zones with their polygon points and event types
    - `available_event_types` list for the event-type dropdown

    Typical flow:
      1. User clicks eye icon → GET /api/v2/cameras/{id}/polygon-editor
      2. Frontend renders live stream + existing polygons
      3. User draws a new polygon → POST /api/v2/cameras/{id}/zones
      4. User changes event type dropdown → PUT /api/v2/cameras/{id}/zones/{zone_id}
    """
    camera = await _get_camera_or_404(db, camera_id)

    # Fetch all detection zones for this camera
    zones_result = await db.execute(
        select(Zone).where(Zone.camera_id == camera_id).order_by(Zone.created_at)
    )
    zones = list(zones_result.scalars().all())

    # Build stream URLs (same logic as CameraService.build_response)
    endpoints = MediaMTXManager().endpoints(camera.id, public_host=request.url.hostname)
    stream_path = camera.stream_path or endpoints.path
    webrtc_url = endpoints.webrtc_url
    hls_url = endpoints.hls_url

    return CameraPolygonEditorResponse(
        id=camera.id,
        name=camera.name,
        store_id=camera.store_id,
        store_name=camera.store.name if camera.store else None,
        store_zone_gate=camera.store.zone_gate if camera.store else None,
        status=camera.status,
        stream_path=stream_path,
        webrtc_url=webrtc_url,
        hls_url=hls_url,
        zones=[_build_detection_zone_response(z) for z in zones],
        available_event_types=_available_event_types(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Detection Zones CRUD  (polygon editor → zone management)
# ─────────────────────────────────────────────────────────────────────────────

@v2_router.get("/{camera_id}/zones", response_model=List[DetectionZoneResponse])
async def list_detection_zones(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all detection polygon zones for a camera."""
    await _get_camera_or_404(db, camera_id)
    result = await db.execute(
        select(Zone).where(Zone.camera_id == camera_id).order_by(Zone.created_at)
    )
    zones = list(result.scalars().all())
    return [_build_detection_zone_response(z) for z in zones]


@v2_router.post("/{camera_id}/zones", response_model=DetectionZoneResponse, status_code=201)
async def create_detection_zone(
    camera_id: UUID,
    data: DetectionZoneCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a detection polygon zone on a camera's live feed.

    Called after the operator draws a polygon in the polygon editor.

    - `zone_type` defaults to **"footfall"** — the zone is immediately active.
    - Change the event type later via **PUT /zones/{zone_id}**.
    - `polygon` format: `{"points": [[x1,y1],[x2,y2],...]}`

    Valid `zone_type` values:
    `footfall` | `dwell_time` | `queue_length` | `entry_exit` | `heatmap` | `purchase_intent`
    """
    await _get_camera_or_404(db, camera_id)

    # Validate zone_type is a recognised detection event type
    valid_types = set(DETECTION_EVENT_TYPES) | {
        # Also allow legacy zone_type values for backward compat
        "entry_line", "exit_line", "billing_zone", "queue_zone",
        "product_zone", "ignore_zone", "restricted_zone", "medicine_pickup_zone",
    }
    if data.zone_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid zone_type '{data.zone_type}'. "
                f"Valid detection event types: {', '.join(DETECTION_EVENT_TYPES)}"
            ),
        )

    zone = Zone(
        camera_id=camera_id,
        name=data.name,
        zone_type=data.zone_type,
        shape=data.shape,
        polygon=data.polygon,
        is_active=data.is_active,
    )
    db.add(zone)
    await db.flush()
    await db.refresh(zone)
    return _build_detection_zone_response(zone)


@v2_router.put("/{camera_id}/zones/{zone_id}", response_model=DetectionZoneResponse)
async def update_detection_zone(
    camera_id: UUID,
    zone_id: UUID,
    data: DetectionZoneUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update a detection polygon zone.

    Use this to:
    - Save polygon points after the operator adjusts the drawn shape
    - Change the detection event type from the dropdown
      (e.g. switch from "footfall" to "dwell_time")
    """
    zone = await _get_zone_or_404(db, camera_id, zone_id)

    update_data = data.model_dump(exclude_unset=True)

    # Validate new zone_type if provided
    if "zone_type" in update_data:
        valid_types = set(DETECTION_EVENT_TYPES) | {
            "entry_line", "exit_line", "billing_zone", "queue_zone",
            "product_zone", "ignore_zone", "restricted_zone", "medicine_pickup_zone",
        }
        if update_data["zone_type"] not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid zone_type '{update_data['zone_type']}'. "
                    f"Valid detection event types: {', '.join(DETECTION_EVENT_TYPES)}"
                ),
            )

    for key, value in update_data.items():
        setattr(zone, key, value)

    await db.flush()
    await db.refresh(zone)
    return _build_detection_zone_response(zone)


@v2_router.delete("/{camera_id}/zones/{zone_id}", status_code=200)
async def delete_detection_zone(
    camera_id: UUID,
    zone_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a detection polygon zone from the polygon editor."""
    zone = await _get_zone_or_404(db, camera_id, zone_id)
    zone_name = zone.name
    await db.delete(zone)
    await db.flush()
    return {"message": f"Detection zone '{zone_name}' deleted successfully"}


# ─────────────────────────────────────────────────────────────────────────────
# Event types reference  (for frontend dropdown population)
# ─────────────────────────────────────────────────────────────────────────────

@v2_router.get("/meta/detection-event-types", response_model=List[Dict[str, str]], tags=["Cameras V2"])
async def list_detection_event_types(
    current_user: User = Depends(get_current_user),
):
    """Return the list of valid detection event types for the polygon editor dropdown.

    Response shape: `[{"value": "footfall", "label": "Footfall"}, ...]`
    """
    return _available_event_types()

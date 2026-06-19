"""Zone API routes.

Zones are bound to a camera. The typical flow:
  1. GET  /api/cameras/{camera_id}/stream/...   -> view the live stream
  2. POST /api/cameras/{camera_id}/zones        -> create a zone (polygon + type)
  3. repeat (2) to add multiple zones to the same camera
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user, require_role
from app.core.db.models.user import User
from app.core.db.models.camera import ZoneType
from app.modules.zones.schemas import ZoneCreate, ZoneUpdate, ZoneResponse
from app.modules.zones.service import ZoneService

router = APIRouter(tags=["Zones"])


@router.get("/api/zones/types", response_model=List[str])
async def list_zone_types(current_user: User = Depends(get_current_user)):
    """Allowed zone types for the zone-binding dropdown."""
    return [t.value for t in ZoneType]


# ----------------------------------------------------------------------
# Camera-scoped zone management (camera -> many zones)
# ----------------------------------------------------------------------

@router.post("/api/cameras/{camera_id}/zones", response_model=ZoneResponse, status_code=201)
async def create_zone_for_camera(
    camera_id: UUID,
    data: ZoneCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Bind a new zone to a camera (admin only)."""
    zone = await ZoneService.create_zone(db, camera_id, data)
    return ZoneResponse.model_validate(zone)


@router.get("/api/cameras/{camera_id}/zones", response_model=List[ZoneResponse])
async def list_zones_for_camera(
    camera_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all zones bound to a camera."""
    zones = await ZoneService.get_zones_for_camera(db, camera_id)
    return [ZoneResponse.model_validate(z) for z in zones]


# ----------------------------------------------------------------------
# Standalone zone access by id
# ----------------------------------------------------------------------

@router.get("/api/zones", response_model=List[ZoneResponse])
async def list_zones(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all zones across all cameras."""
    zones = await ZoneService.get_all_zones(db)
    return [ZoneResponse.model_validate(z) for z in zones]


@router.get("/api/zones/{zone_id}", response_model=ZoneResponse)
async def get_zone(
    zone_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a zone by ID."""
    zone = await ZoneService.get_zone(db, zone_id)
    return ZoneResponse.model_validate(zone)


@router.put("/api/zones/{zone_id}", response_model=ZoneResponse)
async def update_zone(
    zone_id: UUID,
    data: ZoneUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Update a zone (admin only)."""
    zone = await ZoneService.update_zone(db, zone_id, data)
    return ZoneResponse.model_validate(zone)


@router.delete("/api/zones/{zone_id}")
async def delete_zone(
    zone_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Delete a zone (admin only)."""
    return await ZoneService.delete_zone(db, zone_id)
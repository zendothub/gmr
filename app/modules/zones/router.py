"""Zone API routes."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.zones.schemas import ZoneCreate, ZoneUpdate, ZoneResponse
from app.modules.zones.service import ZoneService

router = APIRouter(tags=["Zones"])


@router.post("/api/camera-views/{view_id}/zones", response_model=ZoneResponse, status_code=201)
async def create_zone(
    view_id: UUID,
    data: ZoneCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a zone inside a camera view."""
    zone = await ZoneService.create_zone(db, view_id, data)
    return ZoneResponse.model_validate(zone)


@router.get("/api/camera-views/{view_id}/zones", response_model=List[ZoneResponse])
async def list_zones_for_view(
    view_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all zones for a camera view."""
    zones = await ZoneService.get_zones_for_view(db, view_id)
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
    current_user: User = Depends(get_current_user),
):
    """Update a zone."""
    zone = await ZoneService.update_zone(db, zone_id, data)
    return ZoneResponse.model_validate(zone)


@router.delete("/api/zones/{zone_id}")
async def delete_zone(
    zone_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a zone."""
    return await ZoneService.delete_zone(db, zone_id)

"""Zone service - CRUD for zones bound to a camera (camera -> many zones)."""

from typing import List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException
from loguru import logger

from app.core.db.models.camera import Zone, Camera
from app.modules.zones.schemas import ZoneCreate, ZoneUpdate


class ZoneService:

    @staticmethod
    async def _ensure_camera(db: AsyncSession, camera_id: UUID) -> Camera:
        result = await db.execute(select(Camera).where(Camera.id == camera_id))
        camera = result.scalar_one_or_none()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        return camera

    @staticmethod
    async def create_zone(db: AsyncSession, camera_id: UUID, data: ZoneCreate) -> Zone:
        """Create a new zone bound to a camera after selecting polygon on its stream."""
        await ZoneService._ensure_camera(db, camera_id)
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
        logger.info(f"Zone created: {zone.name} (type={zone.zone_type}) on camera {camera_id}")
        return zone

    @staticmethod
    async def get_zones_for_camera(db: AsyncSession, camera_id: UUID) -> List[Zone]:
        """Get all zones bound to a camera."""
        result = await db.execute(
            select(Zone).where(Zone.camera_id == camera_id).order_by(Zone.created_at)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_all_zones(db: AsyncSession) -> List[Zone]:
        """Get all zones across all cameras."""
        result = await db.execute(select(Zone).order_by(Zone.created_at))
        return list(result.scalars().all())

    @staticmethod
    async def get_zone(db: AsyncSession, zone_id: UUID) -> Zone:
        """Get a zone by ID."""
        result = await db.execute(select(Zone).where(Zone.id == zone_id))
        zone = result.scalar_one_or_none()
        if not zone:
            raise HTTPException(status_code=404, detail="Zone not found")
        return zone

    @staticmethod
    async def update_zone(db: AsyncSession, zone_id: UUID, data: ZoneUpdate) -> Zone:
        """Update a zone."""
        zone = await ZoneService.get_zone(db, zone_id)
        for key, value in data.model_dump(exclude_unset=True).items():
            setattr(zone, key, value)
        await db.flush()
        await db.refresh(zone)
        logger.info(f"Zone updated: {zone.name} (id={zone_id})")
        return zone

    @staticmethod
    async def delete_zone(db: AsyncSession, zone_id: UUID) -> dict:
        """Delete a zone."""
        zone = await ZoneService.get_zone(db, zone_id)
        await db.delete(zone)
        logger.info(f"Zone deleted: {zone.name} (id={zone_id})")
        return {"message": f"Zone '{zone.name}' deleted successfully"}

"""Zone service - CRUD for zones inside camera views."""

from typing import List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException
from loguru import logger

from app.core.db.models.camera import Zone
from app.modules.zones.schemas import ZoneCreate, ZoneUpdate


class ZoneService:

    @staticmethod
    async def create_zone(db: AsyncSession, view_id: UUID, data: ZoneCreate) -> Zone:
        """Create a zone inside a camera view."""
        zone = Zone(
            camera_view_id=view_id,
            name=data.name,
            zone_type=data.zone_type,
            shape=data.shape,
            polygon=data.polygon,
            line_config=data.line_config,
            color=data.color,
            is_active=data.is_active,
        )
        db.add(zone)
        await db.flush()
        await db.refresh(zone)
        logger.info(f"Zone created: {zone.name} (type={zone.zone_type}) in view {view_id}")
        return zone

    @staticmethod
    async def get_zones_for_view(db: AsyncSession, view_id: UUID) -> List[Zone]:
        """Get all zones for a camera view."""
        result = await db.execute(
            select(Zone)
            .where(Zone.camera_view_id == view_id)
            .order_by(Zone.created_at)
        )
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
        update_data = data.model_dump(exclude_unset=True)

        for key, value in update_data.items():
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

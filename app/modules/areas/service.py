"""Area service - CRUD for independent named areas."""

from typing import List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException
from loguru import logger

from app.core.db.models.area import Area
from app.modules.areas.schemas import AreaCreate, AreaUpdate


class AreaService:

    @staticmethod
    async def create_area(db: AsyncSession, data: AreaCreate) -> Area:
        """Create a new independent area (name only)."""
        area = Area(name=data.name)
        db.add(area)
        await db.flush()
        await db.refresh(area)
        logger.info(f"Area created: {area.name}")
        return area

    @staticmethod
    async def get_areas(db: AsyncSession) -> List[Area]:
        """List all areas (used to populate the camera area dropdown)."""
        result = await db.execute(select(Area).order_by(Area.created_at))
        return list(result.scalars().all())

    @staticmethod
    async def get_area(db: AsyncSession, area_id: UUID) -> Area:
        """Get an area by ID."""
        result = await db.execute(select(Area).where(Area.id == area_id))
        area = result.scalar_one_or_none()
        if not area:
            raise HTTPException(status_code=404, detail="Area not found")
        return area

    @staticmethod
    async def update_area(db: AsyncSession, area_id: UUID, data: AreaUpdate) -> Area:
        """Update an area's name."""
        area = await AreaService.get_area(db, area_id)
        for key, value in data.model_dump(exclude_unset=True).items():
            setattr(area, key, value)
        await db.flush()
        await db.refresh(area)
        logger.info(f"Area updated: {area.name} (id={area_id})")
        return area

    @staticmethod
    async def delete_area(db: AsyncSession, area_id: UUID) -> dict:
        """Delete an area. Cameras referencing it have area_id set to NULL."""
        area = await AreaService.get_area(db, area_id)
        await db.delete(area)
        logger.info(f"Area deleted: {area.name} (id={area_id})")
        return {"message": f"Area '{area.name}' deleted successfully"}

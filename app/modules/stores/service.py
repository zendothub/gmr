"""Stores service - retail outlet management."""

from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.store import Store, StoreStatus
from app.core.db.models.store_lookup import StoreCategory, StoreLevel, StoreZone
from app.modules.stores.schemas import (
    StoreCreate, StoreUpdate,
    StoreCategoryCreate, StoreCategoryUpdate,
    StoreLevelCreate, StoreLevelUpdate,
    StoreZoneCreate, StoreZoneUpdate,
)


class StoreService:

    @staticmethod
    async def create_store(db: AsyncSession, data: StoreCreate) -> Store:
        """Create a new store."""
        store = Store(
            name=data.name,
            category=data.category,
            status=StoreStatus(data.status),
            terminal=data.terminal,
            level=data.level,
            zone_gate=data.zone_gate,
            description=data.description,
        )
        db.add(store)
        await db.flush()
        await db.refresh(store)
        logger.info(f"Store created: {store.name} (id={store.id})")
        return store

    @staticmethod
    async def get_stores(
        db: AsyncSession,
        status_filter: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Store]:
        """List all stores with optional status filter and name search."""
        query = select(Store).order_by(Store.created_at.desc())
        if status_filter:
            query = query.where(Store.status == StoreStatus(status_filter))
        if search:
            query = query.where(Store.name.ilike(f"%{search}%"))
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_store(db: AsyncSession, store_id: UUID) -> Store:
        """Get a single store by ID."""
        result = await db.execute(select(Store).where(Store.id == store_id))
        store = result.scalar_one_or_none()
        if not store:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Store not found")
        return store

    @staticmethod
    async def update_store(db: AsyncSession, store_id: UUID, data: StoreUpdate) -> Store:
        """Update a store."""
        store = await StoreService.get_store(db, store_id)
        update_fields = data.model_dump(exclude_unset=True)
        for field, value in update_fields.items():
            if field == "status":
                setattr(store, field, StoreStatus(value))
            else:
                setattr(store, field, value)
        await db.flush()
        await db.refresh(store)
        logger.info(f"Store updated: {store.name} (id={store_id})")
        return store

    @staticmethod
    async def delete_store(db: AsyncSession, store_id: UUID) -> dict:
        """Delete a store."""
        store = await StoreService.get_store(db, store_id)
        await db.delete(store)
        logger.info(f"Store deleted: {store.name} (id={store_id})")
        return {"message": f"Store '{store.name}' deleted successfully"}

    # ------------------------------------------------------------------
    # StoreCategory CRUD
    # ------------------------------------------------------------------

    @staticmethod
    async def create_category(db: AsyncSession, data: StoreCategoryCreate) -> StoreCategory:
        existing = await db.execute(select(StoreCategory).where(StoreCategory.name == data.name))
        if existing.scalar_one_or_none():
            raise HTTPException(status.HTTP_409_CONFLICT, "Category already exists")
        cat = StoreCategory(name=data.name, description=data.description)
        db.add(cat)
        await db.flush()
        await db.refresh(cat)
        logger.info(f"StoreCategory created: {cat.name}")
        return cat

    @staticmethod
    async def get_categories(db: AsyncSession) -> List[StoreCategory]:
        result = await db.execute(select(StoreCategory).order_by(StoreCategory.name))
        return list(result.scalars().all())

    @staticmethod
    async def get_category(db: AsyncSession, category_id: UUID) -> StoreCategory:
        result = await db.execute(select(StoreCategory).where(StoreCategory.id == category_id))
        cat = result.scalar_one_or_none()
        if not cat:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Category not found")
        return cat

    @staticmethod
    async def update_category(db: AsyncSession, category_id: UUID, data: StoreCategoryUpdate) -> StoreCategory:
        cat = await StoreService.get_category(db, category_id)
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(cat, field, value)
        await db.flush()
        await db.refresh(cat)
        return cat

    @staticmethod
    async def delete_category(db: AsyncSession, category_id: UUID) -> dict:
        cat = await StoreService.get_category(db, category_id)
        await db.delete(cat)
        return {"message": f"Category '{cat.name}' deleted successfully"}

    # ------------------------------------------------------------------
    # StoreLevel CRUD
    # ------------------------------------------------------------------

    @staticmethod
    async def create_level(db: AsyncSession, data: StoreLevelCreate) -> StoreLevel:
        existing = await db.execute(select(StoreLevel).where(StoreLevel.name == data.name))
        if existing.scalar_one_or_none():
            raise HTTPException(status.HTTP_409_CONFLICT, "Level already exists")
        lvl = StoreLevel(name=data.name, description=data.description)
        db.add(lvl)
        await db.flush()
        await db.refresh(lvl)
        logger.info(f"StoreLevel created: {lvl.name}")
        return lvl

    @staticmethod
    async def get_levels(db: AsyncSession) -> List[StoreLevel]:
        result = await db.execute(select(StoreLevel).order_by(StoreLevel.name))
        return list(result.scalars().all())

    @staticmethod
    async def get_level(db: AsyncSession, level_id: UUID) -> StoreLevel:
        result = await db.execute(select(StoreLevel).where(StoreLevel.id == level_id))
        lvl = result.scalar_one_or_none()
        if not lvl:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Level not found")
        return lvl

    @staticmethod
    async def update_level(db: AsyncSession, level_id: UUID, data: StoreLevelUpdate) -> StoreLevel:
        lvl = await StoreService.get_level(db, level_id)
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(lvl, field, value)
        await db.flush()
        await db.refresh(lvl)
        return lvl

    @staticmethod
    async def delete_level(db: AsyncSession, level_id: UUID) -> dict:
        lvl = await StoreService.get_level(db, level_id)
        await db.delete(lvl)
        return {"message": f"Level '{lvl.name}' deleted successfully"}

    # ------------------------------------------------------------------
    # StoreZone CRUD  (physical gate/zone labels — NOT camera zones)
    # ------------------------------------------------------------------

    @staticmethod
    async def create_store_zone(db: AsyncSession, data: StoreZoneCreate) -> StoreZone:
        existing = await db.execute(select(StoreZone).where(StoreZone.name == data.name))
        if existing.scalar_one_or_none():
            raise HTTPException(status.HTTP_409_CONFLICT, "Store zone already exists")
        zone = StoreZone(name=data.name, terminal=data.terminal, description=data.description)
        db.add(zone)
        await db.flush()
        await db.refresh(zone)
        logger.info(f"StoreZone created: {zone.name}")
        return zone

    @staticmethod
    async def get_store_zones(db: AsyncSession) -> List[StoreZone]:
        result = await db.execute(select(StoreZone).order_by(StoreZone.name))
        return list(result.scalars().all())

    @staticmethod
    async def get_store_zone(db: AsyncSession, zone_id: UUID) -> StoreZone:
        result = await db.execute(select(StoreZone).where(StoreZone.id == zone_id))
        zone = result.scalar_one_or_none()
        if not zone:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Store zone not found")
        return zone

    @staticmethod
    async def update_store_zone(db: AsyncSession, zone_id: UUID, data: StoreZoneUpdate) -> StoreZone:
        zone = await StoreService.get_store_zone(db, zone_id)
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(zone, field, value)
        await db.flush()
        await db.refresh(zone)
        return zone

    @staticmethod
    async def delete_store_zone(db: AsyncSession, zone_id: UUID) -> dict:
        zone = await StoreService.get_store_zone(db, zone_id)
        await db.delete(zone)
        return {"message": f"Store zone '{zone.name}' deleted successfully"}

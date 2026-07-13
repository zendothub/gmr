"""Stores service - retail outlet management."""

from typing import List, Optional
from uuid import UUID

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.camera import Camera
from app.core.db.models.store import Store, StoreStatus
from app.core.db.models.store_lookup import StoreCategory, StoreLevel, StoreZone, StoreTerminal
from app.modules.stores.schemas import (
    StoreCreate, StoreUpdate, StoreStatusUpdate,
    StoreCategoryCreate, StoreCategoryUpdate,
    StoreLevelCreate, StoreLevelUpdate,
    StoreZoneCreate, StoreZoneUpdate,
    StoreTerminalCreate, StoreTerminalUpdate,
)


async def _build_store_counts(db: AsyncSession, store_ids: List[UUID]) -> dict:
    """
    For a list of store IDs, compute three live counts that exactly match
    what the analytics page shows:

      - camera_count   : number of cameras linked to the store (direct SQL)
      - footfall_count : total_visitors from AnalyticsService (metric=footfall, today)
      - purchase_count : total_purchases from AnalyticsService (metric=purchase, today)

    By delegating footfall and purchase to AnalyticsService.get_analytics_metrics()
    we guarantee the store list shows the SAME numbers as the analytics page.

    Returns a dict keyed by store_id (UUID) → {"camera_count": N, "footfall_count": N, "purchase_count": N}
    """
    if not store_ids:
        return {}

    # Lazy import to avoid circular dependency
    from app.modules.analytics.service import AnalyticsService

    # ── camera_count per store (simple direct count) ─────────────────────
    cam_q = (
        select(Camera.store_id, func.count(Camera.id).label("cnt"))
        .where(Camera.store_id.in_(store_ids))
        .group_by(Camera.store_id)
    )
    cam_rows = (await db.execute(cam_q)).all()
    cam_map: dict = {row.store_id: row.cnt for row in cam_rows}

    # ── footfall_count + purchase_count via AnalyticsService ─────────────
    # Calls the same service function the analytics page uses so the numbers
    # are guaranteed to match.  Sequential per-store to keep the shared
    # AsyncSession safe (no concurrent coroutines on one session).
    result: dict = {}
    for sid in store_ids:
        ff_count = 0
        pur_count = 0

        try:
            ff_resp = await AnalyticsService.get_analytics_metrics(
                db, metric="footfall", store_id=sid, time_range="today"
            )
            ff_count = ff_resp.footfall_data.total_visitors if ff_resp.footfall_data else 0
        except Exception as exc:
            logger.warning(f"Store footfall count failed for {sid}: {exc}")

        try:
            pur_resp = await AnalyticsService.get_analytics_metrics(
                db, metric="purchase", store_id=sid, time_range="today"
            )
            pur_count = pur_resp.purchase_data.total_purchases if pur_resp.purchase_data else 0
        except Exception as exc:
            logger.warning(f"Store purchase count failed for {sid}: {exc}")

        result[sid] = {
            "camera_count": cam_map.get(sid, 0),
            "footfall_count": ff_count,
            "purchase_count": pur_count,
        }

    return result


def _store_to_dict(store: Store, counts: dict) -> dict:
    """Merge a Store ORM object with live-computed counts into a plain dict for Pydantic."""
    return {
        "id": store.id,
        "name": store.name,
        "category": store.category,
        "status": store.status.value if hasattr(store.status, "value") else store.status,
        "terminal": store.terminal,
        "level": store.level,
        "zone_gate": store.zone_gate,
        "description": store.description,
        "footfall_count": counts.get("footfall_count", 0),
        "purchase_count": counts.get("purchase_count", 0),
        "camera_count": counts.get("camera_count", 0),
        "created_at": store.created_at,
        "updated_at": store.updated_at,
    }


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
    async def search_stores(
        db: AsyncSession,
        status_filter: Optional[str] = None,
        name_prefix: Optional[str] = None,
    ) -> List[Store]:
        """Search stores by name prefix and/or status.

        - status_filter=None  → return all statuses
        - status_filter='active' | 'inactive'  → filter by that status
        - name_prefix='Apo'  → returns stores whose name starts with 'Apo'
        """
        query = select(Store).order_by(Store.name)
        if status_filter:
            query = query.where(Store.status == StoreStatus(status_filter))
        if name_prefix:
            # Prefix (startswith) search  e.g. 'Apo%'
            query = query.where(Store.name.ilike(f"{name_prefix}%"))
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
    async def update_store_status(db: AsyncSession, store_id: UUID, data: StoreStatusUpdate) -> Store:
        """Change only the active/inactive status of a store."""
        store = await StoreService.get_store(db, store_id)
        store.status = StoreStatus(data.status)
        await db.flush()
        await db.refresh(store)
        logger.info(f"Store status changed to '{data.status}': {store.name} (id={store_id})")
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

    # ------------------------------------------------------------------
    # StoreTerminal CRUD
    # ------------------------------------------------------------------

    @staticmethod
    async def create_terminal(db: AsyncSession, data: StoreTerminalCreate) -> StoreTerminal:
        existing = await db.execute(select(StoreTerminal).where(StoreTerminal.name == data.name))
        if existing.scalar_one_or_none():
            raise HTTPException(status.HTTP_409_CONFLICT, "Terminal already exists")
        term = StoreTerminal(name=data.name)
        db.add(term)
        await db.flush()
        await db.refresh(term)
        logger.info(f"StoreTerminal created: {term.name}")
        return term

    @staticmethod
    async def get_terminals(db: AsyncSession) -> List[StoreTerminal]:
        result = await db.execute(select(StoreTerminal).order_by(StoreTerminal.name))
        return list(result.scalars().all())

    @staticmethod
    async def get_terminal(db: AsyncSession, terminal_id: UUID) -> StoreTerminal:
        result = await db.execute(select(StoreTerminal).where(StoreTerminal.id == terminal_id))
        term = result.scalar_one_or_none()
        if not term:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Terminal not found")
        return term

    @staticmethod
    async def update_terminal(db: AsyncSession, terminal_id: UUID, data: StoreTerminalUpdate) -> StoreTerminal:
        term = await StoreService.get_terminal(db, terminal_id)
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(term, field, value)
        await db.flush()
        await db.refresh(term)
        return term

    @staticmethod
    async def delete_terminal(db: AsyncSession, terminal_id: UUID) -> dict:
        term = await StoreService.get_terminal(db, terminal_id)
        await db.delete(term)
        return {"message": f"Terminal '{term.name}' deleted successfully"}

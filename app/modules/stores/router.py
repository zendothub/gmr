"""Stores API routes — stores + lookup tables (categories, levels, zones)."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.stores.schemas import (
    StoreCreate, StoreUpdate, StoreStatusUpdate, StoreResponse,
    StoreCategoryCreate, StoreCategoryUpdate, StoreCategoryResponse,
    StoreLevelCreate, StoreLevelUpdate, StoreLevelResponse,
    StoreZoneCreate, StoreZoneUpdate, StoreZoneResponse,
)
from app.modules.stores.service import StoreService

router = APIRouter(prefix="/api/stores", tags=["Stores"])

# ===========================================================================
# Categories  — /api/stores/categories
# ===========================================================================

@router.post("/categories", response_model=StoreCategoryResponse, status_code=201)
async def create_category(
    data: StoreCategoryCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new store category (e.g. Retail, Pharmacy, F&B)."""
    cat = await StoreService.create_category(db, data)
    return StoreCategoryResponse.model_validate(cat)


@router.get("/categories", response_model=List[StoreCategoryResponse])
async def list_categories(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all store categories."""
    cats = await StoreService.get_categories(db)
    return [StoreCategoryResponse.model_validate(c) for c in cats]


@router.get("/categories/{category_id}", response_model=StoreCategoryResponse)
async def get_category(
    category_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a store category by ID."""
    cat = await StoreService.get_category(db, category_id)
    return StoreCategoryResponse.model_validate(cat)


@router.put("/categories/{category_id}", response_model=StoreCategoryResponse)
async def update_category(
    category_id: UUID,
    data: StoreCategoryUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update a store category."""
    cat = await StoreService.update_category(db, category_id, data)
    return StoreCategoryResponse.model_validate(cat)


@router.delete("/categories/{category_id}")
async def delete_category(
    category_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a store category."""
    return await StoreService.delete_category(db, category_id)


# ===========================================================================
# Levels  — /api/stores/levels
# ===========================================================================

@router.post("/levels", response_model=StoreLevelResponse, status_code=201)
async def create_level(
    data: StoreLevelCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new airport level (e.g. Level 1, Level 2)."""
    lvl = await StoreService.create_level(db, data)
    return StoreLevelResponse.model_validate(lvl)


@router.get("/levels", response_model=List[StoreLevelResponse])
async def list_levels(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all airport levels."""
    lvls = await StoreService.get_levels(db)
    return [StoreLevelResponse.model_validate(l) for l in lvls]


@router.get("/levels/{level_id}", response_model=StoreLevelResponse)
async def get_level(
    level_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get an airport level by ID."""
    lvl = await StoreService.get_level(db, level_id)
    return StoreLevelResponse.model_validate(lvl)


@router.put("/levels/{level_id}", response_model=StoreLevelResponse)
async def update_level(
    level_id: UUID,
    data: StoreLevelUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update an airport level."""
    lvl = await StoreService.update_level(db, level_id, data)
    return StoreLevelResponse.model_validate(lvl)


@router.delete("/levels/{level_id}")
async def delete_level(
    level_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete an airport level."""
    return await StoreService.delete_level(db, level_id)


# ===========================================================================
# Store Zones  — /api/stores/zones
# (physical gate/location labels — NOT camera detection zones)
# ===========================================================================

@router.post("/zones", response_model=StoreZoneResponse, status_code=201)
async def create_store_zone(
    data: StoreZoneCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new physical airport zone/gate label (e.g. Gate B4).
    This is a store location reference — NOT a camera detection zone.
    """
    zone = await StoreService.create_store_zone(db, data)
    return StoreZoneResponse.model_validate(zone)


@router.get("/zones", response_model=List[StoreZoneResponse])
async def list_store_zones(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all physical airport zones/gates."""
    zones = await StoreService.get_store_zones(db)
    return [StoreZoneResponse.model_validate(z) for z in zones]


@router.get("/zones/{zone_id}", response_model=StoreZoneResponse)
async def get_store_zone(
    zone_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a physical airport zone by ID."""
    zone = await StoreService.get_store_zone(db, zone_id)
    return StoreZoneResponse.model_validate(zone)


@router.put("/zones/{zone_id}", response_model=StoreZoneResponse)
async def update_store_zone(
    zone_id: UUID,
    data: StoreZoneUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update a physical airport zone."""
    zone = await StoreService.update_store_zone(db, zone_id, data)
    return StoreZoneResponse.model_validate(zone)


@router.delete("/zones/{zone_id}")
async def delete_store_zone(
    zone_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a physical airport zone."""
    return await StoreService.delete_store_zone(db, zone_id)


# ===========================================================================
# Search  — /api/stores/search
# (static path — must remain BEFORE /{store_id})
# ===========================================================================

@router.get("/search", response_model=List[StoreResponse])
async def search_stores(
    status: Optional[str] = Query(
        None,
        pattern="^(active|inactive)$",
        description="Filter by status. Omit or pass null for all stores.",
    ),
    name: Optional[str] = Query(
        None,
        description="Prefix search on store name — e.g. 'Apo' matches 'Apollo Pharmacy'.",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Search/filter stores.

    - **status**: `active` | `inactive` | omit for all
    - **name**: store name prefix (case-insensitive)

    Examples:
    - `GET /api/stores/search` → all stores
    - `GET /api/stores/search?status=active` → active stores only
    - `GET /api/stores/search?name=Apo` → stores starting with "Apo"
    - `GET /api/stores/search?status=active&name=Apo` → active stores starting with "Apo"
    """
    stores = await StoreService.search_stores(db, status_filter=status, name_prefix=name)
    return [StoreResponse.model_validate(s) for s in stores]


# ===========================================================================
# Stores  — /api/stores  (must come AFTER the static sub-paths above)
# ===========================================================================

@router.post("", response_model=StoreResponse, status_code=201)
async def create_store(
    data: StoreCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new retail store."""
    store = await StoreService.create_store(db, data)
    return StoreResponse.model_validate(store)


@router.get("", response_model=List[StoreResponse])
async def list_stores(
    status: Optional[str] = Query(None, pattern="^(active|inactive)$", description="Filter by status"),
    search: Optional[str] = Query(None, description="Search stores by name"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all stores. Optionally filter by status or search by name."""
    stores = await StoreService.get_stores(db, status_filter=status, search=search)
    return [StoreResponse.model_validate(s) for s in stores]


@router.patch("/{store_id}/status", response_model=StoreResponse)
async def update_store_status(
    store_id: UUID,
    data: StoreStatusUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Activate or deactivate a store.  Body: `{ \"status\": \"active\" | \"inactive\" }`"""
    store = await StoreService.update_store_status(db, store_id, data)
    return StoreResponse.model_validate(store)


@router.get("/{store_id}", response_model=StoreResponse)
async def get_store(
    store_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a store by ID."""
    store = await StoreService.get_store(db, store_id)
    return StoreResponse.model_validate(store)


@router.put("/{store_id}", response_model=StoreResponse)
async def update_store(
    store_id: UUID,
    data: StoreUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update a store's details."""
    store = await StoreService.update_store(db, store_id, data)
    return StoreResponse.model_validate(store)


@router.delete("/{store_id}")
async def delete_store(
    store_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a store."""
    return await StoreService.delete_store(db, store_id)

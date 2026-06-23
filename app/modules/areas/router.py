"""Area API routes - manage independent named areas."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.areas.schemas import AreaCreate, AreaUpdate, AreaResponse
from app.modules.areas.service import AreaService

router = APIRouter(prefix="/api/areas", tags=["Areas"])


@router.post("", response_model=AreaResponse, status_code=201)
async def create_area(
    data: AreaCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new independent area."""
    area = await AreaService.create_area(db, data)
    return AreaResponse.model_validate(area)


@router.get("", response_model=List[AreaResponse])
async def list_areas(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List areas - used to populate the area dropdown when adding a camera."""
    areas = await AreaService.get_areas(db)
    return [AreaResponse.model_validate(a) for a in areas]


@router.get("/{area_id}", response_model=AreaResponse)
async def get_area(
    area_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get an area by ID."""
    area = await AreaService.get_area(db, area_id)
    return AreaResponse.model_validate(area)


@router.put("/{area_id}", response_model=AreaResponse)
async def update_area(
    area_id: UUID,
    data: AreaUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update an area's name."""
    area = await AreaService.update_area(db, area_id, data)
    return AreaResponse.model_validate(area)


@router.delete("/{area_id}")
async def delete_area(
    area_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete an area."""
    return await AreaService.delete_area(db, area_id)
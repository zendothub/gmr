"""Debug detection API routes."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.debug.schemas import (
    ActiveTracksRealtimeResponse, PaginatedUniquePersonsResponse, PaginatedTracksResponse
)
from app.modules.debug.service import DebugService


router = APIRouter(prefix="/api/v2/debug", tags=["Debug"])


@router.get("/active-tracks", response_model=ActiveTracksRealtimeResponse)
async def get_active_tracks(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get all active tracks currently in memory across all cameras.
    
    **Returns:**
    - Summary of active, identified, and inactive track counts
    - List of active tracks with current quality, face status, and ReID scores
    """
    return await DebugService.get_active_tracks(db)


@router.get("/unique-persons", response_model=PaginatedUniquePersonsResponse)
async def get_unique_persons(
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None),
    gender: Optional[str] = Query(None, pattern="^(M|F)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get all unique person identities with basic statistics (paginated).

    - **search**: free-text search across gender, age_group, and UUID
    - **gender**: filter by gender ("M" or "F"), default shows all
    """
    return await DebugService.get_unique_persons(db, page=page, size=size, search=search, gender=gender)


@router.get("/unique-persons/{person_id}/tracks", response_model=PaginatedTracksResponse)
async def get_unique_person_tracks(
    person_id: UUID,
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get all track sessions where a person was detected (paginated).
    """
    return await DebugService.get_unique_person_tracks(db, person_id=person_id, page=page, size=size)
